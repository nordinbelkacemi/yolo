import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import glob
import os

from torch.utils.data import Dataset
from PIL import Image
from IPython.display import HTML
from math import pi


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=(384, 512)):
        self.img_files = [list_path +
                          img for img in glob.glob1(list_path, "*.png")]
        self.label_files = [path.replace('images', 'labels').replace(
            '.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        Returns the tuple (img_path, input_img, filled_labels)
            - img_path: path to the image
            - input_img: the image itself as a pytorch tensor
            - filled_labels: a pytorch tensor (with shape (50, 5)) of labels 
        """

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')

        input_img = self.transform(img)

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

        # Fill matrix
        filled_labels = np.zeros((50, 5))

        if labels is not None:
            filled_labels[range(len(labels))[:50]] = labels[:50]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_anchor_ious(w, h, anchors):
    # Get shape of gt box
    gt_box = torch.FloatTensor(np.array([0, 0, w, h])).unsqueeze(0)
    # Get shape of anchor box
    anchor_shapes = torch.FloatTensor(np.concatenate((
        np.zeros((len(anchors), 2)), np.array(anchors)
    ), 1))
    # Calculate iou between gt and anchor shapes
    return bbox_iou(gt_box, anchor_shapes)


def build_targets(pred_boxes, pred_conf, pred_classes, target, anchors, num_anchors, num_classes, grid_size_y, grid_size_x, ignore_thres, img_dim):
    """
    pred_boxes: shape is (batch_size, num_anchor_boxes, grid_y, grid_x, 4) -> for each element in a batch, there are 6 12x16 grids of 4 dimensional vectors (x, y, w, h). x, y, w, and h are in "grid coordinates" (x = 12.41 means 11-th grid box and 0.41 in the x direction)
    pred_conf: shape is (batch_size, num_anchor_boxes, grid_y, grid_x) -> for each element in a batch, there are 6 12x16 grids of floats representing the prediction confidence (between 0 and 1)
    pred_classes: shape is (batch_size, num_anchor_boxes, grid_y, grid_x, 4) -> for each element in a batch, there are 6 12x16 grids of 4 dimensional vectors (s_0, s_1, s_2, s_3), where s_i is a score for class i.
    target: shape is (batch_size, 50, 5) -> for each element in a batch, there is a 50x5 matrix, where each row contains a class index, x, y, w and h.
    anchors: [[16, 8], [23, 103], [28, 23], [56, 47], [96, 123], [157, 248]]
    num_anchors: int
    num_classes: int
    grid_size_y: int
    grid_size_x: int
    ignore_thres: float
    img_dim: (384, 512)
    """
    # print(pred_boxes.size(), pred_conf.size(), pred_classes.size(), target.size(), anchors.size(), num_anchors, num_classes, grid_size_y, grid_size_x, ignore_thres, img_dim)
    #     -> output: torch.Size([32, 6, 12, 16, 4]) torch.Size([32, 6, 12, 16]) torch.Size([32, 6, 12, 16, 4]) torch.Size([32, 50, 5]) torch.Size([6, 2]) 6 4 12 16 0.5 (384, 512)

    nB = target.size(0)  # batch_size
    nA = num_anchors
    nC = num_classes
    nGx = grid_size_x
    nGy = grid_size_y

    # Masks: mask is one for the best bounding box
    # Conf mask is one for BBs, where the confidence is enforced to match target
    mask = torch.zeros(nB, nA, nGy, nGx)
    conf_mask = torch.ones(nB, nA, nGy, nGx)

    # Target values for x, y, w, h and confidence and class
    tbox = torch.zeros(nB, nA, nGy, nGx, 4)
    tconf = torch.ByteTensor(nB, nA, nGy, nGx).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nGy, nGx).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1

            # Convert to position relative to box
            t_class = target[b, t, 0].long()
            gx = target[b, t, 1] * nGx
            gy = target[b, t, 2] * nGy
            gw = target[b, t, 3] * nGx
            gh = target[b, t, 4] * nGy

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            # Get IoU values between target and anchors
            anch_ious = get_anchor_ious(gw, gh, anchors)

            # Where the overlap is larger than threshold set conf_mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)

            # Create ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            pred_class = torch.argmax(pred_classes[b, best_n, gj, gi])

            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            # Target box in grid coordinates
            tbox[b, best_n, gj, gi, :] = gt_box[:]

            # One-hot encoding of label
            tconf[b, best_n, gj, gi] = 1
            tcls[b, best_n, gj, gi] = t_class

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2 = False)
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and score > 0.5 and t_class == pred_class:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tbox, tconf, tcls

class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()
    
    def forward(self, b1, b2):
        # central point coordinates + width and height
        b1_xc, b1_yc, b1_w, b1_h = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
        b2_xc, b2_yc, b2_w, b2_h = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
        
        # corner point coordinates
        b1_x1, b1_y1 = b1_xc - b1_w / 2, b1_yc - b1_h / 2
        b1_x2, b1_y2 = b1_xc + b1_w / 2, b1_yc + b1_h / 2
        b2_x1, b2_y1 = b2_xc - b2_w / 2, b2_yc - b2_h / 2
        b2_x2, b2_y2 = b2_xc + b2_w / 2, b2_yc + b2_h / 2

        # iou
        iou = bbox_iou(b1, b2, x1y1x2y2=False)

        # rho is the distance between the central points of the two boxes
        rho_squared = (b2_xc - b1_xc) ** 2 + (b2_yc - b1_yc) ** 2
        
        # c is the diagonal length of the smallest enclosing box covering two boxes
        x_top_left = torch.minimum(b1_x1, b2_x1)
        y_top_left = torch.minimum(b1_y1, b2_y1)
        x_bottom_right = torch.maximum(b1_x2, b2_x2)
        y_bottom_right = torch.maximum(b1_y2, b2_y2)
        c_squared = (x_bottom_right - x_top_left) ** 2 + (y_bottom_right - y_top_left) ** 2

        # v measures the consistency of aspect ratio
        v = 4 / (pi ** 2) * (torch.arctan(b1_w / b1_h) - torch.arctan(b2_w / b2_h)) ** 2

        # alpha is a positive trade-off parameter
        alpha = v / (1 - iou + v)

        # CIoU loss function
        loss = 1 - iou + rho_squared / c_squared + alpha * v

        return torch.mean(loss)

def print_losses(dataloader, all_losses, idx):
    """
    Args:
        all_losses: numpy array of all the losses. Its shape is (3, n_loss),
        where n_loss is the number of loss metrics that the YoloLayer returns
        when training.
        
        idx: an integer used to select the detection stage.
    """
    stage_losses = all_losses[idx]
    print("\tLosses %d: ciou %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f"
        % (
            idx + 1,
            stage_losses[1] / float(len(dataloader)),
            stage_losses[2] / float(len(dataloader)),
            stage_losses[3] / float(len(dataloader)),
            stage_losses[0] / float(len(dataloader)),
            stage_losses[4] / float(len(dataloader)),
            stage_losses[5] / float(len(dataloader))
        )
    )


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def non_max_suppression(prediction, num_classes, conf_thres = 0.5, nms_thres = 0.4):
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = len(prediction)
    output = [None for _ in range(batch_size)]
    # Run non max suppression algorithm for each image
    for image_idx, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        # If none remain, process next image
        if len(image_pred) == 0:
            continue
        
        # Get score and class with the highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], dim = 1, keepdim = True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), dim = 1)

        # unique_labels is a 1 dimensional tensor containing the unique classes predicted:
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        
        # Iterate through all predicted classes
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness score
            _, conf_sort_idx = torch.sort(detections_class[:, 4], descending = True)
            detections_class = detections_class[conf_sort_idx]

            # Perform non max suppression
            max_detections = []
            while len(detections_class) != 0:
                # Get detection with the highest confidence
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all other boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
            
            max_detections = torch.cat(max_detections)
            # Add max detections to outputs
            output[image_idx] = (
                max_detections if output[image_idx] is None else torch.cat((output[image_idx], max_detections))
            )

    formatted_output = output.new(output.shape)

    formatted_output[:, :, 1] = output[:, :, 0] + (output[:, :, 2] - output[:, :, 0] / 2)
    formatted_output[:, :, 2] = output[:, :, 1] + (output[:, :, 3] - output[:, :, 1] / 2)
    formatted_output[:, :, 3] = output[:, :, 2] - output[:, :, 0]
    formatted_output[:, :, 4] = output[:, :, 3] - output[:, :, 1]
    formatted_output[:, :, 0] = output[:, :, -1]
    return formatted_output
