import torch
import torch.nn as nn
import numpy as np
from IPython.display import HTML
from math import pi, log


def bbox_iou(box1, box2, x1y1x2y2 = True):
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
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

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


def build_targets(pred_boxes, pred_conf, pred_classes, target, all_anchors, anchors, anchor_mask, grid_size_y, grid_size_x, ignore_thres):
    """
    pred_boxes: shape is (batch_size, num_anchor_boxes, grid_y, grid_x, 4) -> for each element in a batch, there are 6 12x16 grids of 4 dimensional vectors (x, y, w, h). x, y, w, and h are in "grid coordinates" (x = 12.41 means 13-th grid box and 0.41 in the x direction)
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

    nB = target.size(0)
    nA = len(anchors)
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

    nGT, nCorrect = 0, 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue

            # Convert to position relative to box
            gx = target[b, t, 1] * nGx
            gy = target[b, t, 2] * nGy
            gw = target[b, t, 3] * nGx
            gh = target[b, t, 4] * nGy

            # Get IoU values between target and anchors
            all_anch_ious = get_anchor_ious(gw, gh, all_anchors)
            anch_ious = all_anch_ious[anchor_mask[0]:anchor_mask[-1] + 1]

            # Find the best matching anchor box and ignore if it is not in the anchor_mask
            best_n = np.argmax(all_anch_ious)
            if best_n in anchor_mask:
                nGT += 1
                best_n = best_n % nA
            else:
                continue

            # target class
            t_class = target[b, t, 0].long()

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            # Where the overlap is larger than threshold, set conf_mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # Create ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            pred_class = torch.argmax(pred_classes[b, best_n, gj, gi])

            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            # x and y coordinates + width and height
            tbox[b, best_n, gj, gi, 0] = gx - gi
            tbox[b, best_n, gj, gi, 1] = gy - gj
            tbox[b, best_n, gj, gi, 2] = log(gw / anchors[best_n][0] + 1e-16)
            tbox[b, best_n, gj, gi, 3] = log(gh / anchors[best_n][1] + 1e-16)

            # One-hot encoding of label
            tconf[b, best_n, gj, gi] = 1
            tcls[b, best_n, gj, gi] = t_class

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2 = False)
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and score > 0.5 and t_class == pred_class:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tbox, tconf, tcls


def print_losses(dataloader, losses, epoch, num_epochs):
    print("[ Epoch %d/%d ]\t" % (epoch + 1, num_epochs), end = "")
    print("Losses: total %f, x %f, y %f, w %f, h %f, conf %f, cls %f, recall: %.5f, precision: %.5f"
        % (
            losses[0] / float(len(dataloader)),
            losses[1] / float(len(dataloader)),
            losses[2] / float(len(dataloader)),
            losses[3] / float(len(dataloader)),
            losses[4] / float(len(dataloader)),
            losses[5] / float(len(dataloader)),
            losses[6] / float(len(dataloader)),
            losses[7] / float(len(dataloader)),
            losses[8] / float(len(dataloader))
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
                # Get the IOUs for all other boxes of the given class with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
            
            max_detections = torch.cat(max_detections)
            # Add max detections to outputs
            output[image_idx] = (
                max_detections if output[image_idx] is None else torch.cat((output[image_idx], max_detections))
            )

    return output
