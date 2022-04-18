import torch
import math
import numpy as np

import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


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
    tx = torch.zeros(nB, nA, nGy, nGx)
    ty = torch.zeros(nB, nA, nGy, nGx)
    tw = torch.zeros(nB, nA, nGy, nGx)
    th = torch.zeros(nB, nA, nGy, nGx)
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

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj

            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            # One-hot encoding of label
            tconf[b, best_n, gj, gi] = 1
            tcls[b, best_n, gj, gi] = t_class

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and score > 0.5 and t_class == pred_class:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
