import math
import torch
from torch import nn
from util import build_targets


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
            (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 /
                                                              h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


img_dim = (384, 512)


class YoloLayer(nn.Module):
    """Detection layer"""
    def __init__(self, all_anchors, anchor_mask, num_classes):
        super(YoloLayer, self).__init__()
        self.all_anchors = all_anchors
        self.anchor_mask = anchor_mask
        self.anchors = all_anchors[anchor_mask[0]:anchor_mask[-1] + 1]
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim # (H, W)
        self.ignore_thres = 0.5
        self.lambda_coord = 1

    def forward(self, x, targets = None):
        nB = x.size(0)
        nA = len(self.anchors)
        nGy = x.size(2)
        nGx = x.size(3)

        # height (and also width) of a grid cell in pixels
        stride = self.image_dim[0] / nGy

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # Reshape x: [batchSize x nGy x nGx x numAnchors*(5+numClass)] -> [batchSize x numAnchors x nGy x nGx x (5+numClass)]
        prediction = x.view(nB, nA, self.bbox_attrs, nGy, nGx).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_class = prediction[..., 5:]  # Class

        # Calculate offsets for each grid
        grid_x = torch.arange(nGx).repeat(nGy, 1).view(
            [1, 1, nGy, nGx]).type(FloatTensor)
        grid_y = torch.arange(nGy).repeat(nGx, 1).t().view(
            [1, 1, nGy, nGx]).type(FloatTensor)

        all_scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.all_anchors])
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.detach() + grid_x
        pred_boxes[..., 1] = y.detach() + grid_y
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_h

        # Training
        if targets is not None:
            if x.is_cuda:
                # self.ciou_loss = self.ciou_loss.cuda()
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tbox, tconf, tcls = build_targets(
                pred_boxes = pred_boxes.cpu().detach(),
                pred_conf = pred_conf.cpu().detach(),
                pred_classes = pred_class.cpu().detach(),
                target = targets.cpu().detach(),
                all_anchors = all_scaled_anchors.cpu().detach(),
                anchors = scaled_anchors.cpu().detach(),
                anchor_mask = self.anchor_mask,
                grid_size_y = nGy,
                grid_size_x = nGx,
                ignore_thres = self.ignore_thres,
            )

            nProposals = int((pred_conf > 0.5).sum().item())

            # Handle masks
            mask = mask.type(ByteTensor).bool()
            conf_mask = conf_mask.type(ByteTensor).bool()

            # Handle target variables
            tbox = tbox.type(FloatTensor)
            tconf = tconf.type(FloatTensor)
            tcls = tcls.type(LongTensor)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask

            return (
                (x[mask], tbox[mask][:, 0]),
                (y[mask], tbox[mask][:, 1]),
                (w[mask], tbox[mask][:, 2]),
                (h[mask], tbox[mask][:, 3]),
                (pred_conf[conf_mask_true], tconf[conf_mask_true]),
                (pred_class[mask], tcls[mask]),
                nGT,
                nProposals,
                nCorrect
            )

        else:
            # If not in training phase return predictions
            output = torch.cat((
                pred_boxes.view(nB, -1, 4) * stride,
                pred_conf.view(nB, -1, 1),
                pred_class.view(nB, -1, self.num_classes)
            ), -1)

        return output
