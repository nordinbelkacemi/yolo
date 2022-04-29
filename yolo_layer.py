import torch
from torch import nn
from util import build_targets

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

        # self.ciou_loss = CIoULoss()  # Coordinate loss
        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')  # Class loss

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
            conf_mask_false = conf_mask ^ mask

            print(x[mask])
            print(tbox[mask][:, 0])
            # loss_box = self.ciou_loss(pred_box[mask], tbox[mask])
            loss_x = self.mse_loss(x[mask], tbox[mask][:, 0])
            loss_y = self.mse_loss(y[mask], tbox[mask][:, 1])
            loss_w = self.mse_loss(w[mask], tbox[mask][:, 2])
            loss_h = self.mse_loss(h[mask], tbox[mask][:, 3])
            loss_conf = 10 * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) \
                        + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = self.ce_loss(pred_class[mask], tcls[mask])

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
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
