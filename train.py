import torch
import torch.nn as nn
# import torch.nn.functional as F
from util import progress
from modules import Yolo
import numpy as np
from IPython.display import display
import math
import matplotlib.pyplot as plt


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
                

class YoloLoss(nn.Module):
    def __init__(self, num_classes, batch_size, all_anchors, anchor_masks, img_size, num_anchors, device = None):
        super(YoloLoss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.anchors = all_anchors
        self.anchor_masks = anchor_masks
        self.ignore_thres = 0.5

        self.lambda_noobj = 1
        self.lambda_obj = 10
        self.lambda_coord = 1

        self.bce_loss = nn.BCELoss(reduction = "sum")
        self.mse_loss = nn.MSELoss(reduction = "sum")

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anchor_masks[i]], dtype = np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype = np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype = np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)

            # calculate pred - xywh obj cls
            fsize = self.img_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch_size, num_anchors, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch_size, num_anchors, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batch_size, fsize, num_ch, output_id, plot_conf_heatmap = False):
        # target assignment

        tgt_mask = torch.zeros(batch_size, self.num_anchors, fsize, fsize).to(self.device)
        conf_mask = torch.ones(batch_size, self.num_anchors, fsize, fsize).to(self.device)
        # obj_mask = torch.zeros(batch_size, self.num_anchors, fsize, fsize).to(self.device)
        # tgt_scale = torch.zeros(batch_size, self.num_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batch_size, self.num_anchors, fsize, fsize, num_ch).to(self.device)

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        # if there are 6 labels on the first image and 2 of those labels are small and the output_id
        # corresponds to the small object detector's id, then num_labels_total[0] = 6 and after the
        # for loop, num_labels = 2
        num_labels_total = (labels.sum(dim = 2) > 0).sum(dim = 1)  # number of objects
        num_labels = 0
        num_correct = 0
        for b in range(batch_size):
            n = int(num_labels_total[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU = True)
            mask_begin, mask_end = self.anchor_masks[output_id][0], self.anchor_masks[output_id][1] + 1
            anchor_ious_masked = anchor_ious_all[:, mask_begin:mask_end]

            best_n_all = anchor_ious_all.argmax(dim = 1)
            best_n = best_n_all % int(len(self.anchors) / 3)
            best_n_mask = torch.isin(best_n_all, torch.Tensor(self.anchor_masks[output_id]))
            
            num_labels_img = best_n_mask.sum().item()
            num_labels += num_labels_img
            if num_labels == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # pred_ious = bboxes_iou(pred[b, ..., :4].view(-1, 4), truth_box, xyxy = False)
            # pred_best_iou, _ = pred_ious.max(dim = 1)
            # pred_best_iou = (pred_best_iou > self.ignore_thres)
            # pred_best_iou = pred_best_iou.view(pred[b, ..., :4].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            # noobj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]

                    conf_mask[b, anchor_ious_masked[ti] > self.ignore_thres, i, j] = 0

                    conf_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    # tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

                    ciou = bboxes_iou(truth_box[ti].unsqueeze(0), pred[b, a, j, i, :4].unsqueeze(0), xyxy = False, CIoU = True)
                    obj_score = pred[b, a, j, i, 4]
                    pred_cls = torch.argmax(pred[b, a, j, i, 5:])
                    t_cls = labels[b, ti, 4]
                    # print(f"prediction\tbox: {pred[b, a, j, i, :4]}, class: {pred_cls}, obj_score: {obj_score}")
                    # print(f"target\t{truth_box[ti]}, class: {t_cls}")
                    # print(f"ciou:{ciou}\tobj_score:{obj_score},pred_cls:{pred_cls},t_cls{t_cls}")
                    if ciou > 0.5 and obj_score > 0.5 and pred_cls == t_cls:
                        num_correct += 1

        if plot_conf_heatmap:
            for img_idx in range(2):
                fig = plt.figure(figsize=(6, 4))
                w, h = 2, 2

                anchor_targets = target[img_idx]
                fig.suptitle(f"output id: {output_id}")
                for i, anchor_target in enumerate(anchor_targets):
                    confs = anchor_target[..., 4]
                    img = confs.cpu().detach().numpy()
                    fig.add_subplot(h, w, i + 1)
                    plt.imshow(img, vmin = 0, vmax = 1)
                
                anchor_preds = pred[img_idx]
                for i, anchor_pred in enumerate(anchor_preds):
                    confs = anchor_pred[..., 4]
                    img = confs.cpu().detach().numpy()
                    fig.add_subplot(h, w, w + i + 1)
                    plt.imshow(img, vmin = 0, vmax = 1)

                plt.show()

        return conf_mask, tgt_mask, target, num_labels, num_correct

    def forward(self, xin, labels = None, plot_conf_heatmap = False, eval = False):
        # losses for each stage in a 3 element tensor
        loss_xy = torch.zeros(3)
        loss_wh = torch.zeros(3)
        loss_conf = torch.zeros(3)
        loss_cls = torch.zeros(3)
        preds = []
        num_labels_total, num_proposals_total, num_correct_total = 0, 0, 0
        for output_id, output in enumerate(xin):
            batch_size = output.shape[0]
            fsize = output.shape[2]
            num_ch = 5 + self.num_classes

            output = output.view(batch_size, self.num_anchors, num_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2).contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:num_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:num_ch]])

            pred = output.clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            num_proposals = (pred[..., 4] > 0.5).sum().item()

            conf_mask, tgt_mask, target, num_labels, num_correct = self.build_target(
                pred = pred.detach(),
                labels = labels.detach(),
                batch_size = batch_size,
                fsize = fsize,
                num_ch = num_ch,
                output_id = output_id,
                plot_conf_heatmap = plot_conf_heatmap
            )

            tgt_mask = tgt_mask.type(torch.ByteTensor).bool()
            conf_mask = conf_mask.type(torch.ByteTensor).bool()
            
            obj_mask = tgt_mask
            noobj_mask = conf_mask ^ tgt_mask

            # x and y loss
            loss_xy[output_id] = self.lambda_coord * self.mse_loss(input = output[..., :2][tgt_mask],
                                                                   target = target[..., :2][tgt_mask])

            # width and height loss
            loss_wh[output_id] = self.lambda_coord * self.mse_loss(input = output[..., 2:4][tgt_mask],
                                                                   target = target[..., 2:4][tgt_mask])
            
            # confidence loss
            loss_obj = self.bce_loss(input = output[..., 4][obj_mask],
                                     target = target[..., 4][obj_mask])
            loss_noobj = self.bce_loss(input = output[..., 4][noobj_mask],
                                       target = target[..., 4][noobj_mask])
            loss_conf[output_id] = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
            
            # classification loss
            loss_cls[output_id] = self.bce_loss(input = output[..., 5:][tgt_mask],
                                                target = target[..., 5:][tgt_mask])

            # add num_labels, num_proposals, and num_correct to their respective totals
            num_labels_total += num_labels
            num_proposals_total += num_proposals
            num_correct_total += num_correct

            if eval:
                pred[..., :4] = pred[..., :4] * self.strides[output_id]
                preds.append(pred.view(batch_size, -1, num_ch))

        if eval:
            return torch.cat(preds, dim = 1)
        else:
            loss = loss_xy.sum() + loss_wh.sum() + loss_conf.sum() + loss_cls.sum()
            return loss, loss_xy.sum(), loss_wh.sum(), loss_conf.sum(), loss_cls.sum(), num_labels_total, num_proposals_total, num_correct_total


def init_model(num_classes, anchors, anchor_masks, model_img_size, device):
    model = Yolo(num_classes, anchors, anchor_masks, model_img_size).to(device)
    return model


def train(model, device, dataloader, num_classes, batch_size, minibatch_size, lr = 0.001, num_epochs = 15):
    # set to training mode
    model.train()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = 1e-4)

    # initialize progress bar
    bar = display(progress(0, len(dataloader)), display_id = True)

    # loss criterion
    criterion = YoloLoss(
        num_classes = num_classes,
        batch_size = minibatch_size,
        all_anchors = model.anchors,
        anchor_masks = model.anchor_masks,
        img_size = model.img_size,
        num_anchors = len(model.anchors) // 3,
        device = device
    )

    steps = batch_size // minibatch_size

    # training loop
    for epoch in range(num_epochs):
        running_losses = np.zeros(5)
        num_labels_epoch, num_proposals_epoch, num_correct_epoch = 0, 0, 0
        num_labels_batch, num_proposals_batch, num_correct_batch = 0, 0, 0

        optimizer.zero_grad()
        for i, (_, imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            targets.requires_grad = False

            losses_batch = np.zeros(5)
            
            prediction = model(imgs, targets)
            loss, loss_xy, loss_wh, loss_conf, loss_cls, num_labels_minibatch, num_proposals_minibatch, num_correct_minibatch = criterion(prediction, targets, plot_conf_heatmap = False)
            loss.backward()
            
            losses_minibatch = np.array([
                loss.cpu().detach().item(),
                loss_xy.cpu().detach().item(),
                loss_wh.cpu().detach().item(),
                loss_conf.cpu().detach().item(),
                loss_cls.cpu().detach().item()
            ])

            # accumulate losses
            losses_batch += losses_minibatch
            # accumulate num_labels, num_proposals and num_correct
            num_labels_batch += num_labels_minibatch
            num_proposals_batch += num_proposals_minibatch
            num_correct_batch += num_correct_minibatch

            if (i + 1) % steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # print("Losses: loss %.2f, loss_xy %.2f, loss_wh %.2f, loss_conf %.2f, loss_cls %.2f, truth boxes %d, proposals %d, correct %d"
                #     % (
                #         losses_batch[0],
                #         losses_batch[1],
                #         losses_batch[2],
                #         losses_batch[3],
                #         losses_batch[4],
                #         num_labels_batch,
                #         num_proposals_batch,
                #         num_correct_batch
                #     )
                # )

                running_losses += losses_batch

                num_labels_epoch += num_labels_batch
                num_proposals_epoch += num_proposals_batch
                num_correct_epoch += num_correct_batch

                num_labels_batch, num_proposals_batch, num_correct_batch = 0, 0, 0
            
            if bar is not None:
                bar.update(progress(i + 1, len(dataloader)))

        num_batches = len(dataloader) * minibatch_size / batch_size
        recall = num_correct_epoch / num_labels_epoch if num_labels_epoch else 1
        precision = num_correct_epoch / num_proposals_epoch if num_proposals_epoch else 0
        print("[ Epoch %d/%d ]\t" % (epoch + 1, num_epochs), end = "")
        print("Losses: loss %.2f, loss_xy %.2f, loss_wh %.2f, loss_conf %.2f, loss_cls %.2f, recall %.2f %%, precision: %.2f %%"
            % (
                running_losses[0] / num_batches,
                running_losses[1] / num_batches,
                running_losses[2] / num_batches,
                running_losses[3] / num_batches,
                running_losses[4] / num_batches,
                recall * 100,
                precision * 100
            )
        )

def save(model, path):
    torch.save(model.state_dict(), path)
