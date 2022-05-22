import torch
from torch import nn
from torch import FloatTensor
import numpy as np
import matplotlib.pyplot as plt


class YoloLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchor_mask = [], num_classes = 0, anchors = [], num_anchors = 1, stride = 32, model_out = False, img_size = 416):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.stride = stride

        self.model_out = model_out

        self.img_size = img_size
        self.stride = stride
        self.anchors = [anchors[i] for i in anchor_mask]

    def forward(self, output):
        if self.training:
            return output
    

        device = torch.device("cuda" if output.is_cuda else "cpu")

        batch_size = output.shape[0]
        num_anchors = len(self.anchor_mask)
        fsize = output.shape[2]
        num_ch = 5 + self.num_classes

        grid_x = torch.arange(fsize, dtype = torch.float).repeat(batch_size, num_anchors, fsize, 1).to(device)
        grid_y = torch.arange(fsize, dtype = torch.float).repeat(batch_size, num_anchors, fsize, 1).permute(0, 1, 3, 2).to(device)

        anchors_grid = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        anchor_w = anchors_grid[:, 0:1].view((1, num_anchors, 1, 1)).to(device)
        anchor_h = anchors_grid[:, 1:2].view((1, num_anchors, 1, 1)).to(device)

        output = output.view(batch_size, self.num_anchors, num_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()
    
        pred = output.clone()
        pred[..., 0] = (torch.sigmoid(output[..., 0]) + grid_x) * self.stride
        pred[..., 1] = (torch.sigmoid(output[..., 1]) + grid_y) * self.stride
        pred[..., 2] = (torch.exp(output[..., 2]) * anchor_w) * self.stride
        pred[..., 3] = (torch.exp(output[..., 3]) * anchor_h) * self.stride
        pred[..., 4:] = torch.sigmoid(pred[..., 4:])

        fig = plt.figure(figsize = (4, 2))
        fig.suptitle(f"fsize: {fsize}")
        for i, anchor_pred in enumerate(pred[0]):
            confs = anchor_pred[..., 4]
            heatmap = confs.cpu().detach().numpy()
            fig.add_subplot(1, 2, i + 1)
            plt.imshow(heatmap, vmin = 0, vmax = 1)
        plt.show()

        return pred.view(batch_size, -1, num_ch)