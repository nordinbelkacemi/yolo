import torch
from torch import nn
import torch.nn.functional as F
from yolo_layer import YoloLayer


# ANCHORS = [
#     [12, 16],
#     [19, 36],
#     [40, 28],
#     [36, 75],
#     [76, 55],
#     [72, 146],
#     [142, 110],
#     [192, 243],
#     [459, 401]
# ]


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn = True, bias = False):
        super(ConvBnActivation, self).__init__()
        padding = kernel_size // 2
        
        self.module_list = nn.ModuleList()

        # conv
        if bias:
            self.module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            self.module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))

        # batch norm
        if bn:
            self.module_list.append(nn.BatchNorm2d(out_channels))
        
        # activation
        if activation == "mish":
            self.module_list.append(Mish())
        elif activation == "relu":
            self.module_list.append(nn.ReLU(inplace = True))
        elif activation == "leaky":
            self.module_list.append(nn.LeakyReLU(0.1, inplace = True))
        else:
            print(f"Invalid activation function: \"{activation}\"")

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResBlock, self).__init__()

        self.module_list = nn.ModuleList()
        for _ in range(num_blocks):
            module = nn.Sequential(
                ConvBnActivation(in_channels, in_channels // 2, 1, 1, "mish"),
                ConvBnActivation(in_channels // 2, in_channels, 3, 1, "mish")
            )
            self.module_list.append(module)
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(3, 32, 3, 1, "mish")
        self.conv2 = ConvBnActivation(32, 64, 3, 2, "mish")
        self.conv3 = ConvBnActivation(64, 64, 1, 1, "mish")
        self.conv4 = ConvBnActivation(64, 64, 1, 1, "mish")

        self.res = ResBlock(in_channels = 64, num_blocks = 1)
        self.conv5 = ConvBnActivation(64, 64, 1, 1, "mish")
        self.conv6 = ConvBnActivation(128, 64, 1, 1, "mish")

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)

        r = self.res(x4)
        x5 = self.conv5(r)

        x5 = torch.cat((x5, x3), dim = 1)
        x6 = self.conv6(x5)
        return x6


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(64, 128, 3, 2, "mish")
        self.conv2 = ConvBnActivation(128, 64, 1, 1, "mish")
        self.conv3 = ConvBnActivation(128, 64, 1, 1, "mish")

        self.res = ResBlock(in_channels = 64, num_blocks = 2)
        self.conv4 = ConvBnActivation(64, 64, 1, 1, "mish")
        self.conv5 = ConvBnActivation(128, 128, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.res(x3)
        x4 = self.conv4(r)

        x4 = torch.cat((x4, x2), dim = 1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(128, 256, 3, 2, "mish")
        self.conv2 = ConvBnActivation(256, 128, 1, 1, "mish")
        self.conv3 = ConvBnActivation(256, 128, 1, 1, "mish")

        self.res = ResBlock(in_channels = 128, num_blocks = 8)
        self.conv4 = ConvBnActivation(128, 128, 1, 1, "mish")
        self.conv5 = ConvBnActivation(256, 256, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.res(x3)
        x4 = self.conv4(r)

        x4 = torch.cat((x4, x2), dim = 1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(256, 512, 3, 2, "mish")
        self.conv2 = ConvBnActivation(512, 256, 1, 1, "mish")
        self.conv3 = ConvBnActivation(512, 256, 1, 1, "mish")

        self.res = ResBlock(in_channels = 256, num_blocks = 8)
        self.conv4 = ConvBnActivation(256, 256, 1, 1, "mish")
        self.conv5 = ConvBnActivation(512, 512, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.res(x3)
        x4 = self.conv4(r)

        x4 = torch.cat((x4, x2), dim = 1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnActivation(512, 1024, 3, 2, "mish")
        self.conv2 = ConvBnActivation(1024, 512, 1, 1, "mish")
        self.conv3 = ConvBnActivation(1024, 512, 1, 1, "mish")

        self.res = ResBlock(in_channels = 512, num_blocks = 4)

        self.conv4 = ConvBnActivation(512, 512, 1, 1, "mish")
        self.conv5 = ConvBnActivation(1024, 1024, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.res(x3)
        x4 = self.conv4(r)

        x4 = torch.cat((x4, x2), dim = 1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

        self.conv1 = ConvBnActivation(1024, 512, 1, 1, "leaky")
        self.conv2 = ConvBnActivation(512, 1024, 3, 1, "leaky")
        self.conv3 = ConvBnActivation(1024, 512, 1, 1, "leaky")

        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size = 5, stride = 1, padding = 5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 9, stride = 1, padding = 9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 13, stride = 1, padding= 13 // 2)

        self.conv4 = ConvBnActivation(2048, 512, 1, 1, "leaky")
        self.conv5 = ConvBnActivation(512, 1024, 3, 1, "leaky")
        self.conv6 = ConvBnActivation(1024, 512, 1, 1, "leaky")
        self.conv7 = ConvBnActivation(512, 256, 1, 1, "leaky")

        self.upsample1 = Upsample()

        self.conv8 = ConvBnActivation(512, 256, 1, 1, "leaky")

        self.conv9 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv10 = ConvBnActivation(256, 512, 3, 1, "leaky")
        self.conv11 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv12 = ConvBnActivation(256, 512, 3, 1, "leaky")
        self.conv13 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv14 = ConvBnActivation(256, 128, 1, 1, "leaky")

        self.upsample2 = Upsample()

        self.conv15 = ConvBnActivation(256, 128, 1, 1, "leaky")

        self.conv16 = ConvBnActivation(256, 128, 1, 1, "leaky")
        self.conv17 = ConvBnActivation(128, 256, 3, 1, "leaky")
        self.conv18 = ConvBnActivation(256, 128, 1, 1, "leaky")
        self.conv19 = ConvBnActivation(128, 256, 3, 1, "leaky")
        self.conv20 = ConvBnActivation(256, 128, 1, 1, "leaky")

    def forward(self, downsample5, downsample4, downsample3):
        x1 = self.conv1(downsample5)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat((m3, m2, m1, x3), dim = 1)

        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        # UP
        up = self.upsample1(x7, downsample4.size())

        x8 = self.conv8(downsample4)
        x8 = torch.cat((x8, up), dim = 1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        up = self.upsample2(x14, downsample3.size())

        x15 = self.conv15(downsample3)
        x15 = torch.cat((x15, up), dim = 1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)

        return x20, x13, x6


class YoloHead(nn.Module):
    def __init__(self, num_out_channels, num_classes, anchors):
        super().__init__()

        # anchors for yolo layers: i.e. if we have 9 anchor boxes, we split them up into 3 arrays of 3 anchor boxes
        assert(len(anchors) % 3 == 0)
        step = len(anchors) // 3
        anchor_mask_sml = [i for i in range(step * 0, step * 1)]
        anchor_mask_med = [i for i in range(step * 1, step * 2)]
        anchor_mask_lrg = [i for i in range(step * 2, step * 3)]


        self.conv1 = ConvBnActivation(128, 256, 3, 1, "leaky")
        self.conv2 = nn.Conv2d(256, num_out_channels, 1)

        self.yolo1 = YoloLayer(anchors, anchor_mask_sml, num_classes)

        self.conv3 = ConvBnActivation(128, 256, 3, 2, "leaky")

        self.conv4 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv5 = ConvBnActivation(256, 512, 3, 1, "leaky")
        self.conv6 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv7 = ConvBnActivation(256, 512, 3, 1, "leaky")
        self.conv8 = ConvBnActivation(512, 256, 1, 1, "leaky")
        self.conv9 = ConvBnActivation(256, 512, 3, 1, "leaky")
        self.conv10 = nn.Conv2d(512, num_out_channels, 1)

        self.yolo2 = YoloLayer(anchors, anchor_mask_med, num_classes)

        self.conv11 = ConvBnActivation(256, 512, 3, 2, "leaky")

        self.conv12 = ConvBnActivation(1024, 512, 1, 1, "leaky")
        self.conv13 = ConvBnActivation(512, 1024, 3, 1, "leaky")
        self.conv14 = ConvBnActivation(1024, 512, 1, 1, "leaky")
        self.conv15 = ConvBnActivation(512, 1024, 3, 1, "leaky")
        self.conv16 = ConvBnActivation(1024, 512, 1, 1, "leaky")
        self.conv17 = ConvBnActivation(512, 1024, 3, 1, "leaky")
        self.conv18 = nn.Conv2d(1024, num_out_channels, 1)

        self.yolo3 = YoloLayer(anchors, anchor_mask_lrg, num_classes)

    def forward(self, input1, input2, input3, targets = None):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        x3 = torch.cat((x3, input2), dim = 1)

        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        x11 = self.conv11(x8)
        x11 = torch.cat((x11, input3), dim = 1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        if targets is not None:
            losses_1 = self.yolo1(x2, targets)
            losses_2 = self.yolo2(x10, targets)
            losses_3 = self.yolo3(x18, targets)
            total_loss = losses_1[0] + losses_2[0] + losses_3[0]
            
            return (
                total_loss,
                losses_1,
                losses_2,
                losses_3
            )
        else:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return torch.cat((y1, y2, y3), dim = 1)


class Yolo(nn.Module):
    def __init__(self, anchors, num_classes):
        super(Yolo, self).__init__()

        num_out_channels = (5 + num_classes) * (len(anchors) // 3)

        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()

        self.neck = Neck()

        self.head = YoloHead(num_out_channels, num_classes, anchors)

    def forward(self, x, targets = None):
        d1 = self.downsample1(x)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6, targets)
        return output
