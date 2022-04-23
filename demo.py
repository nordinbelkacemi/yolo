import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from modules import Yolo
from util import non_max_suppression

# img_path = "Finetune/test/271.png"
# label_path = "Finetune/test/271.txt"
# num_classes = 4
# anchors = [[16, 8], [23, 103], [28, 23], [56, 47], [96, 123], [157, 248]]

def run_demo(img_path, label_path, num_classes, anchors):
    # read PIL image, convert it to RGB, then into a pytorch tensor
    img = (transforms.ToTensor())(Image.open("Finetune/test/181.png").convert('RGB'))

    # read label from text, then write the data into a 50 by 5 pytorch tensor
    label = np.loadtxt(label_path).reshape(-1, 5)
    label[:, 1], label[:, 3] = label[:, 1] * 512, label[:, 3] * 512
    label[:, 2], label[:, 4] = label[:, 2] * 384, label[:, 4] * 384
    # filled_labels = np.zeros((50, 5))
    # if label is not None:
    #     filled_labels[range(len(label))[:50]] = label[:50]
    #     filled_labels = torch.from_numpy(filled_labels)

    # model
    model = Yolo(anchors, num_classes)

    # prediction
    input = img.unsqueeze(0)
    detections = model(input)
    detections = non_max_suppression(detections, 4)

    print(detections)
    print(label)

