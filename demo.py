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

def run_demo(img_path, label_path, num_classes, anchors, using_cuda, saved_model_params_path = None):
    # read PIL image, convert it to RGB, then into a pytorch tensor
    img = (transforms.ToTensor())(Image.open(img_path).convert('RGB'))

    # read label from text, then write the data into a 50 by 5 pytorch tensor
    label = np.loadtxt(label_path).reshape(-1, 5)
    label[:, 1], label[:, 3] = label[:, 1] * 512, label[:, 3] * 512
    label[:, 2], label[:, 4] = label[:, 2] * 384, label[:, 4] * 384

    # model
    model = Yolo(anchors, num_classes)
    if saved_model_params_path is not None:
        model.load_state_dict(torch.load(saved_model_params_path))
    model.eval()

    # prediction
    input = img.unsqueeze(0)
    if using_cuda:
        input = input.cuda()
    detections = model(input)
    detections = non_max_suppression(detections, 4)

    # printing
    print(detections[0])
    print(label)

