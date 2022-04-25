import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from modules import Yolo
from util import non_max_suppression
import cv2
from matplotlib import pyplot as plt

# img_path = "Finetune/test/271.png"
# label_path = "Finetune/test/271.txt"
# num_classes = 4
# anchors = [[16, 8], [23, 103], [28, 23], [56, 47], [96, 123], [157, 248]]






# def run_demo(img_path, label_path, num_classes, anchors, using_cuda, saved_model_params_path = None):
#     # read PIL image, convert it to RGB, then into a pytorch tensor
#     img = (transforms.ToTensor())(Image.open(img_path).convert('RGB'))

#     # read label from text, then write the data into a 50 by 5 pytorch tensor
#     label = np.loadtxt(label_path).reshape(-1, 5)
#     label[:, 1], label[:, 3] = label[:, 1] * 512, label[:, 3] * 512
#     label[:, 2], label[:, 4] = label[:, 2] * 384, label[:, 4] * 384

#     # model
#     model = Yolo(anchors, num_classes)
#     if saved_model_params_path is not None:
#         model.load_state_dict(torch.load(saved_model_params_path))
#     model.eval()

#     # prediction
#     input = img.unsqueeze(0)
#     if using_cuda:
#         input = input.cuda()
#     detections = model(input)
#     detections = non_max_suppression(detections, 4)

#     # display prediction
#     img = cv2.imread(img_path)
#     for bbox in detections[0]:
#         p1 = min(bbox[0], 0), max(bbox[2], 512)
#         p2 = min(bbox[1], 0), max(bbox[3], 384)
#         cv2.rectangle(img, p1, p2, (255, 0, 0))
#     cv2.imshow(image)
