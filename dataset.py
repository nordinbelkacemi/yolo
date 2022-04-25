import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

# image 307 in coco is a good example

def get_square_padding(img):
    max_dim = max(img.size)
    pad_left, pad_top = [(max_dim - s) // 2 for s in img.size]
    pad_right, pad_bottom = [max_dim - (s + pad) for s, pad in zip(img.size, [pad_left, pad_top])]
    return (pad_left, pad_top, pad_right, pad_bottom)

class SquarePad:
    def __call__(self, img):
        padding = get_square_padding(img)
        return F.pad(img, padding, 0, 'constant')

class ListDataset(Dataset):
    def __init__(self, list_path, model_input_img_size = (384, 512)):
        self.img_files = [list_path + img for img in glob.glob1(list_path, "*.png")]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = model_input_img_size
        self.transform = transforms.Compose([
            transforms.Resize(model_input_img_size),
            transforms.ToTensor()
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
