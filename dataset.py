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
    return pad_left, pad_top, pad_right, pad_bottom

class SquarePad:
    def __call__(self, img):
        padding = get_square_padding(img)
        return F.pad(img, padding, 0, 'constant')

class ListDataset(Dataset):

    def __init__(self, list_path, anchors, anchor_img_size, model_img_size, tiny = False):
        #---------
        #  Data
        #---------
        
        self.img_files = [list_path + img for img in glob.glob1(list_path, "*.png")]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
       
        if tiny:
            # for random selection without replacement, conversion to np array is needed
            img_files = np.array(self.img_files)
            label_files = np.array(self.label_files)
            
            # select 4 elements at random without replacement
            sel_idx = np.random.choice(len(self.img_files), size = 4, replace = False)
            self.img_files = img_files[sel_idx].tolist()
            self.label_files = label_files[sel_idx].tolist()
        
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(model_img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4051, 0.4807, 0.4872),
                                 (0.2181, 0.0336, 0.0453))
        ])

        #---------
        #  Anchors and anchor masks
        #---------
        
        # rescale anchors to the model image size
        rescaled_anchors = np.zeros((len(anchors), 2))
        rescaled_anchors[:, 0] = np.array(anchors)[:, 0] * (model_img_size / anchor_img_size[0])
        rescaled_anchors[:, 1] = np.array(anchors)[:, 1] * (model_img_size / anchor_img_size[1])
        self.anchors = rescaled_anchors.tolist()

        # anchor masks for small, medium and large anchors
        step = len(anchors) // 3
        self.anchor_masks = [
            [i for i in range(0 * step, 1 * step)],
            [i for i in range(1 * step, 2 * step)],
            [i for i in range(2 * step, 3 * step)]
        ]

        self.model_img_size = model_img_size

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
            # labels should have shape (n, 5), where n is the number of objects on the image and the 5 numbers in one row are cls, x, y, w, h.
            labels = np.loadtxt(label_path).reshape(-1, 5)
        num_labels = labels.shape[0]

        # label transformation due to square padding and resizing to 608
        img_w, img_h = img.size
        x, y = np.copy(labels[:, 1]) * img_w, np.copy(labels[:, 2]) * img_h
        w, h = np.copy(labels[:, 3]) * img_w, np.copy(labels[:, 4]) * img_h
        cls = np.copy(labels[:, 0])
    
        pad_left, pad_top, pad_right, pad_bottom = get_square_padding(img)
        sqare_size = img_w + pad_left + pad_right

        x_padded, y_padded = (x + pad_left), (y + pad_top)
        w_padded, h_padded = w, h

        x = x_padded * (self.model_img_size / sqare_size)
        y = y_padded * (self.model_img_size / sqare_size)
        w = w_padded * (self.model_img_size / sqare_size)
        h = h_padded * (self.model_img_size / sqare_size)

        labels_scaled = np.zeros((num_labels, 5))
        labels_scaled[:, 0] = x - w / 2    # x1
        labels_scaled[:, 1] = y - h / 2    # y1
        labels_scaled[:, 2] = x + w / 2    # x2
        labels_scaled[:, 3] = y + h / 2    # y2
        labels_scaled[:, 4] = cls          # cls

        # Fill matrix
        filled_labels = np.zeros((50, 5))
        filled_labels[range(len(labels))[:50]] = labels_scaled[:50]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
