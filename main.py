from ctypes import util
import torch
from torchsummary import summary
from modules import Yolo
from util import ListDataset
import numpy as np

if __name__ == "__main__":

    # anchors = [
    #     [12, 16], [19, 36], [40, 28],
    #     [36, 75], [76, 55], [72, 146],
    #     [142, 110], [192, 243], [459, 401]
    # ]

    dataloader = torch.utils.data.DataLoader(ListDataset("Finetune/train/", img_size = (384, 512)), batch_size = 4, shuffle = True)

    anchors = [[16, 8], [23, 103], [28, 23], [56, 47], [96, 123], [157, 248]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Yolo(anchors, 4).to(device)
    print("Initialized model")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    print("Initialized model optimizer")
    
    for i, (_, imgs, targets) in enumerate(dataloader):
        # imgs = imgs.cuda()
        # targets = targets.cuda().requires_grad_(False)

        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        optimizer.step()
        print("Completed a batch.")

