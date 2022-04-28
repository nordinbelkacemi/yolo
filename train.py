import torch
from util import progress, print_losses
from modules import Yolo
import numpy as np
from IPython.display import display

def init_model(num_classes, anchors, using_cuda):
    device = torch.device("cuda" if using_cuda else "cpu")
    model = Yolo(anchors, num_classes).to(device)
    return model

def train(model, dataloader, using_cuda, num_epochs = 20):
    # set to training mode
    model.train()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    # initialize progress bar
    bar = display(progress(0, len(dataloader)), display_id = True)

    # training loop
    for epoch in range(num_epochs):
        # 3 stages of detection: meaning 3 separate losses, where each loss is a tuple of 6 floats: (loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls, recall, precision)
        losses = np.zeros(9)

        for i, (_, imgs, targets) in enumerate(dataloader):
            if using_cuda:
                imgs = imgs.cuda()
                targets = targets.cuda().requires_grad_(False)

            optimizer.zero_grad()
            loss = model(imgs, targets)
            loss[0].backward()
            optimizer.step()

            losses += torch.Tensor(loss).cpu().detach().numpy()
            
            if bar is not None:
                bar.update(progress(i + 1, len(dataloader)))

        print_losses(dataloader, losses, epoch, num_epochs)

def save(model, path):
    torch.save(model.state_dict(), path)
