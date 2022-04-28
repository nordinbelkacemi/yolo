import torch
from util import progress, print_losses
from modules import Yolo
import numpy as np
from IPython.display import display

def init_model(num_classes, anchors, using_cuda):
    device = torch.device("cuda" if using_cuda else "cpu")
    model = Yolo(anchors, num_classes).to(device)
    return model

def train(model, dataloader, using_cuda, num_epochs = 10):
    # set to training mode
    model.train()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    # initialize progress bar
    bar = display(progress(0, len(dataloader)), display_id = True)

    # training loop
    for epoch in range(num_epochs):
        # 3 stages of detection: meaning 3 separate losses, where each loss is a tuple of 6 floats: (loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls, recall, precision)
        all_losses_sum = np.zeros((3, 9))

        for i, (_, imgs, targets) in enumerate(dataloader):
            if using_cuda:
                imgs = imgs.cuda()
                targets = targets.cuda().requires_grad_(False)

            optimizer.zero_grad()
            all_losses = model(imgs, targets)
            with torch.no_grad:
                print(all_losses)
            all_losses[0].backward()
            optimizer.step()

            all_losses_sum += torch.Tensor(all_losses[:][1:]).cpu().detach().numpy()
            
            if bar is not None:
                bar.update(progress(i + 1, len(dataloader)))

        print("[ Epoch %d/%d ]" % (epoch + 1, num_epochs))
        for i in range(3):
            print_losses(dataloader, all_losses_sum, i)

def save(model, path):
    torch.save(model.state_dict(), path)
