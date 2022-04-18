import torch
from torchsummary import summary
from modules import Yolo
import time

if __name__ == "__main__":

    anchors = [
        [12, 16],
        [19, 36],
        [40, 28],
        [36, 75],
        [76, 55],
        [72, 146],
        [142, 110],
        [192, 243],
        [459, 401]
    ]

    model = Yolo(anchors, 4)
    # summary(model, (3, 384, 512))

    x = torch.randn(32, 3, 384, 512)

    start_time = time.time()
    y = model(x)
    end_time = time.time()

    print(end_time - start_time)

