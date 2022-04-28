import torch
import torchvision.transforms as transforms
import glob
from PIL import Image

list_path = "Finetune/train/"
img_paths = [list_path + img for img in glob.glob1(list_path, "*.png")]

imgs = []
for i, path in enumerate(img_paths):
    img = Image.open(path).convert('RGB')
    img = (transforms.ToTensor())(img)
    imgs.append(img.unsqueeze(0))

data = torch.cat(imgs)

mean = torch.mean(data, dim=[0, 2, 3])
std = torch.std(data, dim=[0, 2, 3], unbiased=False)

print(f"mean: {mean}")
print(f"std: {std}")
