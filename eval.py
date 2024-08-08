import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import dataloader
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import ImageFolder
from MODEL import Net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='학번_이름.pth', help="the model to evaluate (.pth)")
parser.add_argument("--batch", default=100)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(args.path, map_location=device)
model = model.to(device)

test_dataset = ImageFolder(root='./data/test', transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

#test
correct = 0
model = model.eval()


with torch.no_grad():
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
test_accuracy = 100. * correct/len(test_loader.dataset)

print("Test Accuracy: {:.2f}".format(test_accuracy))
    
    