import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


namaClass = ["pulpen","kalkulator","ponsel","buku","jam tangan",
             "correction tape","gunting","botol","penggaris","garpu"]

def normalisasi(x,x_min,x_max):
    return 2*((x-x_min)/(x_max-x_min))-1

def denormalisasi(x,x_min,x_max):
    return 0.5*(x+1)*(x_max-x_min)+x_min

    


class LensModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 25)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 10)
        self.fc1 = nn.Linear(16 * 32 * 47, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 32 * 47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#create random image 3 x 173 x 231
img = np.random.randint(0,255,(3,173,231))

#convert to tensor
img = torch.from_numpy(img)
img = img.unsqueeze(0)


print(img.shape)

input = img.float()
conv1 = nn.Conv2d(3, 6, 25)
x = conv1(input)
print(x.shape)
pool = nn.MaxPool2d(2, 2)
x = pool(x)
print(x.shape)
conv2 = nn.Conv2d(6, 16, 10)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)


# tens_input = torch.from_numpy(img, dtype=torch.float32)


# print(tens_input.shape)

