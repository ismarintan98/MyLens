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


datasetPATH = "dataset"
numfile = len(os.listdir(datasetPATH))

images = []
labels = []

for i in range(int(numfile)):
    if os.path.splitext(os.listdir(datasetPATH)[i])[1] == ".png":
        img = cv.imread(datasetPATH + "/" + os.listdir(datasetPATH)[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        images.append(img)
    elif os.path.splitext(os.listdir(datasetPATH)[i])[1] == ".txt":
        with open(datasetPATH + "/" + os.listdir(datasetPATH)[i], "r") as f:
            label = int(f.read())
            labels.append(label)

images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1)


#normalisasi
images = normalisasi(images,0,255)


Lens = LensModel()

#load model jika suda pernah di train
Lens.load_state_dict(torch.load("LensModel.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Lens.parameters(), lr=0.001, momentum=0.9)

# print("------ Program Training ------")
# print("Jumlah data: ", len(images))


# for epoch in range(30):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i in range(len(images)):

#         optimizer.zero_grad()

#         outputs = Lens(images[i].unsqueeze(0))
#         loss = criterion(outputs, labels[i])

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     if epoch % 10 == 9:
#         print('[%d] loss: %.3f' %
#               (epoch + 1, running_loss / len(images)))
#         # running_loss = 0.0

    
#save model
# torch.save(Lens.state_dict(), "LensModel.pth")




print('Finished Training')

print("------ Program Testing ------")

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # img_resize = cv.resize(frame, (0,0), fx=0.6, fy=0.6)
    img_resize = cv.resize(frame, (384, 288))

    #crop 80% dari frame
    height, width, _ = img_resize.shape
    img_crop = img_resize[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]

    img_buff = img_crop.copy()
    img_buff = cv.cvtColor(img_buff, cv.COLOR_BGR2RGB)

    img_buff = torch.tensor(np.transpose(img_buff, (2, 0, 1)), dtype=torch.float32)
    img_buff = normalisasi(img_buff,0,255)

    outputs = Lens(img_buff.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)

    cv.putText(img_crop, namaClass[predicted], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("frame", img_crop)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

print("Program selesai")










    






































