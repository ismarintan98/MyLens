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

images_val = []
labels_val = []


cnt_image = 0
cnt_label = 0

for i in range(int(numfile)):
    if os.path.splitext(os.listdir(datasetPATH)[i])[1] == ".png":
        img = cv.imread(datasetPATH + "/" + os.listdir(datasetPATH)[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if cnt_image == 0 or cnt_image == 1 or cnt_image == 2:          
            images.append(img)
        else:
            images_val.append(img)

        cnt_image += 1
        if cnt_image == 4:
            cnt_image = 0


    elif os.path.splitext(os.listdir(datasetPATH)[i])[1] == ".txt":
        with open(datasetPATH + "/" + os.listdir(datasetPATH)[i], "r") as f:
            label = int(f.read())
            if cnt_label == 0 or cnt_label == 1 or cnt_label == 2:
                labels.append(label)
            else:
                labels_val.append(label)

        cnt_label += 1
        if cnt_label == 4:
            cnt_label = 0


# print("Jumlah data: ", len(images))
# print("Jumlah label: ", len(labels))

# for i in range(len(images)):
#     print("Label: ", namaClass[labels[i]])
#     cv.imshow("img", images[i])
#     cv.waitKey(0)

# print("Jumlah data val: ", len(images_val))
# print("Jumlah label val: ", len(labels_val))

# for i in range(len(images_val)):
#     print("Label val: ", namaClass[labels_val[i]])
#     cv.imshow("img val", images_val[i])
#     cv.waitKey(0)




images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1)


#normalisasi
images = normalisasi(images,0,255)


Lens = LensModel()

#load model jika suda pernah di train
Lens.load_state_dict(torch.load("LensModel3.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Lens.parameters(), lr=0.001)

print("------ Program Training ------")
# print("Jumlah data: ", len(images))


# for epoch in range(30):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i in range(len(images)):

#         optimizer.zero_grad()

#         outputs = Lens(images[i].unsqueeze(0))#forward
#         loss = criterion(outputs, labels[i])

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     if epoch % 10 == 9:
#         print('[%d] loss: %.3f' %
#               (epoch + 1, running_loss / len(images)))
#         # running_loss = 0.0

    
# # save model
# torch.save(Lens.state_dict(), "LensModel3.pth")




print('Finished Training')

print("------ Program Testing ------")

images_val = torch.tensor(np.transpose(images_val, (0, 3, 1, 2)), dtype=torch.float32)
labels_val = torch.tensor(labels_val, dtype=torch.long).unsqueeze(1)

#normalisasi
images_val = normalisasi(images_val,0,255)

correct = 0
total = 0


with torch.no_grad():
    for i in range(len(images_val)):
        outputs = Lens(images_val[i].unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        total += labels_val[i].size(0)
        correct += (predicted == labels_val[i]).sum().item()

        buff_img = images_val[i]
        buff_img = denormalisasi(buff_img,0,255)
        buff_img = buff_img.numpy().astype(np.uint8)
        buff_img = np.transpose(buff_img, (1, 2, 0))
        buff_img = cv.cvtColor(buff_img, cv.COLOR_RGB2BGR)

        cv.putText(buff_img, "Prediksi: " + namaClass[predicted.item()], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(buff_img, "Label: " + namaClass[labels_val[i].item()], (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow("img", buff_img)
        cv.waitKey(0)

        
        




print("Total data: ", total)
print("Total data benar: ", correct)
print("Akurasi: ", correct/total*100, "%")


print("Program selesai")










    






































