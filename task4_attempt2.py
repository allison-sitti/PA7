import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import pickle
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def processImage(self, imagePath):
        sourceImg = Image.open(imagePath)
        resizedImg = sourceImg.resize((32, 32))
        img = np.reshape(resizedImg, (32, 32, 3), order='F')

        return img

    def plotImage(self, image):
        plt.figure(figsize=(2, 2))
        plt.imshow(image)
        plt.show()

    def normalize(self, image):
        normImg = np.true_divide(image, 255.0)
        dataReshaped = np.reshape(normImg, (3, 32, 32))

        return dataReshaped

    def convertToTensor(self, image):
        tensor = torch.tensor(image, dtype=torch.float32)
        return tensor

    def predictLabel(self, net, image):
        image = image.unsqueeze(0)
        out = net(image)
        _, prediction = torch.max(out, dim=1)
        print(prediction)
    
def main():
    net = Net()
    previous = torch.load('./Saved_Networks/bestNetwork.pth')
    net.load_state_dict(previous)

    imageFile = sys.argv[1]
    imagePath = './task4_images/' + imageFile
    resizedImg = net.processImage(imagePath)

    net.plotImage(resizedImg)

    normalizedImg = net.normalize(resizedImg)
    tensorImg = net.convertToTensor(normalizedImg)

    net.predictLabel(net, tensorImg)

if __name__ == "__main__":
    main()