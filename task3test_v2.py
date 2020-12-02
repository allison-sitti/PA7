import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import pickle
import random
from os import listdir

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def normalize(self, minibatches):
        normalizedMinibatch = []
        normalizedMinibatches = []
        
        for minibatch in minibatches:
            normalizedMinibatch.clear()
            for i in range(len(minibatch)):
                normImg = np.true_divide(minibatch[i], 255.0)
                dataReshaped = np.reshape(normImg, (3, 32, 32))
                normalizedMinibatch.append(dataReshaped)
            normalizedMinibatchArr = np.array(normalizedMinibatch)
            normalizedMinibatches.append(normalizedMinibatchArr)

        return normalizedMinibatches
            
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def parseData(self, data, labels):
        testData = []
        for i in range(len(data)):
            testData.append([data[i], labels[i]])
        random.shuffle(testData)
        
        return testData

    def processData(self, net):
        # get dictionary of data and label
        imageNetDict = net.unpickle("ImageNet_test_batch")
        cifarDict = net.unpickle("test_batch_osu")

        imageNet = net.parseData(imageNetDict['data'], imageNetDict['labels'])
        cifar = net.parseData(cifarDict[b'data'], cifarDict[b'labels'])

        return imageNet, cifar

    def getSampleImages(self, samples):
        sampleImages = []

        for imageLabelPair in samples:
           sampleImages.append(imageLabelPair[0])
                
        return sampleImages

    def getSampleLabels(self, samples):
        sampleLabels = []

        for imageLabelPair in samples:
           sampleLabels.append(imageLabelPair[1])

        return sampleLabels

    def makeMiniBatches(self, sampleImages):
        minibatches = [sampleImages[x:x+4] for x in range(0, len(sampleImages), 4)]
        return minibatches

    def makeMiniBatchLabels(self, sampleLabels):
        labels = np.array(sampleLabels)
        minibatchLabels = [labels[x:x+4] for x in range(0, len(labels), 4)]
        return minibatchLabels
    
    def convertToTensor(self, minibatches): 
        minibatchesToTensorsList = []
        for minibatch in minibatches:
            tensor = torch.tensor(minibatch, dtype=torch.float32)
            minibatchesToTensorsList.append(tensor)

        return minibatchesToTensorsList

    def convertLabelsToTensor(self, labels):
        labelsToTensorsList = []
        for label in labels:
            tensor = torch.tensor(label, dtype=torch.int64)
            labelsToTensorsList.append(tensor)

        return labelsToTensorsList

    def combineMinibatchTensorsAndLabels(self, tensors, labels):
        miniBatchesAndLabels = []

        for i in range(len(tensors)):
            minibatch = tensors[i]
            minibatchLabels = labels[i]
            miniBatchesAndLabels.append([minibatch, minibatchLabels])

        return miniBatchesAndLabels

    def getMiniBatch(self, minibatchAndLabel):
        return minibatchAndLabel[0], minibatchAndLabel[1]

    def test(self, net, testSample, network, testIdx):
        runningCorrects = 0.0

        images = net.getSampleImages(testSample)
        labels = net.getSampleLabels(testSample)

        miniBatches = net.makeMiniBatches(images)
        miniBatchLabels = net.makeMiniBatchLabels(labels)

        # normalize
        normalizedData= net.normalize(miniBatches)
    
        # convert to tensors
        miniBatchTensor = net.convertToTensor(normalizedData)
        miniBatchLabelTensor = net.convertLabelsToTensor(miniBatchLabels)

        # combine labels and data into one list
        minibatchesAndLabels = net.combineMinibatchTensorsAndLabels(miniBatchTensor, miniBatchLabelTensor)

        with torch.no_grad():
            for i in range(len(minibatchesAndLabels)):
                data, labels = net.getMiniBatch(minibatchesAndLabels[i])
                out = net(data)
                _, predictions = torch.max(out, dim=1)
                for i in range(len(labels)):
                    runningCorrects += torch.sum(predictions == labels[i])
        accuracy = (runningCorrects.item() / (len(minibatchesAndLabels) * 4)) 

        if testIdx == 0:
            #TODO: same file name as the first line in main
            f = open("testresults_task3_v2.txt", "a")
            f.write('Best Network, Retrained. Using ImageNet: ' + str(accuracy) + '\n')
            f.close()
        else:
            #TODO: same file name as the first line in main
            f = open("testresults_task3_v2.txt", "a")
            f.write('Best Network, Retrained. Using Cifar-10: ' + str(accuracy) + '\n')
            f.close()


def main():
    #TODO change file name
    open('testresults_task3_v2.txt', 'w').close()
    net = Net()
    #TODO: load the filename you saved in task3Allison_v1 or whatever
    network = torch.load('./Saved_Networks/bestNetwork_task3Clint_v2.pth')
    net.load_state_dict(network)

    testData = []
    testDataImageNet, testDataCifar = net.processData(net)
    testData.append(testDataImageNet)
    testData.append(testDataCifar)

    for i in range(len(testData)):
        net.test(net, testData[i], network, i)


if __name__ == '__main__':
    main()