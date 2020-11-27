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

    def putDataInList(self, data, labels):
        dataList = []

        for i in range(len(data)):
            for j in range(len(labels)):
                if labels[j] == i:
                    dataList.append([data[j], i])
        random.shuffle(dataList)

        return dataList

    def processData(self, net, batchSizes):
        # get dictionary of data and label
        batchDict = net.unpickle("test_batch_osu")

        dataInList = net.putDataInList(batchDict[b'data'], batchDict[b'labels'])
        return dataInList 

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

    def test(self, net, testSample, networks, batchSize):
        correct = 0
        total = 0
        runningCorrects = 0.0
        accuracyHistory = []
        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

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

        for i in range(len(networks)):
            net.load_state_dict(torch.load('./Saved_Networks/' + networks[i]))
            with torch.no_grad():
                for i in range(len(minibatchesAndLabels)):
                    dataVal, labelsVal = net.getMiniBatch(minibatchesAndLabels[i])
                    out = net(dataVal)
                    _, predictions = torch.max(out, dim=1)
                    lossVal = criterionCE(out, labelsVal)
                    for i in range(len(labelsVal)):
                        runningCorrects += torch.sum(predictions == labelsVal[i])
            accuracy = (runningCorrects.item() / (len(minibatchesAndLabels) * 4)) 
            accuracyHistory.append(accuracy)
            runningCorrects = 0.0

        f = open("testresults.txt", "a")
        f.write(str(batchSize) + ' ' + str(accuracyHistory[0]) + ' ' + str(accuracyHistory[1]) + ' ' + str(accuracyHistory[2]) + '\n')
        f.close()


def main():
    batchSizes = [100, 220, 600, 720, 1100, 1220, 1600, 1720, 2100, 2220, 2600, 2720,
    3100, 3220, 3600, 3720, 4100, 4220, 4600, 4720]
    open('testresults.txt', 'w').close()

    net = Net()

    testData = net.processData(net, batchSizes)

    networks = sorted(listdir('./Saved_Networks'))
    groupsOfThree = list(zip(*[iter(networks)]*3))

    for i in range(len(groupsOfThree)):
        net.test(net, testData, groupsOfThree[i], batchSizes[i])


if __name__ == '__main__':
    main()