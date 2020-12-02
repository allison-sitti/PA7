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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
    
    def combineBatches(self, batchedData):
        combinedBatches = []
        combinedData = []
        for dictionary in batchedData:
            data = dictionary[b'data']
            labels = dictionary[b'labels']
            combinedBatches.append([data, labels])

        for dataAndLabelPair in combinedBatches:
            for i in range(len(dataAndLabelPair[0])):
                combinedData.append([dataAndLabelPair[0][i], dataAndLabelPair[1][i]])
        
        random.shuffle(combinedData)

        return combinedData

    def processData(self, net):
        batch1Dict = net.unpickle("cifar-10-python/data_batch_1")
        batch2Dict = net.unpickle("cifar-10-python/data_batch_2")
        batch3Dict = net.unpickle("cifar-10-python/data_batch_3")
        batch4Dict = net.unpickle("cifar-10-python/data_batch_4")
        batch5Dict = net.unpickle("cifar-10-python/data_batch_5")
        
        batchDicts = [batch1Dict, batch2Dict, batch3Dict, batch4Dict, batch5Dict]
        combinedData = net.combineBatches(batchDicts)

        return combinedData
    
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
    
    def trainBatch(self, net, samples, outputFile):
        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

        sampleImages = net.getSampleImages(samples)
        sampleLabels = net.getSampleLabels(samples)

        minibatches = net.makeMiniBatches(sampleImages)
        minibatchLabels = net.makeMiniBatchLabels(sampleLabels)
    
        normalizedData = net.normalize(minibatches)

        minibatchTensors = net.convertToTensor(normalizedData)
        minibatchLabelTensors = net.convertLabelsToTensor(minibatchLabels)

        minibatchAndLabels = net.combineMinibatchTensorsAndLabels(minibatchTensors, minibatchLabelTensors)

        epochs = 100
        printFreq = 100

        for e in range(0, epochs):
            for i in range(len(minibatchAndLabels)):
                data, labels = net.getMiniBatch(minibatchAndLabels[i])
                optimizer.zero_grad()
                out = net(data)
                loss = criterionCE(out, labels)
                loss.backward()
                optimizer.step()
                if i % printFreq == 0:
                    print('[Epoch: %d] loss: %.4f' % (e + 1, loss.item()))
        torch.save(net.state_dict(), outputFile)
        print('Training completed, network saved to \'', outputFile, '\'.')


def main():
    net = Net()
    trainingData = net.processData(net)

    net.trainBatch(net, trainingData, './Saved_Networks/bestNetwork_task3Clint_v2.pth')

if __name__ == "__main__":
    main()