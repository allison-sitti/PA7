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
    
    def separateDataByClass(self, data, labels):
        dataByClass0 = []
        dataByClass1 = []
        dataByClass2 = []
        dataByClass3 = []
        dataByClass4 = []
        dataByClass5 = []
        dataByClass6 = []
        dataByClass7 = []
        dataByClass8 = []
        dataByClass9 = []
        dataByClass = [dataByClass0, dataByClass1, dataByClass2, dataByClass3, dataByClass4, dataByClass5, dataByClass6, dataByClass7, dataByClass8, dataByClass9]

        for i in range(len(dataByClass)):
            for j in range(len(labels)):
                if labels[j] == i:
                    dataByClass[i].append([data[j], i])
        
        return dataByClass
    
    def getRandomSamples(self, batchSizes, combinedClassData):
        combined0 = []
        combined1 = []
        combined2 = []
        combined3 = []
        combined4 = []
        combined5 = []
        combined6 = []
        combined7 = []
        combined8 = []
        combined9 = []
        combined10 = []
        combined11 = []
        combined12 = []
        combined13 = []
        combined14 = []
        combined15 = []
        combined16 = []
        combined17 = []
        combined18 = []
        combined19 = []
        combinedBatchSizeSamples = [combined0, combined1, combined2, combined3, combined4, combined5, combined6, combined7, combined8, combined9,
        combined10, combined11, combined12, combined13, combined14, combined15, combined16, combined17, combined18, combined19]

        samples = []
        for aBatchSize in batchSizes:
            samplesFromEachBatchSize = []
            for aClass in combinedClassData:
                samplesFromABatchSize = random.sample(aClass, aBatchSize)
                samplesFromEachBatchSize.append(samplesFromABatchSize)
            samples.append(samplesFromEachBatchSize)
        for i in range(len(combinedBatchSizeSamples)):
            combinedBatchSizeSamples[i] = [y for x in samples[i] for y in x]
            random.shuffle(combinedBatchSizeSamples[i]) 

        return combinedBatchSizeSamples

    def processData(self, net, batchSizes):
        # get dictionary of data and label
        batchDict = net.unpickle("test_batch_osu")

        batchByClass = []
        dataByClass = net.separateDataByClass(batchDict[b'data'], batchDict[b'labels'])
        batchByClass.append(dataByClass)

        randomSamples = net.getRandomSamples(batchSizes, batchByClass)

        return randomSamples

    def test(self, net, randomSample, outputFile):
        correct = 0
        total = 0

        # get list of all networks inside of Saved_Networks foler
        networks = listdir('./Saved_Networks')

        net.load_state_dict(torch.load(networks[0]))
        



def main():
    batchSizes = [100, 220, 600, 720, 1100, 1220, 1600, 1720, 2100, 2220, 2600, 2720,
    3100, 3220, 3600, 3720, 4100, 4220, 4600, 4720]

    net = Net()

    randomSample = net.processData(net, batchSizes)

    outputFile = './testresults.txt'

    dirs = listdir('./Saved_Networks')

    print(dirs)

    # for i in range(0, 3):
    #     net.testBatch(net, randomSample, outputFile)


if __name__ == '__main__':
    main()