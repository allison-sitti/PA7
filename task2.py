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

    def combineClassesAcrossBatches(self, batchesByClass):
        combinedClass0 = []
        combinedClass1 = []
        combinedClass2 = []
        combinedClass3 = []
        combinedClass4 = []
        combinedClass5 = []
        combinedClass6 = []
        combinedClass7 = []
        combinedClass8 = []
        combinedClass9 = []
        combinedClassesData = [combinedClass0, combinedClass1, combinedClass2, combinedClass3, combinedClass4, combinedClass5, combinedClass6, combinedClass7, combinedClass8, combinedClass9]
        flattenedCombinedClass0 = []
        flattenedCombinedClass1 = []
        flattenedCombinedClass2 = []
        flattenedCombinedClass3 = []
        flattenedCombinedClass4 = []
        flattenedCombinedClass5 = []
        flattenedCombinedClass6 = []
        flattenedCombinedClass7 = []
        flattenedCombinedClass8 = []
        flattenedCombinedClass9 = []
        flattenedCombinedClasses = [flattenedCombinedClass0, flattenedCombinedClass1, flattenedCombinedClass2, flattenedCombinedClass3, flattenedCombinedClass4, flattenedCombinedClass5, flattenedCombinedClass6, flattenedCombinedClass7, flattenedCombinedClass8, flattenedCombinedClass9]
        
        for batch in batchesByClass:
            for i in range(len(combinedClassesData)):
                combinedClassesData[i].append(batch[i])

        for i in range(len(combinedClassesData)):
            flattenedCombinedClasses[i] = [y for x in combinedClassesData[i] for y in x]        

        return flattenedCombinedClasses
        
    def getRandomSamples(self, batchSize, combinedClassData):
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
        for aClass in combinedClassData:
            samplesFromABatchSize = random.sample(aClass, batchSize)
            samples.append(samplesFromABatchSize)

        print(len(combinedBatchSizeSamples))
        print(len(samples))

        for i in range(len(combinedBatchSizeSamples)):
            combinedBatchSizeSamples[i] = [y for x in samples[i] for y in x]
            random.shuffle(combinedBatchSizeSamples[i]) 

        return combinedBatchSizeSamples

    
    def processData(self, net, batchSize):
        batch1Dict = net.unpickle("cifar-10-python/data_batch_1")
        batch2Dict = net.unpickle("cifar-10-python/data_batch_2")
        batch3Dict = net.unpickle("cifar-10-python/data_batch_3")
        batch4Dict = net.unpickle("cifar-10-python/data_batch_4")
        batch5Dict = net.unpickle("cifar-10-python/data_batch_5")

        batchDicts = [batch1Dict, batch2Dict, batch3Dict, batch4Dict, batch5Dict]

        batchDataByClass = []
        for dictionary in batchDicts: 
            dataByClass = net.separateDataByClass(dictionary[b'data'], dictionary[b'labels'])
            batchDataByClass.append(dataByClass)
        

        combinedClassData = net.combineClassesAcrossBatches(batchDataByClass)

        randomSamples = net.getRandomSamples(batchSize, combinedClassData)
    
        return randomSamples
    
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
    
    def trainBatch(self, net, samples, valSamples, outputFile):
        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

        sampleImages = net.getSampleImages(samples)
        sampleLabels = net.getSampleLables(samples)

        minibatches = net.makeMiniBatches(sampleImages)
        minibatchLabels = net.makeMiniBatchLables(sampleLabels)
    
        normalizedData = net.normalize(minibatches)

        minibatchTensors = net.convertToTensor(normalizedData)
        minibatchLabelTensors = net.convertLabelsToTensor(minibatchLabels)

        minibatchAndLabels = net.combineMinibatchTensorsAndLabels(minibatchTensors, minibatchLabelTensors)

        epochs = 200
        runningLossHistory = []
        runningCorrectsHistory = []

        printFreq = 100

        for e in range(0, epochs):
            runningLoss= 0.0
            runningCorrects= 0.0

            for i in range(len(minibatchAndLabels)):
                data, labels = net.getMiniBatch(minibatchAndLabels[i])
                optimizer.zero_grad()
                out = net(data)
                loss = criterionCE(out, labels)
                loss.backwards()
                optimizer.step()
                if i % printFreq == 0:
                    print('[Range: %d Epoch: %d] loss: %.4f' % (len(samples) / 10, e + 1, loss.item()))



def main():
    net = Net()
    # We were assigned the following sample sizes:
    # Allison: 100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600
    # Clint: 220, 720, 1220, 1720, 2220, 2720, 3220, 3720, 4220, 4720
    bestBatchSize = 4100

    trainingSamples, validationSamples = net.processData(net, bestBatchSize)

    samplesIter = iter(range(0, 20))
    for i in samplesIter:
        net.trainBatch(net, trainingSamples[i], validationSamples, './Saved_Networks/bestNetwork.pth')

if __name__ == "__main__":
    main()