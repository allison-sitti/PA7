# Authors: Clint Lawson and Allison Cuba

# An image classifier using Torch

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

    def putValDataInList(self, data, labels):
        valDataList = []

        for i in range(len(data)):
            for j in range(len(labels)):
                if labels[j] == i:
                    valDataList.append([data[j], i])
        random.shuffle(valDataList)

        return valDataList

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
        # Get the dictionaries for each data batch. 
        # Each batch is a dict of <numpy array of data, list of labels>
        # where data[i] corresponds to list[i] and i represents one row
        # of the data array which is data for one 32x32 image
        # Also, the first 1024 entries in the array are for red channel
        # of each image, the next 1024 are for green channel of same images
        # and the final 1024 are for blue channel of same images
        batch1Dict = net.unpickle("cifar-10-python/data_batch_1")
        batch2Dict = net.unpickle("cifar-10-python/data_batch_2")
        batch3Dict = net.unpickle("cifar-10-python/data_batch_3")
        batch4Dict = net.unpickle("cifar-10-python/data_batch_4")
        batch5Dict = net.unpickle("cifar-10-python/data_batch_5")

        # get dictionaries for the validation batch
        valDict = net.unpickle("val_batch_osu")

        # throw all the data dictionaries into a list
        batchDicts = [batch1Dict, batch2Dict, batch3Dict, batch4Dict, batch5Dict]

        # split up the data by class for each batch and put class labels with each image
        batchDataByClass = []
        for dictionary in batchDicts:
            dataByClass = net.separateDataByClass(dictionary[b'data'], dictionary[b'labels'])
            batchDataByClass.append(dataByClass)

        # Put all the data from the validation batch into a list and shuffle the list
        valData = net.putValDataInList(valDict[b'data'], valDict[b'labels'])

        # Combine all the data with the same class from each batch.
            # Each element in the combinedClassData list corresponds to the entire set of images that 
            # belong to one of the 10 classes. The index of the list is the class label of that 
            # set of images. So, combinedClassData[0] is a list that is the set of all images 
            # belonging to class 0. Each image also has its classification label stored with it.
        combinedClassData = net.combineClassesAcrossBatches(batchDataByClass)
        # Note: the validation data is only one batch, so it already has all images separated by class with
        # nothing to combine

        # Take the data that has been organized by classification and remove random samples equal 
        # to the sample sizes we've been assigned from each classification, storing the samples for each
        # sample size in a list organized by classification, where each classification contains samples
        # for all 20 batch sizes randomly shuffled. So, randomSamples[0] would be for all classification 0 data, and
        # randomSamples[0][0] would be classification 0 data with 10*100 = 1000 total images.
        randomSamples = net.getRandomSamples(batchSizes, combinedClassData)
        # Note: We use the entire validation set each iteration through the loop, so we don't need to
        # get random samples from it. 

        return randomSamples, valData

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

    def plotImage(self, image):
        img = np.reshape(image, (32, 32, 3), order='F')
        tr = ndimage.rotate(img, -90)
        plt.figure(figsize=(2, 2))
        plt.imshow(tr)
        plt.show()

    def trainBatch(self, net, samples, valSamples, outputFile):
        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

        # Split the data from the labels for each image in the training set samples.
            # Since we passed in only one batch size worth of samples, 'samples' is 
            # a single list of image / label pairs.
        sampleImages = net.getSampleImages(samples)
        sampleLabels = net.getSampleLabels(samples)
        # do the same for validation
        valImages = net.getSampleImages(valSamples)
        valLabels = net.getSampleLabels(valSamples)

        # plot an image to ensure it is correctly displayed and loaded in
        # net.plotImage(sampleImages[0])

        # Create minibatches of size 4 for the images and the labels
        minibatches = net.makeMiniBatches(sampleImages)
        minibatchLabels = net.makeMiniBatchLabels(sampleLabels)
        # Create minibatches of size 4 for the images and labels of validation data
        valMinibatches = net.makeMiniBatches(valImages)
        valMinibatchLabels = net.makeMiniBatchLabels(valLabels)

        # normalize the images in the minibatches
        normalizedData = net.normalize(minibatches)
        # normalize the images in validation data
        normalizedValData = net.normalize(valMinibatches)

        # convert minibatch images into a tensors
        minibatchTensors = net.convertToTensor(normalizedData)
        # convert minibatch labels into a tensors
        minibatchLabelTensors = net.convertLabelsToTensor(minibatchLabels)
        # convert validation images into tensors
        valMinibatchTensors = net.convertToTensor(normalizedValData)
        # convert validation labels into tensors
        valMinibatchLabelTensors = net.convertLabelsToTensor(valMinibatchLabels)

        # put the tensors and the labels in one list
        minibatchesAndLabels = net.combineMinibatchTensorsAndLabels(minibatchTensors, minibatchLabelTensors)
        valDataMinibatchesAndLabels = net.combineMinibatchTensorsAndLabels(valMinibatchTensors, valMinibatchLabelTensors)

        epochs = 200
        runningLossHistory = []
        runningCorrectsHistory = []
        printFreq = 100
        
        #train the data
        for e in range(0, epochs):
            runningLoss = 0.0
            runningCorrects = 0.0

            for i in range(len(minibatchesAndLabels)):
                data, labels = net.getMiniBatch(minibatchesAndLabels[i])
                optimizer.zero_grad()
                out = net(data)
                loss = criterionCE(out, labels)
                loss.backward()
                optimizer.step()
                if i % printFreq == 0:
                    print('[Range: %d Epoch: %d] loss: %.4f' % (len(samples) / 10, e + 1, loss.item()))
            
            # Validation
            with torch.no_grad():
                for i in range(len(valDataMinibatchesAndLabels)):
                    dataVal, labelsVal = net.getMiniBatch(valDataMinibatchesAndLabels[i])
                    outVal = net(dataVal)
                    _, predictions = torch.max(outVal, dim=1)
                    lossVal = criterionCE(outVal, labelsVal)
                    runningLoss += lossVal.item()
                    for i in range(len(labelsVal)):
                        runningCorrects += torch.sum(predictions == labelsVal[i])
            epochLoss = runningLoss/len(valDataMinibatchesAndLabels)
            accuracy = runningCorrects.float()/len(valDataMinibatchesAndLabels)
            runningLossHistory.append(epochLoss)
            runningCorrectsHistory.append(accuracy)
        print('Validation Accuracy: %2.2f' % (100 * (runningCorrects.item() / (len(valDataMinibatchesAndLabels) * 4))))
        #wait = input('wait') # use this if you want to test code but don't want to save (overwrite) the network
        torch.save(net.state_dict(), outputFile) 
        print('Training completed, network saved to \'', outputFile, '\'.')

def main():
    net = Net()
    # We were assigned the following sample sizes:
    # Allison: 100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600
    # Clint: 220, 720, 1220, 1720, 2220, 2720, 3220, 3720, 4220, 4720
    batchSizes = [100, 220, 600, 720, 1100, 1220, 1600, 1720, 2100, 2220, 2600, 2720,
    3100, 3220, 3600, 3720, 4100, 4220, 4600, 4720]

    trainingSamples, validationSamples = net.processData(net, batchSizes)

    outputFiles = []
    for i in range(len(batchSizes)):
        if not i % 2 == 0:
            outputFile = './Saved_Networks/clintNetRange' + str(batchSizes[i]) + '_v3.pth'
        else:
            outputFile = './Saved_Networks/allisonNetRange' + str(batchSizes[i]) + '_v3.pth'
        outputFiles.append(outputFile)

    samplesIter = iter(range(0, 20))
    for i in samplesIter:
        net.trainBatch(net, trainingSamples[i], validationSamples, outputFiles[i])
        net.trainBatch(net, trainingSamples[i+1], validationSamples, outputFiles[i+1])
        next(samplesIter, None)

if __name__ == "__main__":
    main()