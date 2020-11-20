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
                # dataReshaped = np.reshape(normImg, (3, 32, 32), order='F')
                dataReshaped = np.reshape(normImg, (32, 32, 3), order='F')
                tr = ndimage.rotate(dataReshaped, -90)
                normalizedMinibatch.append(tr)
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

    def getRandomSamples(self, batchSizes, combinedClassData):
        samples = []
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
        combinedBatchSizeSamples = [combined0, combined1, combined2, combined3, combined4, combined5, combined6, combined7, combined8, combined9]
                
        for batchSize in batchSizes:
            samplesPerBatchSize = []
            for classImagesList in combinedClassData:
                batchSizeSamples = random.sample(classImagesList, batchSize)
                samplesPerBatchSize.append(batchSizeSamples)
            samples.append(samplesPerBatchSize)

        for i in range(len(combinedBatchSizeSamples)):
            combinedBatchSizeSamples[i] = [y for x in samples[i] for y in x]
            random.shuffle(combinedBatchSizeSamples[i]) 

        return combinedBatchSizeSamples

    def processData(self, net):
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

        # split up the data by class similarly for the validation batch
        valDataByClass = net.separateDataByClass(valDict[b'data'], valDict[b'labels'])

        # Combine all the data with the same class from each batch.
            # Each element in the combinedClassData list corresponds to the entire set of images that 
            # belong to one of the 10 classes. The index of the list is the class label of that 
            # set of images. So, combinedClassData[0] is a list that is the set of all images 
            # belonging to class 0. Each image also has its classification label stored with it.
        combinedClassData = net.combineClassesAcrossBatches(batchDataByClass)

        # We were assigned the following sample sizes:
        # Allison: 100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600
        # Clint: 220, 720, 1220, 1720, 2220, 2720, 3220, 3720, 4220, 4720
        batchSizes = [100, 220, 600, 720, 1100, 1220, 1600, 1720, 2100, 2220, 2600, 2720,
        3100, 3220, 3600, 3720, 4100, 4220, 4600, 4720]

        # Take the data that has been organized by classification and remove random samples equal 
        # to the sample sizes we've been assigned from each classification, storing the samples for each
        # sample size in a list organized by sample size in ascending order. So, samples[0] will have
        # 100 random samples per class label i.e. 10*100 = 1000 total random samples. samples[1] will have
        # 10*220 total random samples, and so on up to samples[9] with 10*4720 random samples. 
        randomSamples = net.getRandomSamples(batchSizes, combinedClassData)

        #TODO: Do we get a small batch of the total validation sample for validating in the training loop or 
        # do we use the whole thing every time we validate?

        return randomSamples, valDataByClass

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
        epochs = 20
        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
        #running_loss = 0.0

        #print(len(valSamples), len(samples))

        # Split the data from the labels for each image in the training set samples.
            # Since we passed in only one batch size worth of samples, 'samples' is 
            # a single list of image / label pairs.
        sampleImages = net.getSampleImages(samples)
        sampleLabels = net.getSampleLabels(samples)

        # do the same for validation
        valImages = net.getSampleImages(valSamples)
        valLabels = net.getSampleLabels(valSamples)

        # plot the first image in the random sample to check that they are loading into the program properly
        # img = np.reshape(sampleImages[0], (32, 32, 3), order='F')
        # tr = ndimage.rotate(img, -90)
        # plt.imshow(tr)
        # plt.show()
        
        # Create minibatches of size 4 for the images and the labels
        minibatches = net.makeMiniBatches(sampleImages)
        minibatchLabels = net.makeMiniBatchLabels(sampleLabels)
        #print(minibatchLabels[0])

        #TODO: I don't know if we're supposed to get minibatches of validation data or not, but if we are
        valMinibatches = net.makeMiniBatches(valImages)
        valMinibatchLabels = net.makeMiniBatchLabels(valLabels)

        # normalize the images in the minibatch
        normalizedData = net.normalize(minibatches)

        #TODO: same as above, idk if we're supposed to do this or not
        normalizedValData = net.normalize(valMinibatches)

        # convert minibatch images into a tensor
        minibatchTensors = net.convertToTensor(normalizedData)
        # convert minibatch labels into a tensor
        minibatchLabelTensors = net.convertLabelsToTensor(minibatchLabels)
        #print(minibatchTensors[1])

        #TODO: same as above
        valMinibatchTensors = net.convertToTensor(normalizedValData)
        valMinibatchLabelTensors = net.convertLabelsToTensor(valMinibatchLabels)

        # put the tensors and the labels in one list
        minibatchesAndLabels = net.combineMinibatchTensorsAndLabels(minibatchTensors, minibatchLabelTensors)

        #TODO: same as above
        valMinibatchesAndLabels = net.combineMinibatchTensorsAndLabels(valMinibatchTensors, valMinibatchLabelTensors)

        #train the data
        for e in range(0, epochs):
            for i in range(len(minibatchesAndLabels)):
                net.train()
                data, labels = net.getMiniBatch(minibatchesAndLabels[i])
                dataReshaped = torch.reshape(data, (4, 3, 32, 32))
                optimizer.zero_grad()
                out = net(dataReshaped)
                loss = criterionCE(out, labels)
                loss.backward()
                print('[Epoch %d] loss: %.3f' % (e + 1, loss.item()))
                #running_loss += loss.item() 
                #print('[Epoch %d] loss: %.3f' % (e + 1, running_loss))
                optimizer.step

                #TODO: validation, here is the start of it, I think
                net.eval()
                with torch.no_grad():
                    data, labels = net.getMiniBatch(valMinibatchesAndLabels)


                #TODO: store network in output file

def main():
    net = Net()
    trainingSamples, validationSamples = net.processData(net)
    #TODO: Figure out how to we're supposed to train the network using all the different sample sizes' 
    # instead of just trainingSamples[0], which is the size 100 random samples set.
    net.trainBatch(net, trainingSamples[0], validationSamples[0], '/myNet.pth')

if __name__ == "__main__":
    main()