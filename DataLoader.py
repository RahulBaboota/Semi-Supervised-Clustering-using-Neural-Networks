import os
import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

# Defining the transforms to be applied.
Transforms = transforms.Compose([transforms.ToTensor()])

## DataLoader for CIFAR DataSet.
def loadCIFARBatch(filename):

    ## Load single batch of CIFAR.
    with open(filename, 'rb') as f:

        ## Loading the Data Dictionary.
        datadict = pickle.load(f, encoding = 'latin')

        ## Extracting the images and labels.
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def loadCIFAR10(ROOT):

    ## Load Entire CIFAR data.
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = loadCIFARBatch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    return Xtr, Ytr

## DataLoader for the DataSet.
def dataSplit(percentage = 1.0, dataset = 'MNIST'):

    ## Defining the list of labels.
    classToConsider = []
    
    if (dataset == 'MNIST'):

        ## Download MNIST Dataset if not not present.
        DataSet = MNIST('./MNIST', download = True, transform = Transforms)

        ## Loading the training data as pyTorch tensors.
        trainImages, trainLabels = torch.load('MNIST/processed/training.pt')
        
        ## Setting the list of labels.
        classToConsider = list(range(10))

    if (dataset == 'CIFAR10'):

        ## Download CIFAR Dataset if not not present.
        DataSet = CIFAR10('./CIFAR10', download = True, transform = Transforms)

        ## Loading the CIFAR 10 data.
        cifar10Dir = 'CIFAR10/cifar-10-batches-py'
        trainImages, trainLabels = loadCIFAR10(cifar10Dir)
        
        ## Setting the list of labels.
        classToConsider = list(range(10))

    # if (dataset == 'CIFAR10'):

    #     ## Loading the CIFAR data.
    #     cifarData = pickle.load(open('./Data/cifarTrain.pickle', 'rb'))
    #     trainImages = np.array(cifarData['data']).transpose(0, 3, 2, 1).astype(np.uint8)
    #     trainLabels = np.array(cifarData['labels']).astype(np.uint8)
        
    #     ## Setting the list of labels.
    #     classToConsider = list(range(10))
        
    if (dataset == 'USPS'):
        
        ## Loading the USPS data.
        uspsData = pickle.load(open('./Data/uspsTrain.pickle', 'rb'))
        trainImages = np.array(uspsData['data']).transpose(0, 3, 2, 1).astype(np.uint8)
        trainLabels = np.array(uspsData['labels']).astype(np.uint8)
        
        ## Setting the list of labels.
        classToConsider = list(range(10))
        
    if (dataset == 'FRGC'):
        
        ## Loading the FRGC data.
        frgcData = pickle.load(open('./Data/frgcTrain.pickle', 'rb'))
        trainImages = np.array(frgcData['data']).transpose(0, 3, 2, 1).astype(np.uint8)
        trainLabels = np.array(frgcData['labels']).astype(np.uint8)
        
        ## Setting the list of labels.
        classToConsider = list(range(20))
        
    if (dataset == 'YTF'):
        
        ## Loading the YTF data.
        ytfData = pickle.load(open('./Data/ytfTrain.pickle', 'rb'))
        trainImages = np.array(ytfData['data']).transpose(0, 2, 3, 1).astype(np.uint8)
        trainLabels = np.array(ytfData['labels']).astype(np.uint8)
        
        ## Setting the list of labels.
        classToConsider = list(range(41))
        
    ## Converting pyTorch tensors to numpy array (if required).
    try:
        trainImages, trainLabels = trainImages.numpy(), trainLabels.numpy()
    except AttributeError:
        pass

    ## Creating dictionaries to hold the labelled and the unlabelled data.
    Lab = {}
    Lab['data'] = []
    Lab['labels'] = []

    Unlab = {}
    Unlab['data'] = []
    Unlab['labels'] = []

    for label in classToConsider:

        ## Finding those image indices which correspond to the class label under consideration.
        inds = np.where(trainLabels == label)[0]

        ## Obtaining the images and the labels corresponding to the indices obtained above.
        data = trainImages[inds]
        labels = trainLabels[inds]

        ## Computing the percentage of data to be considered.
        choose = int(len(inds) * percentage)
        
        if (choose < 2):
            choose = 2

        ## Finding the random subset of data to sample according to the percentage of labelled data to be considered.
        ins = np.random.choice(len(labels), choose, replace = False)

        ## Creating a boolean array with "True" at those indices where the label is not available and vice-versa.
        unLabInd = np.ones(len(labels), dtype = bool)
        unLabInd[ins] = False

        ## Adding the data to the relevant dictionaries.
        Lab['data'].extend(data[ins])
        Lab['labels'].extend(labels[ins])

        Unlab['data'].extend(data[unLabInd])
        Unlab['labels'].extend(labels[unLabInd])

    ## Converting the images and labels to numpy arrays.
    Lab['data'], Lab['labels'] = np.array(Lab['data']), np.array(Lab['labels'])
    Unlab['data'], Unlab['labels'] = np.array(Unlab['data']) , np.array(Unlab['labels'])

    if (dataset == 'MNIST'):

        ## Adding another dimension..
        Lab['data'] = np.expand_dims(Lab['data'], axis = 3)

        Unlab['data'] = np.expand_dims(Unlab['data'], axis = 3)

    return Lab, Unlab

## PyTorch DataLoader.
class Data(Dataset):
    
    def __init__(self, dataset, transform = Transforms):
        
        self.dataset = dataset
        self.transform = transform
        self.data_dict = {}
        
        for i in range(self.__len__()):
            
            image, label = self.dataset['data'][i], self.dataset['labels'][i]
            image = self.transform(image)
            
            try:
                self.data_dict[label]
            except KeyError:
                self.data_dict[label] = []
            
            self.data_dict[label].append(image)
            
    def __len__(self):
        
        return len(self.dataset['labels'])
    
    def __getitem__(self, index):
        
        image, label = self.dataset['data'][index], self.dataset['labels'][index]
        image = self.transform(image)
        
        return image, label

## PyTorch DataLoader.
class UnlabelledData(Dataset):
    
    def __init__(self, dataset, transform = Transforms):
        
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        
        return len(self.dataset['labels'])
    
    def __getitem__(self, index):
        
        image, label = self.dataset['data'][index], self.dataset['labels'][index]
        image = self.transform(image)
        
        ## return another image of the same class randomly selected from the data dictionary
        return image, label

## PyTorch DataLoader.
class labelledData(Dataset):
    
    def __init__(self, dataset, transform = Transforms):
        
        self.dataset = dataset
        self.transform = transform
        self.dataDict = {}
        
        for i in range(self.__len__()):
            
            image, label = self.dataset['data'][i], self.dataset['labels'][i]
            image = self.transform(image)
            
            try:
                self.dataDict[label]
            except KeyError:
                self.dataDict[label] = []
            
            self.dataDict[label].append(image)
            
    def __len__(self):
        
        return len(self.dataset['labels'])
    
    def __getitem__(self, index):
        
        image, label = self.dataset['data'][index], self.dataset['labels'][index]
        image = self.transform(image)
        
        ## return another image of the same class randomly selected from the data dictionary
        return image, random.SystemRandom().choice(self.dataDict[label]), label

## PyTorch DataLoader.
class testDataSet(Dataset):

    def __init__(self, dataset, transform = Transforms):

        ## Loading the data.
        if (dataset == 'USPS'):
            temp = pickle.load(open('./Data/uspsTest.pickle','rb'))
            self.data = np.array(temp['data']).transpose(0, 3, 2, 1).astype(np.uint8)

        if (dataset == 'YTF'):
            temp = pickle.load(open('./Data/ytfTest.pickle','rb'))
            self.data = np.array(temp['data']).transpose(0, 2, 3, 1).astype(np.uint8)

        if (dataset == 'FRGC'):
            temp = pickle.load(open('./Data/frgcTest.pickle','rb'))
            self.data = np.array(temp['data']).transpose(0, 2, 3, 1).astype(np.uint8)

        self.labels = np.array(temp['labels'])
        self.transform = transform

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label