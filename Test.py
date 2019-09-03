import os
import gc
import csv
import time
import copy
import torch
import argparse
import numpy as np
from Model import *
from torch import optim
from DataLoader import *
from helperUtils import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from tensorboard_logger import configure, log_value
from sklearn.metrics import normalized_mutual_info_score

from Models.ytfModel import *
from Models.frgcModel import *
from Models.uspsModel import *
from Models.mnistModel import *

## Initiating the parser.
parser = argparse.ArgumentParser()

## Argument for choosing the dataset.
parser.add_argument('--dataSet', type = str, default = 'MNIST', help = 'Define the dataSet to be chosen.')

## Argument for choosing the percentage of labelled data to be considered.
parser.add_argument('--percentLabData', type = float, default = '0.1', help = 'Define the percentage of labelled of data to be chosen.')

## Argument for choosing the batch size.
parser.add_argument('--testBatchSize', type = int, default = '64', help = 'Define the testing batch size.')

argList = parser.parse_args()

## Loading the testing data.
testData = testDataSet(argList.dataSet)

## Creating the test loader.
testLoader = DataLoader(testData, batch_size = argList.testBatchSize, shuffle = True)

## Loading the trained model.
## Model = torch.load('./TrainingResults/' + str(argList.dataSet) +'/PreTraining/Percentage' + str(int(argList.percentLabData * 100)) + '/λAnnealResults/AutoEncoderλAnneal.pkl')
## print('./TrainingResults/' + str(argList.dataSet) +'/PreTraining/Percentage' + str(int(argList.percentLabData * 100)) + '/λAnnealResults/AutoEncoderλAnneal.pkl')

if (argList.dataSet == 'USPS'):
	numClusters = 10
	Encoder = uspsEncoder()
	Decoder = uspsDecoder()
	Model = uspsAutoEncoder(Encoder, Decoder)

	## Loading the pretrained model.
	Model.load_state_dict(torch.load('./TrainingResults/' + str(argList.dataSet) +'/PreTraining/Percentage' + str(int(argList.percentLabData * 100)) + '/λAnnealResults/AutoEncoderλAnneal.pkl'))

## Loading the cluster centers.
clusterCenters = torch.from_numpy(np.load('./TrainingResults/'+ str(argList.dataSet) + '/PreTraining/Percentage' + str(int(argList.percentLabData * 100)) +'/λAnnealResults/clusterCenters.npy'))

## Directory for saving the original and reconstructed images.
saveDir = './TestingResults/' + str(argList.dataSet) + '/Images/'

## Creating placeholders for storing the predicted and true cluster labels.
yPredAll = []
yTrueAll = []

## Setting the AutoEncder to evaluation model.
Model.eval()

with torch.no_grad():
    
    for batchIndex, data in enumerate(testLoader):
        
        ## Loading the data batch.
        imgBatch, labelBatch = data
        ## imgBatchV, labelBatchV = Variable(imgBatch), Variable(labelBatch)
        imgBatchV, labelBatchV = Variable(imgBatch.cuda()), Variable(labelBatch.cuda())
        
        ## Performing the forward pass.
        encodings, reconstructions = Model(imgBatchV)
        
        ## Saving the original inputs as well as the reconstructions.
        utils.save_image(imgBatch.data.cpu(), saveDir + str(batchIndex) + 'Original.png', nrow = 4)
        utils.save_image(reconstructions.data.cpu(), saveDir + str(batchIndex) + 'Reconstructed.png', nrow = 4)
        
        ## Expanding the encodings and the cluster centers.
        encodingsExpand = encodings.unsqueeze(1).expand(encodings.size(0), 10, 32).cuda()
        clusterCentersExpand = clusterCenters.clone().unsqueeze(0).expand(encodings.size(0), 10, 32).cuda()
        
        ## Computing the distances of the encodings from the cluster centers.
        distMat = torch.pow(encodingsExpand - clusterCentersExpand, 2).sum(2)
        
        ## Computing the cluster center label having the minimum distance from the particular image encoding.
        _ , predClus = torch.min(distMat, dim = 1)
        
        ## Adding the predictions to the global list.
        yPredAll.extend(list(predClus.data.cpu().numpy()))
        
        ## Adding the true labels to the global list.
        yTrueAll.extend(list(labelBatchV.data.cpu().numpy()))

## Computing the test NMI and Purity.
testPurity = purityScore(yPredAll, yTrueAll)
testNMI = normalized_mutual_info_score(yPredAll, yTrueAll)

print(testPurity)
print(testNMI)

# ## Writing data.
data = [str(int(argList.percentLabData * 100)), str(testPurity), str(testNMI)]
with open('./TestingResults/' + str(argList.dataSet) + '/Test.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(data)