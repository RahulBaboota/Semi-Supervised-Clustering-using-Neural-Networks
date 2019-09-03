import torch
import numpy as np
from sklearn import metrics
from torch.autograd import Variable
from itertools import combinations as comb
from torch.utils.data import DataLoader


## Function to produce the subset of data according to number of constraints.
def produceSubset(trainData, numConstraints):
    
    ## Creating a list to contain the tuple (image index, label).
    subsetList = []

    labelList = []
    indexList = []

    ## Appending a random sample of each class to the subsetList.
    for index, row in trainData.iterrows() : 

        if (index == 0):

            infoTuple = (index, row['label'])
            subsetList.append(infoTuple)
            labelList.append(row['label'])
            indexList.append(index)

        if (row['label'] not in labelList and len(labelList)):

            infoTuple = (index, row['label'])
            subsetList.append(infoTuple)
            labelList.append(row['label'])      
            indexList.append(index)
            
    ## Creating a numpy array for random image indices.
    indices = np.random.randint(0, 40000, 1000)

    totConstraints = (len(indexList) * (len(indexList) - 1))/2
    simConstraints = 0
    dissimConstraints = totConstraints

    for i in range(0, len(indices)):

        ## Check if image is already in subsetList.
        if (indices[i] in indexList):
            continue

        ## Image will check labels of existing images to update the values of the constraints.
        for j in range(0, len(subsetList)):

            ## Similar Constraint.
            if (trainData['label'][indices[i]] == subsetList[j][1]):
                simConstraints += 1

            ## Dissimilar Constraint.
            if (trainData['label'][indices[i]] != subsetList[j][1]):
                dissimConstraints += 1

        infoTuple = (indices[i], trainData['label'][indices[i]])
        subsetList.append(infoTuple)
        indexList.append(indices[i])

        ## Updating the value of total number of constraints.
        totConstraints += (len(subsetList) - 1)

        ## Stopping Criteria.
        if (totConstraints >= 5000):
            break
            
    # Selecting the Subframe based on the indices list.
    trainDataSub = trainData.iloc[indexList]
    
    return trainDataSub


def createConstraintList(labelInfoList, yPred):
    
    ## Index Image 1 Scores.
    indexImage1Scores = yPred[labelInfoList[:, 0]]

    ## Index Image 2 Scores.
    indexImage2Scores = yPred[labelInfoList[:, 1]]
    
    return indexImage1Scores, indexImage2Scores

def klDivergenceSim(p, q):
    
    loss = torch.mul(p, torch.log(torch.div(p, q)))
    loss = torch.sum(loss, dim = 1)
    loss = torch.sum(loss, dim = 0)
    return loss

def klDivergenceDissim(p, q):
        
    ## Max clip number.
    clip = Variable(torch.from_numpy(np.array([0]))).float().cuda()

    loss = torch.mul(p, torch.log(torch.div(p, q)))
    loss = torch.sum(loss, dim = 1)
    loss = torch.max(clip, 2 - loss)
    loss = torch.sum(loss, dim = 0)
    return loss


def createLabelInfoList(imgBatch, labelBatch):

    ## Creating a numpy array for the image indices.
    indices = np.arange(len(imgBatch))

    indexMatrix = np.array(list(comb(indices, r = 2)))

    labelInfoListSim = []
    labelInfoListDissim = []

    ## Looping over each image combination to assign them the similarity label.
    for i in range(0, indexMatrix.shape[0]):

        ## Image 1 index.
        index1 = indexMatrix[i][0]

        ## Image 2 index.
        index2 = indexMatrix[i][1]

        ## Compare label values.
        if (labelBatch[index1] == labelBatch[index2]):
            similarIndx = 1
            infoTuple = (index1, index2, similarIndx)
            labelInfoListSim.append(infoTuple)

        else:
            similarIndx = 0
            infoTuple = (index1, index2, similarIndx)
            labelInfoListDissim.append(infoTuple)

    labelInfoListSim, labelInfoListDissim = np.array(labelInfoListSim), np.array(labelInfoListDissim)

    labelInfoListSim, labelInfoListDissim = torch.from_numpy(labelInfoListSim), torch.from_numpy(labelInfoListDissim)

    return labelInfoListSim, labelInfoListDissim


def purityScore(yPred, yTrue):

    ## Compute contingency matrix (also called confusion matrix).
    contingencyMatrix = metrics.cluster.contingency_matrix(yTrue, yPred)

    ## Return purity.
    return np.sum(np.amax(contingencyMatrix, axis = 0)) / np.sum(contingencyMatrix) 

def clusterLoss(clusterCenters, encodings, labelBatch):

    clusterLoss = 0.0

    for i in range(encodings.shape[0]):

        ## Extracting the label.
        label = labelBatch[i]

        ## Computing the distance of the encoding from the cluster center of the true label.
        value = (encodings[i, ] - clusterCenters[int(label), ]) ** 2

        ## Summing up across all dimensions.
        clusterLoss += torch.sum(value)

    return clusterLoss

## Defining the reconstruction loss function.
def mseLoss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()

def loadLabeledBatch(trainDataLab, labDataLoader):
    try:
        imgBatch1, imgBatch2, labBatch1 = next(labDataLoader)
    except StopIteration:
        labDataLoader = iter(DataLoader(dataset = trainDataLab, batch_size = 32, shuffle = True, drop_last = True))
        imgBatch1, imgBatch2, labBatch1 = next(labDataLoader)

    return imgBatch1, imgBatch2, labBatch1, labDataLoader


def loadUnlabeledBatch(trainDataUnlab, UnlabDataLoader):
    try:
        imgBatch3, labBatch3 = next(UnlabDataLoader)
    except StopIteration:
        UnlabDataLoader = iter(DataLoader(dataset = trainDataUnlab, batch_size = 64, shuffle = True, drop_last = True))
        imgBatch3, labBatch3 = next(UnlabDataLoader)

    return imgBatch3, labBatch3, UnlabDataLoader
