import os
import gc
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

## Initiating the parser.
parser = argparse.ArgumentParser()

## Argument for choosing the argList.dataset.
parser.add_argument('--dataSet', type = str, default = 'MNIST', help = 'Define the argList.dataSet to be chosen.')

## Argument for choosing the percentage of labelled data to be considered.
parser.add_argument('--percentLabData', type = float, default = '0.1', help = 'Define the percentage of labelled of data to be chosen.')

argList = parser.parse_args()

## Load the different lists from the pretrained models.
klLossListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/KLLossListλAnneal.npy')
totalLossListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/totalLossListλAnneal.npy')
reconErrorListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/reconErrorListλAnneal.npy')
clusterLossListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/clusterLossListλAnneal.npy')
labelledLossListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/labelledLossListλAnneal.npy')
unLabelledLossListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/unLabelledLossListλAnneal.npy')

trainNMILabelledListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMILabelledListλAnneal.npy')
trainNMIUnlabelledListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMIUnlabelledListλAnneal.npy')
trainPurityLabelledListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMIUnlabelledListλAnneal.npy')
trainPurityUnlabelledListPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainPurityUnlabelledListλAnneal.npy')

## Load the different lists from the non-pretrained models.
klLossListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/KLLossListλAnneal.npy')
totalLossListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/totalLossListλAnneal.npy')
reconErrorListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/reconErrorListλAnneal.npy')
clusterLossListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/clusterLossListλAnneal.npy')
labelledLossListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/labelledLossListλAnneal.npy')
unLabelledLossListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/unLabelledLossListλAnneal.npy')

trainNMILabelledListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMILabelledListλAnneal.npy')
trainNMIUnlabelledListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMIUnlabelledListλAnneal.npy')
trainPurityLabelledListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainNMIUnlabelledListλAnneal.npy')
trainPurityUnlabelledListNPT1 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λAnnealResults/trainPurityUnlabelledListλAnneal.npy')


## Load the different lists from the pretrained models.
klLossListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/KLLossListλ10.npy')
totalLossListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/totalLossListλ10.npy')
reconErrorListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/reconErrorListλ10.npy')
clusterLossListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/clusterLossListλ10.npy')
labelledLossListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/labelledLossListλ10.npy')
unLabelledLossListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/unLabelledLossListλ10.npy')

trainNMILabelledListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMILabelledListλ10.npy')
trainNMIUnlabelledListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMIUnlabelledListλ10.npy')
trainPurityLabelledListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMIUnlabelledListλ10.npy')
trainPurityUnlabelledListPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainPurityUnlabelledListλ10.npy')

## Load the different lists from the non-pretrained models.
klLossListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/KLLossListλ10.npy')
totalLossListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/totalLossListλ10.npy')
reconErrorListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/reconErrorListλ10.npy')
clusterLossListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/clusterLossListλ10.npy')
labelledLossListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/labelledLossListλ10.npy')
unLabelledLossListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/unLabelledLossListλ10.npy')

trainNMILabelledListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMILabelledListλ10.npy')
trainNMIUnlabelledListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMIUnlabelledListλ10.npy')
trainPurityLabelledListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainNMIUnlabelledListλ10.npy')
trainPurityUnlabelledListNPT2 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ1.0Results/trainPurityUnlabelledListλ10.npy')

## Load the different lists from the pretrained models.
klLossListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/KLLossListλ01.npy')
totalLossListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/totalLossListλ01.npy')
reconErrorListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/reconErrorListλ01.npy')
clusterLossListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/clusterLossListλ01.npy')
labelledLossListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/labelledLossListλ01.npy')
unLabelledLossListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/unLabelledLossListλ01.npy')

trainNMILabelledListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMILabelledListλ01.npy')
trainNMIUnlabelledListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01.npy')
trainPurityLabelledListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01.npy')
trainPurityUnlabelledListPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/PreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainPurityUnlabelledListλ01.npy')

## Load the different lists from the non-pretrained models.
klLossListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/KLLossListλ01.npy')
totalLossListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/totalLossListλ01.npy')
reconErrorListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/reconErrorListλ01.npy')
clusterLossListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/clusterLossListλ01.npy')
labelledLossListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/labelledLossListλ01.npy')
unLabelledLossListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/unLabelledLossListλ01.npy')

trainNMILabelledListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMILabelledListλ01.npy')
trainNMIUnlabelledListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01.npy')
trainPurityLabelledListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01.npy')
trainPurityUnlabelledListNPT3 = np.load('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainPurityUnlabelledListλ01.npy')


## Plotting the total loss of pretraining vs nopretraining.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), totalLossListPT1, 'r', label = 'Loss with AutoEncoder Pretraining')
ax.plot(list(range(60)), totalLossListNPT1, 'b', label = 'Loss without AutoEncoder Pretraining')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Total Loss')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/totalLoss.png')

## Plotting the total loss of Purity vs NMI.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), trainNMILabelledListPT1, 'r*', label = 'Training Labelled NMI')
ax.plot(list(range(60)), trainPurityLabelledListPT1, 'b*', label = 'Training Labelled Purity')
ax.plot(list(range(60)), trainNMIUnlabelledListPT1, 'y', label = 'Training Unlabelled NMI')
ax.plot(list(range(60)), trainPurityUnlabelledListPT1, 'g', label = 'Training Unlabelled Purity')
ax.legend()
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMI / Purity')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/NMIPurity.png')

## Plotting the Total Loss, Labelled Loss and Unlabelled Loss.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), totalLossListPT1, 'r', label = 'Total Loss')
ax.plot(list(range(60)), labelledLossListPT1, 'b', label = 'Labelled Loss')
ax.plot(list(range(60)), unLabelledLossListPT1, 'g', label = 'Unlabelled Loss')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/DifferentLoss1.png')

## Plotting the different labelled loss.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), totalLossListPT1, 'r', label = 'Total Loss')
ax.plot(list(range(60)), klLossListPT1, 'y', label = 'Pairwise Loss')
ax.plot(list(range(60)), reconErrorListPT1, 'g', label = 'Reconstruction Loss')
ax.plot(list(range(60)), clusterLossListPT1, 'm', label = 'Clustering Loss')
ax.plot(list(range(60)), labelledLossListPT1, 'k', label = 'Labelled Loss')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/DifferentLoss2.png')

## Plotting the total loss for the different annealing strategies.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), totalLossListPT1, 'r', label = 'Total Loss (Annealing λ)')
ax.plot(list(range(60)), totalLossListPT2, 'b', label = 'Total Loss (λ = 1.0)')
ax.plot(list(range(60)), totalLossListPT3, 'g', label = 'Total Loss (λ = 0.1)')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/AnnealingLoss.png')

## Plotting the NMI and Purity for the different annealing strategies.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(60)), trainNMILabelledListPT1, 'r*', label = 'NMI (Annealing λ)')
ax.plot(list(range(60)), trainPurityLabelledListPT1, 'b*', label = 'Purity (Annealing λ)')
ax.plot(list(range(60)), trainNMILabelledListPT2, 'm--', label = 'NMI (λ = 1.0)')
ax.plot(list(range(60)), trainPurityLabelledListPT2, 'k--', label = 'Purity (λ = 1.0)')
ax.plot(list(range(60)), trainNMILabelledListPT3, 'y', label = 'NMI (λ = 0.1)')
ax.plot(list(range(60)), trainPurityLabelledListPT3, 'g', label = 'Purity (λ = 0.1)')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/Plots/Percentage'+ str(int(argList.percentLabData * 100)) + '/NMIPurityAnnealing.png')