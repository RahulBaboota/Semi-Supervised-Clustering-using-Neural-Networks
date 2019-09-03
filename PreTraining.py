import os
import time
import copy
import torch
import argparse
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from DataLoader import dataSplit, Data
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from tensorboard_logger import configure, log_value

from Models.ytfModel import *
from Models.frgcModel import *
from Models.uspsModel import *
from Models.cifarModel import *
from Models.mnistModel import *

## Initiating the parser.
parser = argparse.ArgumentParser()

## Argument for choosing the dataset.
parser.add_argument('--dataSet', type = str, default = 'MNIST', help = 'Define the dataSet to be chosen.')

## Argument for choosing the number of epochs.
parser.add_argument('--numEpochs', type = int, default = '100', help = 'Define the number of epochs.')

## Argument for choosing the batch size.
parser.add_argument('--batchSize', type = int, default = '128', help = 'Define the batch size.')

## Argument for choosing the optimizer parameters.
parser.add_argument('--learningRate', type = float, default = '1e-4', help = 'Define the learning rate.')
parser.add_argument('--weightDecay', type = float, default = '1e-5', help = 'Define the weight decay.')
parser.add_argument('--beta1', type = float, default = '0.9', help = 'Define the value of beta1.')
parser.add_argument('--beta2', type = float, default = '0.999', help = 'Define the value of beta2.')

argList = parser.parse_args()


if __name__ == '__main__':

	## Defining the transformations to be applied.
	Transforms = transforms.Compose([transforms.ToTensor()])

	## Preparing the data.
	labData, unlabData = dataSplit(percentage = 1.0, dataset = argList.dataSet)
	trainDataSet = Data(labData)

	## Creating the data loaders.
	trainLoader = DataLoader(dataset = trainDataSet, batch_size = argList.batchSize, shuffle = True, drop_last = True)

	## Defining the reconstruction loss function.
	def mseLoss(input, target):
		return torch.sum((input - target).pow(2)) / input.data.nelement()

	## Instantiating the Model.
	if (argList.dataSet == 'CIFAR10'):
		Encoder = cifarEncoder()
		Decoder = cifarDecoder()
		Model = cifarAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'FRGC'):
		Encoder = frgcEncoder()
		Decoder = frgcDecoder()
		Model = frgcAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'MNIST'):
		Encoder = mnistEncoder()
		Decoder = mnistDecoder()
		Model = mnistAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'USPS'):
		Encoder = uspsEncoder()
		Decoder = uspsDecoder()
		Model = uspsAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'YTF'):
		Encoder = ytfEncoder()
		Decoder = ytfDecoder()
		Model = ytfAutoEncoder(Encoder, Decoder)

	Model = Model.cuda()

	## Defining the optimizer.
	optimizer = optim.Adam(Model.parameters(), lr = argList.learningRate, weight_decay = argList.weightDecay, betas = [argList.beta1, argList.beta2])

	## Initailize Tensorboard logging values.
	configure('./PreTrainingResults/' + argList.dataSet + '/Logs/')
	log_value('ReconLoss', 1.0, 0)

	## List for holding the loss after each epoch.
	epochLossList = []

	## Training the Auto-Encoder.
	for epoch in range(argList.numEpochs):
		
		## Variable for holding the running loss.
		runningLoss = 0.0

		for batchIndex, data in enumerate(trainLoader):

			image, labels = data
			x = Variable(image.cuda())
			x = x.float()

			## Forward Pass.
			encoded, reconstructions = Model(x)

			## Evaluating loss.
			loss = mseLoss(reconstructions, x)
			runningLoss += loss.item()

			##Initialise all parameter gradients to 0.
			optimizer.zero_grad()
			
			## Backpropagation.
			loss.backward()

			## Optimisation.
			optimizer.step()

			if batchIndex % 100 == 0:
				print('Epoch: ', epoch, '| Train Loss: %.4f' % loss.item())

		## Log Data for tensorboard.
		log_value('ReconLoss', (runningLoss / len(labData)), epoch)
		
		## Update epochLossList.
		epochLossList.append(runningLoss / len(labData))
		print('Snapshot Taken !')
		print()

	## Saving the pretrained model.
	torch.save(Model, './PreTrainingResults/' + argList.dataSet + '/AutoEncoder.pkl')
	torch.save(Model.state_dict(), './PreTrainingResults/' + argList.dataSet + '/AutoEncoder.pth')

	## Plotting the loss.
	plt.plot(epochLossList)
	plt.xlabel('Epoch')
	plt.ylabel('Reconstruction Loss')
	plt.title('Reconstruction Loss')
	plt.savefig('./PreTrainingResults/' + argList.dataSet + '/ReconLoss.png')