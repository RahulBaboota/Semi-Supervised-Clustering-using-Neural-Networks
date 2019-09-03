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

from Models.ytfModel import *
from Models.frgcModel import *
from Models.uspsModel import *
from Models.cifarModel import *
from Models.mnistModel import *

## Initiating the parser.
parser = argparse.ArgumentParser()

## Argument for choosing the dataset.
parser.add_argument('--dataSet', type = str, default = 'MNIST', help = 'Define the dataSet to be chosen.')

## Argument for choosing the percentage of labelled data to be considered.
parser.add_argument('--percentLabData', type = float, default = '0.1', help = 'Define the percentage of labelled of data to be chosen.')

## Argument for choosing the number of epochs.
parser.add_argument('--numEpochs', type = int, default = '60', help = 'Define the number of epochs.')

## Argument for deciding the embedding space size.
parser.add_argument('--embeddingSpace', type = int, default = '32', help = 'Define the embedding space size.')

## Argument for choosing the batch size.
parser.add_argument('--labBatchSize', type = int, default = '32', help = 'Define the labelled batch size.')
parser.add_argument('--unlabBatchSize', type = int, default = '64', help = 'Define the unlabelled batch size.')
parser.add_argument('--testBatchSize', type = int, default = '64', help = 'Define the testing batch size.')

## Argument for choosing the optimizer parameters.
parser.add_argument('--learningRate', type = float, default = '1e-4', help = 'Define the learning rate.')
parser.add_argument('--weightDecay', type = float, default = '1e-5', help = 'Define the weight decay.')
parser.add_argument('--beta1', type = float, default = '0.9', help = 'Define the value of beta1.')
parser.add_argument('--beta2', type = float, default = '0.999', help = 'Define the value of beta2.')

argList = parser.parse_args()


if __name__ == '__main__':

	## Variable for deciding the number of clusters.
	numClusters = 0

	## Instantiating the Model.
	if (argList.dataSet == 'CIFAR10'):
		numClusters = 10
		Encoder = cifarEncoder()
		Decoder = cifarDecoder()
		Model = cifarAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'FRGC'):
		numClusters = 20
		Encoder = frgcEncoder()
		Decoder = frgcDecoder()
		Model = frgcAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'MNIST'):
		numClusters = 10
		Encoder = mnistEncoder()
		Decoder = mnistDecoder()
		Model = mnistAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'USPS'):
		numClusters = 10
		Encoder = uspsEncoder()
		Decoder = uspsDecoder()
		Model = uspsAutoEncoder(Encoder, Decoder)

	if (argList.dataSet == 'YTF'):
		numClusters = 41
		Encoder = ytfEncoder()
		Decoder = ytfDecoder()
		Model = ytfAutoEncoder(Encoder, Decoder)

	Model = Model.cuda()

	## Loading the pretrained model.
	## Model.load_state_dict(torch.load('./NoPreTrainingResults/' + str(argList.dataSet) + '/AutoEncoder.pth'))

	## Defining the transformations to be applied.
	Transforms = transforms.Compose([transforms.ToTensor()])

	## Defining the optimizer.
	optimizer = optim.Adam(Model.parameters(), lr = argList.learningRate, weight_decay = argList.weightDecay, betas = [argList.beta1, argList.beta2])

	## Loading the labelled and unlabelled data.
	labData, unlabData = dataSplit(percentage = argList.percentLabData, dataset = argList.dataSet)
	trainDataLab = labelledData(labData, transform = Transforms)
	trainDataUnlab = UnlabelledData(unlabData, transform = Transforms)

	## Creating the data loaders.
	labDataLoader = iter(DataLoader(dataset = trainDataLab, batch_size = argList.labBatchSize, shuffle = True, drop_last = True))
	UnlabDataLoader = iter(DataLoader(dataset = trainDataUnlab, batch_size = argList.unlabBatchSize, shuffle = True, drop_last = True))

	## List for holding the different losses after each epoch.
	KLLossList = []
	totalLossList = []
	reconErrorList = []
	clusterLossList = []
	labelledLossList = []
	unLabelledLossList = []
	trainNMILabelledList = []
	trainNMIUnlabelledList = []
	trainPurityLabelledList = []
	trainPurityUnlabelledList = []

	## Initialising the unlabelled data clustering and pairwise KL Loss.
	lossCUnlab = 0.0
	lossKLUnlab = 0.0

	## Initialise a vector to count the number of labelled instances for each class.
	sizeVec = np.empty([numClusters, 1])
	countAssignCenters = torch.FloatTensor(numClusters, 1).zero_().cuda()

	for epoch in range(0, argList.numEpochs):
		
		totalVal = 0
		
		## Resetting loss values after each epoch.
		runningKLLoss = 0.0
		runningTotalLoss = 0.0
		runningReconError = 0.0
		runningClusterLoss = 0.0
		runningLabelledLoss = 0.0
		runningUnlabelledLoss = 0.0

		## Lists to hold the NMI and purity values generated after each iteration.
		nmiLabList = []
		nmiUnlabList = []
		purityLabList = []
		purityUnlabList = []

		lambdaVal = 0.1	
		
		if (epoch == 0):
			
			## Set the AutoEncoder to evaluation mode.
			Model.eval()

			## Initialise matrix to hold the cluster centers.
			# clusterCenters = Variable(torch.FloatTensor(10, 32))
			clusterCenters = Variable(torch.FloatTensor(numClusters, argList.embeddingSpace)).cuda()

			## Compute the cluster centers using constrained K-Means Clustering.
			for i in range(numClusters):

				## Obtaining the labelled data points corresponding to the current label.
				dataSamples = trainDataLab.dataDict[i]

				## Convert all data samples to a single pyTorch tensor.
				dataSamples = torch.stack(dataSamples)

				## Computing the number of data samples for the current label.
				numSamples = dataSamples.size()[0]

				## Assigning this number to the relevant row in sizeVec.
				sizeVec[i] = numSamples

				## Creating a feature vector container for holding the embeddings for the 
				## current data samples.
				# featureVec = torch.FloatTensor(numSamples, 32)
				featureVec = torch.FloatTensor(numSamples, argList.embeddingSpace).cuda()

				## Computing the embeddings for each image.
				for j in range(numSamples):

					## Extracting a individual image.
					image = dataSamples[j]
					image = image.float()

					## Filling the relevant values in the feature vector.
					# featureVec[j, ] = Model.encode(Variable(image.unsqueeze(0)))[0].data
					featureVec[j, ] = Model.encode(Variable(image.cuda().unsqueeze(0)))[0].data

				## Computing the cluster centers for the current label.
				clusterCenters[i, ] = Variable(torch.mean(featureVec, dim = 0))

				## print("Processed label : ", str(i))
			
			# sizeVec = torch.from_numpy(sizeVec).type(torch.FloatTensor) 
			sizeVec = torch.from_numpy(sizeVec).type(torch.FloatTensor).cuda()
			
			## Copy sizeVec to countAssignCenters.
			countAssignCenters.copy_(sizeVec)
			
		if(len(trainDataLab) > len(trainDataUnlab)):
			maxSamples = len(trainDataLab)
			maxBatchSize = argList.labBatchSize
		else:
			maxSamples = len(trainDataUnlab)
			maxBatchSize = argList.unlabBatchSize

		## Set the AutoEncoder to training mode.
		Model.train()

		for i in range(0, int(maxSamples / maxBatchSize)):

			totalVal += 1
			
			## Loading in the batch of labelled data.
			imgBatch1, imgBatch2, labBatch1, labDataLoader = loadLabeledBatch(trainDataLab, labDataLoader)
			imgBatch1, imgBatch2, labBatch1 = imgBatch1.float(), imgBatch2.float(), labBatch1.float()
			imgBatch1, imgBatch2, labBatch1 = imgBatch1.cuda(), imgBatch2.cuda(), labBatch1.cuda()

			## Loading in the batch of unlabelled data.
			imgBatch3, labBatch3, UnlabDataLoader = loadUnlabeledBatch(trainDataUnlab, UnlabDataLoader)
			imgBatch3, labBatch3 = imgBatch3.float(), labBatch3.float()
			imgBatch3, labBatch3 = imgBatch3.cuda(), labBatch3.cuda()

			## Computing the forward pass for all the image batches.
			encoded1, reconstructions1 = Model(Variable(imgBatch1))
			encoded2, reconstructions2 = Model(Variable(imgBatch2))
			encoded3, reconstructions3 = Model(Variable(imgBatch3))

			## Computing the reconstruction error.
			reconError1 = mseLoss(reconstructions1, Variable(imgBatch1))
			reconError2 = mseLoss(reconstructions2, Variable(imgBatch2))
			reconError3 = mseLoss(reconstructions3, Variable(imgBatch3))

			## Normalising the reconstruction error.
			reconError = (reconError1 + reconError2 + reconError3) / 3.0
			runningReconError += reconError.item()

			## Concatenate image batches and encodings for the labelled data.
			labels12 = torch.cat([labBatch1, labBatch1], dim = 0)
			images12 = torch.cat([imgBatch1, imgBatch2], dim = 0)
			encodings12 = torch.cat([encoded1, encoded2], dim = 0)

			## Expanding the encodings and the cluster centers.
			encodingsAllExpand3 = encoded3.unsqueeze(1).expand(argList.unlabBatchSize , numClusters, argList.embeddingSpace)
			encodingsAllExpand12 = encodings12.unsqueeze(1).expand(2 * argList.labBatchSize, numClusters, argList.embeddingSpace)
			clusterCentersExpand3 = clusterCenters.clone().unsqueeze(0).expand(argList.unlabBatchSize, numClusters, argList.embeddingSpace)
			clusterCentersExpand12 = clusterCenters.clone().unsqueeze(0).expand(2 * argList.labBatchSize, numClusters, argList.embeddingSpace)

			## Computing the distances of the encodings from the cluster centers.
			distMat3 = torch.pow(encodingsAllExpand3 - clusterCentersExpand3, 2).sum(2)
			distMat12 = torch.pow(encodingsAllExpand12 - clusterCentersExpand12, 2).sum(2)
			
			## Computing the cluster center label having the minimum distance from the particular image encoding.
			_ , predClus3 = torch.min(distMat3, dim = 1)
			_ , predClus12 = torch.min(distMat12, dim = 1)

			## Computing the assignment probabilities based on distances from the cluster centers.
			distanceProb3 = F.softmax(-distMat3, dim = 1)
			distanceProb12 = F.softmax(-distMat12, dim = 1)

			for li in range(labels12.size(0)):

				## Extracting the class label.
				classLabel = labels12[li].cpu().numpy()

				## Updating the count of the labelled instances for the particular label.
				sizeVec[classLabel] += 1

				## Computing the distance of the cluster center for the particular label from the encodings.
				distV = clusterCenters[classLabel, :] - encodings12[li, :]

				## updating the cluster centers.
				clusterCenters[classLabel, :] = clusterCenters[classLabel, :] - Variable(1 / (sizeVec[classLabel])) * distV

			## Creating the image pairs along with their similarity index for the labelled data.
			labelInfoListSimL, labelInfoListDissimL = createLabelInfoList(images12, labels12)

			## Extracting the scores for the image pairs.
			pSL, qSL = createConstraintList(labelInfoListSimL.cuda(), distanceProb12)
			pDL, qDL = createConstraintList(labelInfoListDissimL.cuda(), distanceProb12)

			## Computing the pairwise loss for the labelled data.
			lossSLab = klDivergenceSim(pSL, qSL) + klDivergenceSim(qSL, pSL)
			lossDLab = klDivergenceDissim(pDL, qDL) + klDivergenceDissim(qDL, pDL)

			## Normalizing the KL-Loss for the labelled data.
			lossKLLab1 = (lossSLab / len(pSL))
			lossKLLab2 = (lossDLab / len(pDL))

			## Computing the cluster loss for the labelled data.
			lossCLab = clusterLoss(clusterCenters, encodings12, labels12)

			## Normalizing the cluster loss for the labelled data.
			lossCLab /= len(labels12)

			## If unlabelled data loss contribution is to be considered.
			if (lambdaVal != 0):

				## Using the unlabelled data to update the cluster centers.
				for k in range(predClus3.size(0)):

					## Extracting the class predicted on the basis of the softmax score.
					softmaxScoreClass = predClus3[k].data.cpu().numpy()
				   
					## Updating the count of the labelled instances for the particular label.
					countAssignCenters[softmaxScoreClass] += 1

					## Computing the distance of the cluster center for the particular label from the encodings.
					distV = clusterCenters[softmaxScoreClass, :] - encoded3[k, :]

					## Updating the cluster centers.
					clusterCenters[softmaxScoreClass, :] = clusterCenters[softmaxScoreClass, :] - Variable(1 / (countAssignCenters[softmaxScoreClass])) * distV

			## Creating the image pairs along with their similarity index for the labelled data.
			labelInfoListSimU, labelInfoListDissimU = createLabelInfoList(imgBatch3, labBatch3)

			## Extracting the scores for the image pairs.        
			pSU, qSU = createConstraintList(labelInfoListSimU.cuda(), distanceProb3)
			pDU, qDU = createConstraintList(labelInfoListDissimU.cuda(), distanceProb3)

			## Computing the pairwise loss for the unlabelled data.
			lossSUnlab = klDivergenceSim(pSU, qSU) + klDivergenceSim(qSU, pSU)
			lossDUnlab = klDivergenceDissim(pDU, qDU) + klDivergenceDissim(qDU, pDU)

			## Normalizing the KL-Loss for the unlabelled data.
			lossKLUnlab1 = (lossSUnlab) / (len(pSU))
			lossKLUnlab2 = (lossDUnlab) / (len(pDU))

			## Computing the cluster loss for the unlabelled data.
			lossCUnlab = clusterLoss(clusterCenters, encoded3, predClus3)

			## Normalizing the cluster loss for the unlabelled data.
			lossCUnlab /= len(labBatch3)

			## Computing the total pairwise KL Loss.
			pairwiseLossTot = lossKLLab1 + lossKLLab2 + lambdaVal * (lossKLUnlab1 + lossKLUnlab2)
			runningKLLoss += pairwiseLossTot.item()

			## Computing the total clustering loss.
			clusteringLossTot = lossCLab + lambdaVal * lossCUnlab
			runningClusterLoss += clusteringLossTot.item()

			## Computing the total combined loss.
			totalLoss = reconError + pairwiseLossTot + clusteringLossTot
			runningTotalLoss += totalLoss.item()

			## Computing the labelled loss.
			lossLabelledTotal = lossKLLab1 + lossKLLab2 + lossCLab
			runningLabelledLoss += lossLabelledTotal.item()

			## Computing the unlabelled loss.
			lossUnlabelledTotal = lossKLUnlab1 + lossKLUnlab2  + lossCUnlab
			runningUnlabelledLoss += lossUnlabelledTotal.item()

			## Initialise all parameter gradients to 0.
			optimizer.zero_grad()

			## Backpropagation.
			totalLoss.backward()

			## Optimisation.
			optimizer.step()
			
			## Detach the cluster centers.
			clusterCenters = clusterCenters.detach()
			
			del totalLoss, encoded1, encoded2, encoded3
			torch.cuda.empty_cache()
			gc.collect()

			## Computing the NMI for the labelled data.
			trainNMILab = normalized_mutual_info_score(predClus12.data.cpu().numpy(), labels12.cpu())
			nmiLabList.append(trainNMILab)

			## Computing the NMI for the unlabelled data.
			trainNMIUnlab = normalized_mutual_info_score(predClus3.data.cpu().numpy(), labBatch3.cpu())
			nmiUnlabList.append(trainNMIUnlab)

			## Computing the Purity for the labelled data.
			trainPurityLab = purityScore(predClus12.data.cpu().numpy(), labels12.cpu())
			purityLabList.append(trainPurityLab)

			## Computing the Purity for the unlabelled data.
			trainPurityUnlab = purityScore(predClus3.data.cpu().numpy(), labBatch3.cpu())
			purityUnlabList.append(trainPurityUnlab)

			# if (i % 5 == 0):

			# 	print("Processed Epoch : ", epoch)
			# 	print("Lambda Value : ", lambdaVal)
			# 	print("Total Loss : ", runningTotalLoss / totalVal)
			# 	print("Pairwise KL Loss : ", runningKLLoss / totalVal)
			# 	print("Clustering Loss : ", runningClusterLoss / totalVal)
			# 	print("Labelled Data Loss : ", runningLabelledLoss / totalVal)
			# 	print("Reconstruction Error : ", runningReconError / totalVal)
			# 	print("Unlabelled Data Loss : ", runningUnlabelledLoss / totalVal)
			# 	print(" ")
			
		## Updating the lists for NMI and Purity score.
		trainNMILabelledList.append(np.average(nmiLabList))
		trainNMIUnlabelledList.append(np.average(nmiUnlabList))
		trainPurityLabelledList.append(np.average(purityLabList))
		trainPurityUnlabelledList.append(np.average(purityUnlabList))

		## Update the different lists for holding the loss.
		KLLossList.append(runningKLLoss / totalVal)
		totalLossList.append(runningTotalLoss / totalVal)
		reconErrorList.append(runningReconError / totalVal)
		clusterLossList.append(runningClusterLoss / totalVal)
		labelledLossList.append(runningLabelledLoss / totalVal)
		unLabelledLossList.append(runningUnlabelledLoss / totalVal)

		print("Processed Epoch : ", epoch)
		print("Lambda Value : ", lambdaVal)
		print("Total Loss : ", runningTotalLoss / totalVal)
		print("Pairwise KL Loss : ", runningKLLoss / totalVal)
		print("Clustering Loss : ", runningClusterLoss / totalVal)
		print("Labelled Data Loss : ", runningLabelledLoss / totalVal)
		print("Reconstruction Error : ", runningReconError / totalVal)
		print("Unlabelled Data Loss : ", runningUnlabelledLoss / totalVal)
		print(" ")
		print("Training NMI Labelled : ", np.average(nmiLabList))
		print("Training NMI Unlabelled : ", np.average(nmiUnlabList))
		print("Training Purity Labelled : ", np.average(purityLabList))
		print("Training Purity Unlabelled : ", np.average(purityUnlabList))
		print(" ")


	## Plotting the loss.
	plt.plot(totalLossList)
	plt.xlabel('Epoch')
	plt.ylabel('Total Loss')
	plt.title('Total Loss')
	plt.savefig('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage' + str(int(argList.percentLabData * 100)) +'/λ0.1Results/TotalLossλ01.png')

	## Saving the pretrained model.
	torch.save(Model, './TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/AutoEncoderλ01.pkl')
	torch.save(Model.state_dict(), './TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/AutoEncoderλ01.pth')

	## Saving the cluster centers.
	dataCenters = clusterCenters.data.cpu().numpy()
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/clusterCenters', dataCenters, allow_pickle = True)

	## Saving the different tracking lists.
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/KLLossListλ01', KLLossList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/totalLossListλ01', totalLossList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/reconErrorListλ01', reconErrorList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/clusterLossListλ01', clusterLossList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/labelledLossListλ01', labelledLossList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/unLabelledLossListλ01', unLabelledLossList)

	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMILabelledListλ01', trainNMILabelledList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01', trainNMIUnlabelledList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainNMIUnlabelledListλ01', trainPurityLabelledList)
	np.save('./TrainingResults/' + str(argList.dataSet) + '/NoPreTraining/Percentage'+ str(int(argList.percentLabData * 100)) + '/λ0.1Results/trainPurityUnlabelledListλ01', trainPurityUnlabelledList)