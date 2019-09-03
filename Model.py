import torch
from torch import nn
import torch.nn.functional as F

## Defining the architecture of the Encoder.
class Encoder(nn.Module):

    def __init__(self, dataset):

        super(Encoder, self).__init__()

        ## Convolutional layer with 32 3*3 filters.
        if (dataset == 'MNIST' or dataset == 'USPS'):
            self.eConv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        if (dataset == 'CIFAR10' or dataset == 'FRGC'):
            self.eConv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        if (dataset == 'YTF'):
            self.eConv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 2, padding = 1)

        ## Apply Instance Normalization.
        self.eBn1 = nn.InstanceNorm2d(num_features = 32)

        ## Convolutional layer with 64 5*5 filters.
        if (dataset == 'USPS'):
            self.eConv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 1)
        if (dataset == 'YTF'):
            self.eConv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7, stride = 2, padding = 1)
        else:
            self.eConv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)

        ## Apply Instance Normalization.
        self.eBn2 = nn.InstanceNorm2d(num_features = 64)

        ## Convolutional layer with 128 5*5 filters.
        if (dataset == 'YTF'):
            self.eConv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 7, stride = 1, padding = 1)
        else:
            self.eConv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 1)

        ## Apply Instance Normalization.
        self.eBn3 = nn.InstanceNorm2d(num_features = 128)

        ## Fully Connected Layer.
        if (dataset == 'MNIST' or dataset == 'USPS' or dataset == 'YTF'):
            self.eFc = nn.Linear(in_features = 128 * 4 * 4, out_features = 32)
        if (dataset == 'CIFAR10' or dataset == 'FRGC'):
            self.eFc = nn.Linear(in_features = 128 * 5 * 5, out_features = 32)

    ## Forward Pass.
    def forward(self, x):

        x = F.dropout(F.leaky_relu(self.eBn1(self.eConv1(x))), p = 0.1, training = self.training)

        x = F.dropout(F.leaky_relu(self.eBn2(self.eConv2(x))), p = 0.1, training = self.training)

        x = F.dropout(F.leaky_relu(self.eBn3(self.eConv3(x))), p = 0.1, training = self.training)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        return torch.tanh(self.eFc(x))

## Defining the architecture of the Decoder.
class Decoder(nn.Module):

    def __init__(self, dataset):

        super(Decoder, self).__init__()

        ## Fully Connected Layer.
        if (dataset == 'MNIST' or dataset == 'USPS' or dataset == 'YTF'):
            self.dFc = nn.Linear(in_features = 32, out_features = 128 * 4 * 4)
        if (dataset == 'CIFAR10' or dataset == 'FRGC'):
            self.dFc = nn.Linear(in_features = 32, out_features = 128 * 5 * 5)

        ## Apply Instance Normalization.
        self.dBn3 = nn.InstanceNorm2d(num_features = 128)

        ## Convolutional layer with 64 5*5 filters.
        if (dataset == 'YTF'):
            self.dConv3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 7, stride = 2, padding = 1)
        else:
            self.dConv3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 1, padding = 1)

        ## Apply Instance Normalization.
        self.dBn2 = nn.InstanceNorm2d(num_features = 64)

        ## Convolutional layer with 32 5*5 filters.
        if (dataset == 'USPS'):
            self.dConv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5, stride = 1, padding = 1)
        if (dataset == 'YTF'):
            self.dConv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 7, stride = 2, padding = 1)
        else:
            self.dConv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5, stride = 2, padding = 1)

        ## Apply Instance Normalization.
        self.dBn1 = nn.InstanceNorm2d(num_features = 32)

        ## Convolutional layer with 1 4*4 filter.
        if (dataset == 'MNIST' or dataset == 'USPS'):
            self.dConv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 2)
        if (dataset == 'CIFAR10' or dataset == 'FRGC'):
            self.dConv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 4, stride = 2)
        if (dataset == 'YTF'):
            self.dConv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 5, stride = 2)


    ## Forward Pass.
    def forward(self, z):

        z = self.dFc(z)

        if (self.dataset == 'MNIST' or self.dataset == 'USPS' or self.dataset == 'YTF'):
            z = F.dropout(torch.tanh(self.dBn3(z.view(z.size(0), 128, 4, 4))), p = 0.1, training = self.training)
        if (self.dataset == 'CIFAR10' or self.dataset == 'FRGC'):
            z = F.dropout(torch.tanh(self.dBn3(z.view(z.size(0), 128, 5, 5))), p = 0.1, training = self.training)

        z = F.dropout(F.leaky_relu(self.dBn2(self.dConv3(z))), p = 0.1, training = self.training)

        z = F.dropout(F.leaky_relu(self.dBn1(self.dConv2(z))), p = 0.1, training = self.training)

        z = F.leaky_relu(self.dConv1(z))

        return z

## Wrapper class for the entire autoencoder.
class AutoEncoder(nn.Module):

    def __init__(self, Encoder, Decoder):

        super(AutoEncoder, self).__init__()

        self.Encoder = Encoder
        self.Decoder = Decoder

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)

    def forward(self, x):

        Encoding = self.Encoder(x)
        Reconstruction = self.Decoder(Encoding)

        return Encoding, Reconstruction