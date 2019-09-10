import torch
from torch import nn
import torch.nn.functional as F

## Defining the architecture of the Encoder.
class frgcEncoder(nn.Module):

    def __init__(self):

        super(frgcEncoder, self).__init__()

        ## Convolutional layer with 32 3*3 filters.
        self.eConv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        ## Apply Instance Normalization.
        self.eBn1 = nn.InstanceNorm2d(num_features = 32)

        ## Convolutional layer with 64 5*5 filters.        
        self.eConv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)

        ## Apply Instance Normalization.
        self.eBn2 = nn.InstanceNorm2d(num_features = 64)

        ## Convolutional layer with 128 5*5 filters.
        self.eConv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 1)

        ## Apply Instance Normalization.
        self.eBn3 = nn.InstanceNorm2d(num_features = 128)

        ## Fully Connected Layer.
        self.eFc = nn.Linear(in_features = 128 * 5 * 5, out_features = 32)

    ## Forward Pass.
    def forward(self, x):

        x = F.dropout(F.leaky_relu(self.eBn1(self.eConv1(x))), p = 0.1, training = self.training)

        x = F.dropout(F.leaky_relu(self.eBn2(self.eConv2(x))), p = 0.1, training = self.training)

        x = F.dropout(F.leaky_relu(self.eBn3(self.eConv3(x))), p = 0.1, training = self.training)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        return torch.tanh(self.eFc(x))

## Defining the architecture of the Decoder.
class frgcDecoder(nn.Module):

    def __init__(self):

        super(frgcDecoder, self).__init__()

        ## Fully Connected Layer.
        self.dFc = nn.Linear(in_features = 32, out_features = 128 * 5 * 5)

        ## Apply Instance Normalization.
        self.dBn3 = nn.InstanceNorm2d(num_features = 128)
        
        self.dConv3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 1, padding = 1)

        ## Apply Instance Normalization.
        self.dBn2 = nn.InstanceNorm2d(num_features = 64)

        ## Convolutional layer with 32 5*5 filters.
        self.dConv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5, stride = 2, padding = 1)

        ## Apply Instance Normalization.
        self.dBn1 = nn.InstanceNorm2d(num_features = 32)

        ## Convolutional layer with 1 4*4 filter.        
        self.dConv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 4, stride = 2)
        
    ## Forward Pass.
    def forward(self, z):

        z = self.dFc(z)
        
        z = F.dropout(torch.tanh(self.dBn3(z.view(z.size(0), 128, 5, 5))), p = 0.1, training = self.training)

        z = F.dropout(F.leaky_relu(self.dBn2(self.dConv3(z))), p = 0.1, training = self.training)

        z = F.dropout(F.leaky_relu(self.dBn1(self.dConv2(z))), p = 0.1, training = self.training)

        z = F.leaky_relu(self.dConv1(z))

        return z

## Wrapper class for the entire autoencoder.
class frgcAutoEncoder(nn.Module):

    def __init__(self, Encoder, Decoder):

        super(frgcAutoEncoder, self).__init__()

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