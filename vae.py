import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal
import numpy

from c302 import *
from tcn import TemporalConvNet

class TCN(PyroModule):

    n_channels = [25] * 4
    kernel_size = 3
    dropout = 0.25

    def __init__(self, input_size, output_size, num_channels=None, kernel_size = None, dropout = None):
        super(TCN, self).__init__()
        if dropout is not None: 
            self.dropout = dropout
        if kernel_size is not None: 
            self.kernel_size = kernel_size
        if num_channels is not None: 
            self.n_channels = num_channels
        self.tcn = TemporalConvNet(input_size, self.n_channels, self.kernel_size, dropout=self.dropout)
        self.linear = nn.Linear(self.n_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        print("TCN.x", x.shape)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        print("TCN.tcn", output.shape)
        output = self.linear(output).double()
        print("TCN.linear", output.shape)
        return self.sig(output)

class Encoder(PyroModule):
    def __init__(self, neuron_dim, wicks_dim, gap_dim):
        super().__init__()
        neuron_dim = 300
        self.neuron_dim = neuron_dim
        self.wicks_dim = wicks_dim
        self.gap_dim = gap_dim
        self.neuron = TCN(input_size=neuron_dim, output_size=neuron_dim*6)
        self.wick = TCN(input_size=neuron_dim, output_size=wicks_dim *6)
        self.gap = TCN(input_size=neuron_dim, output_size=gap_dim*2)

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.input_size)
        # then compute the hidden units
        neuron = self.neuron(x)
        neuron = neuron.reshape((6, self.neuron_dim, neuron.shape[1] ))
        neuron_loc = neuron[0:3]
        neuron_scale = neuron[3:6]
        print(neuron_loc.shape, neuron_scale.shape)

        wick = self.wick(x)
        wick = wick.reshape((6, self.wicks_dim, wick.shape[1] ))
        wicks_loc = wick[0:3]
        wicks_scale = wick[3:6]

        gap = self.gap(x)
        gap = gap.reshape((6, self.gap_dim, wick.shape[1] ))
        gap_loc, gap_scale = gap[0:3], gap[3:6]
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        return neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale

class VAElegans(PyroModule):
    def __init__(self, model: RecurrentNematode, neuron_dim, wicks_dim, gap_dim):
        super().__init__()
        self.encoder = Encoder(neuron_dim, wicks_dim, gap_dim)
        self.decoder = model

        # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = self.encoder(x)
        print("neuron_loc", neuron_loc.shape)
        # decode the image (note we don't sample in image space)
        self.decoder.assign(neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale )
        result = self.decoder()
        return result
    def forward(self, x):
            # setup hyperparameters for prior p(z)
            neuron_loc = torch.zeros(( 3, self.encoder.neuron_dim))
            neuron_scale = torch.ones((  3, self.encoder.neuron_dim))
            wicks_loc = torch.zeros((  3, self.encoder.wicks_dim))
            wicks_scale = torch.ones((  3, self.encoder.wicks_dim))
            gap_loc = torch.zeros((  self.encoder.gap_dim))
            gap_scale =  torch.ones((  self.encoder.gap_dim))
            # TODO: Assigning z_loc and z_scale
            self.decoder.assign(neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale )
            result = self.decoder( y=x)

            # score against actual images
            # pyro.sample("obs", dist.Bernoulli(result).to_event(1), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("neuron_latent", dist.Normal(neuron_loc, neuron_scale).to_event(1))
            pyro.sample("gap_latent", dist.Normal(gap_loc, gap_scale).to_event(1))
            pyro.sample("wick_latent_0", dist.Normal(wicks_loc[0], wicks_scale[0]).to_event(1))
            pyro.sample("wick_latent_1", dist.Normal(wicks_loc[1], wicks_scale[1]).to_event(1))
            pyro.sample("wick_latent_0", dist.Normal(wicks_loc[2], wicks_scale[2]).to_event(1))

