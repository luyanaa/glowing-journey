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

class Encoder(PyroModule):
    def __init__(self, input_size, hidden_dim, neuron_dim, synapse_dim):
        super().__init__()
        self.input_size = input_size
        # setup the three linear transformations used
        levels = 4
        nhid = 150
        n_channels = [nhid] * levels

        self.fc1 = nn.Sequential(TemporalConvNet(input_size, num_channels=n_channels), nn.Linear(n_channels[-1], hidden_dim))

        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.input_size)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        return z_loc, z_scale

class VAElegans(PyroModule):
    def __init__(self, model: RecurrentNematode, hidden_dim):
        self.decoder = model
        self.encoder = Encoder(hidden_dim, self.decoder.model.NeuronSize, self.decoder.model.synapseSize)

        # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # decode the image (note we don't sample in image space)
        res = self.decoder()
        return res
    def forward(self, x):
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # TODO: Assigning z_loc and z_scale
            self.decoder.assign()
            result = self.decoder()

            # score against actual images
            pyro.sample("obs", dist.Bernoulli(res).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
