import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal

# https://www.sciencedirect.com/science/article/pii/S0168010223000068
# Choose the dx/dt + d2x/dt2 model. 
class ASH(PyroModule):
    def __init__(self, initInput, k, name=="left"):
        self.pastInput = initInput
        self.pastpastInput = initInput

        self.delta_b = PyroParam(torch.zeros(1)) # Unknown Parameter
        # Original Model is b1-b2 one, hard to implement, choose arthimetic average. 
        self.Xe = PyroSample(Normal(0.9800, 0.0511)) 
        self.k1 = PyroParam(torch.tensor(1.),) 
        self.k2 = PyroParam(torch.tensor(1.),)
        
    def forward(self, input):
        I_1 = - self.k1 * (input-self.pastInput) / timeStep
        I_2 = - self.k2 * ((input-self.pastInput) / timeStep-(self.pastInput-self.pastpastInput)/timeStep) / timeStep
        if I_2 < 0.14:
            I_2 = 0.14
        if self.X > 0.1: 
            tau = (0.02*(torch.log(self.X)) + 0.11)
        else :
            tau = (0.02*(torch.log(0.1)) + 0.11)
        dX = I_1 + I_2 - (self.X-self.Xe) / tau
        self.X = self.X+dX

# https://elifesciences.org/articles/21629
class AWB(PyroModule):
    def __init__


# We mainly focuses on odor, 
class sensoryModel(PyroModule):
    def __init__