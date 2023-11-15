import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal

# import sensory, motor
timeStep = 0.001

# For Neuron: Input Current, Output Voltage (Easy to log, easy to calculate Vpre-Vpost)
# For Synapse: Input Voltage, Output Current. 

# Neuron Type: Conductance Model (Done), Sensory, Motor (Need case-by-case solution)
# Synapse Type: Gap Junction, Chemical Synapse (A variety of), extrasynaptic 

class conductanceModel (PyroModule):
    def __init__(self, neuronSize, E=None, G=None, neuronCapacity=None):
        self.V = torch.Tensor(neuronSize)
        if E is None:
            self.E = PyroSample(dist.Normal(0., 1.).expand([neuronSize, neuronSize]).to_event(2))
            self.G = PyroSample(dist.Normal(0., 1.).expand([neuronSize, neuronSize]).to_event(2))
            self.C = PyroSample(dist.Normal(0., 1.).expand([neuronSize]).to_event(1))
        else: 
            self.E = E
            self.G = G
            self.C = neuronCapacity
        self.SNR = 0.0
        self.neuronSize = neuronSize

    # Input Current, Output Voltage
    def forward(self, inputSignal):
        current = torch.sum(self.G*self.X(torch.expand(self.V, self.E.shape) - self.E), axis = 1)
        dv = (inputSignal - current) / self.C
        dv = dv + self.SNR * PyroSample(dist.Normal(0., 1.).expand([self.neuronSize]).to_event(1))
        v = dv * timeStep
        return v

class extrasynaptic(PyroModule):
    def __init__(self):
        return
    def forward():
        return 
    
class neuronLayer(PyroModule):
    def __init__(self, neuronDynamicsCallback, neuronInput, neuronOutput, neuronDynamics, neuronParameter):
        for neuronModel in neuronDynamicsCallback:
            self._neuron_layer = neuronModel(neuronInput, neuronOutput, neuronParameter)
        for neuron in neuronDynamics:
            self._neuron_layer[neuron].addParam(neuronParameter[neuron])
    def forward(self, inputSignal):
        for layer in self._neuron_layer:
            output = output + layer(inputSignal)
        return output

class synapseLayer(PyroModule):
    def __init__(self, synapseCallback, synapseInput, synapseOutput, synapseParameter):
        self._synpase_list = []
        for singleSynapse in synapseCallback:
            self._synpase_list.append(singleSynapse(synapseInput, synapseOutput, synapseParameter))
    
    def forward(self, inputSignal):
        for layer in self._synpase_list:
            output = output + layer(inputSignal)
        return output

class c302(PyroModule):
    def __init__ (self, InputSize, ConnectomeSize, OutputSize):
        # Sensory Input/Neuron
        self.SensoryNeuron = sensoryModel(InputSize, InputSize)
        # From Sensory to Connectome
        self.SensoryToInterneuron = synapseLayer(InputSize, ConnectomeSize)
        # Connectome Neuron
        self.Interneuron = neuronLayer(ConnectomeSize)
        # From Interneuron to Motor Neuron
        self.InterneuronToMotor = synapseLayer(ConnectomeSize, OutputSize) 
        # Motor neuron
        self.MotorNeuron = neuronLayer(OutputSize)
        # Motor Output
        # self.OutputLayer = nn.Linear(OutputSize)

    def forward(self, Input, InputSignal, experiment): 
        if experiment == "simulation":
            SensoryOutput = self.SensoryNeuron(Input)
            SensoryOutput = SensoryOutput + InputSignal
            # Model extrasynaptic things in Connectome
            ConnectomeOutput = self.SensoryToInterneuron(SensoryOutput)
            ConnectomeOutput = self.Interneuron(ConnectomeOutput)
            MotorOutput = self.InterneuronToMotor(ConnectomeOutput)
            MotorOutput = self.MotorNeuron(MotorOutput)
            return (SensoryOutput, ConnectomeOutput, MotorOutput)

        if experiment == "clamp":




def readConnectome(path):

    Sensory = pandas.read_excel(path, sheet_name="Sensory")
    Connectome = pandas.read_excel(path, sheet_name="Connectome")
    NeuronsToMuscle = pandas.read_excel(path, sheet_name="NeuronsToMuscle")

    Sensory = Sensory[["Neuron", "Neurotransmitter"]]
    Connectome = Connectome

    # Neurons
    NeuronsToMuscle = NeuronsToMuscle

# Excluding Sensory from Connectome

    model = c302(Sensory.shape[0], Connectome.shape[0], NeuronsToMuscle.shape[0])

    return model
