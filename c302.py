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

class GRUModel(PyroModule):
    _neuron_list = []
    hidden_size = 8
    num_layers = 2
    def __init__(self, neuronSize):
        self.neuronSize = neuronSize
        for i in range(neuronSize):
            self._neuron_list.append(PyroModule[nn.Sequential](PyroModule[nn.GRU](1, self.hidden_size, num_layers=self.num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](self.hidden_size, 1)))
    def __forward__(self, inputSignal):
        result = torch.zeros(self.neuronSize)
        for i in range(inputSignal):
            result[i]=self._neuron_list[i](inputSignal[i])
        return result
            
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
        self.neuronSize = neuronSize

    # Input Current, Output Voltage
    def forward(self, inputSignal):
        current = torch.sum(self.G*self.X(torch.expand(self.V, self.E.shape) - self.E), axis = 1)
        dv = (inputSignal - current) / self.C
        v = dv * timeStep
        return v

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

# A Rough Estimation for Gap Junction and Synaptic ones. 
# Refer to https://www.mdpi.com/2227-7390/11/11/2442

# Extrasynaptic Connectome Refer to https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005283#sec002
# Monoamine
# Serotonin: https://cell.com/cell/pdf/S0092-8674(23)00419-1.pdf , looks like a current input
# Dealt as a basic conductance model
class GeneralSynapse(PyroModule):
    def __init__(self, synapseInput, synapseOutput, g=None):
        self.synapseInput = synapseInput
        self.synapseOutput = synapseOutput
        if g is None:
            self.g = PyroSample(dist.Normal(0., 1.).expand([len(synapseInput)]))
        else:
            self.g = g
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(302)
        preVoltage = preVoltage[self.synapseInput]
        postVoltage = postVoltage[self.synapseOutput]
        current = self.g*(postVoltage-preVoltage)
        output[self.synapseOutput] = current
        return output

# Refer to https://hal.science/hal-03705452/file/new_simple_model_pg.pdf
# Wicks et al. (1999)
class WicksSynapse(PyroModule):
    inputSize = 0
    def __init__(self, synapseInput, synapseOutput, g_max=None, V_rest=None, V_slope=None):
        if V_rest is None or V_slope is None:
            # Assign
            # In the following, we set Vslope = 15 mV, gsyn = 0.6 nS and Vrest = −76 mV (Wicks et al., 1996).
            self.inputSize = len(synapseInput)
            self.synapseInput = synapseInput
            self.synapseOutput = synapseOutput
            self.g_max = PyroSample(dist.Normal(0.6, 1.).expand([self.inputSize]))
            self.V_rest = PyroSample(dist.Normal(15, 1.).expand([self.inputSize]))
            self.V_slope = PyroSample(dist.Normal(-76, 1.).expand([self.inputSize]))
        else:
            self.g_max = g_max
            self.V_rest = V_rest
            self.V_slope = V_slope
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(302)
        preVoltage = preVoltage[self.synapseInput]
        postVoltage = postVoltage[self.synapseOutput]
        current = self.g_max * 1 / (1+torch.exp((preVoltage-self.V_rest) / self.V_slope)) * (postVoltage)
        output[self.synapseOutput] = current
        return output

# Using GRU for Synapse Simulation for Synapse Plasticity
class RecurrentSynapse(PyroModule):
    hidden_size = 8
    num_layers = 2
    def __init__(self, synapseInput, synapseOutput):
        self.RNN = []
        for i in range (synapseInput):
            self.RNN.append(PyroModule[nn.Sequential](PyroModule[nn.GRU](2, self.hidden_size, num_layers=self.num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](self.hidden_size, 1)))
        self.synapseInput=synapseInput
        self.synapseOutput=synapseOutput
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(302)
        for i in range(self.synapseInput):
            output[self.synapseOutput[i]] += self.RNN[i](( preVoltage[i], postVoltage[i]))
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

class NematodeForStep(PyroModule):
    # Work for Single-Step Inference for MCMC/Sequential MC and Bayesian Filter.
    # Need a wrapper for SVI. 
    def __init__ (self, InputSize, ConnectomeSize, OutputSize, batch_sizes):
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
        self.expected_input_dim = 2 if batch_sizes is not None else 3
        
    def forward(self, InputSignal): 
        SensoryOutput = self.SensoryNeuron(Input)
        SensoryOutput = SensoryOutput + InputSignal
                # Model extrasynaptic things in Connectome
                # Calculate Pre-Synaptic and Post-Synaptic Voltage Difference here. 
        self.ConnectomeOutput = self.SensoryToInterneuron(SensoryOutput, self.ConnectomeOutput)
        self.ConnectomeOutput = self.Interneuron(self.ConnectomeOutput)
        self.MotorOutput = self.InterneuronToMotor(self.MotorOutput, self.ConnectomeOutput)
        self.MotorOutput = self.MotorNeuron(self.MotorOutput)
        return (SensoryOutput, self.ConnectomeOutput, self.MotorOutput)
    
    def step(self, state, y=None):
        self.t += 1
        state["z"] = pyro.sample(
            "z_{}".format(self.t),
            dist.Normal(self.forward(state["z"]), self.SNR).to_event(302),
        )
        y = pyro.sample(
            "y_{}".format(self.t), dist.Normal(state["z"][mask], self.sigma), obs=y
        )
        return state["z"], y

def readConnectome(path):
    Sensory = pandas.read_excel(path, sheet_name="Sensory")
    Connectome = pandas.read_excel(path, sheet_name="Connectome")
    NeuronsToMuscle = pandas.read_excel(path, sheet_name="NeuronsToMuscle")
    Sensory = Sensory[["Neuron", "Neurotransmitter"]]
    Connectome = Connectome

    # Neurons
    NeuronsToMuscle = NeuronsToMuscle

# Excluding Sensory from Connectome

    model = nematode(Sensory.shape[0], Connectome.shape[0], NeuronsToMuscle.shape[0])

    return model
