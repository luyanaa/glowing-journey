import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal
from utils import timeStep
import numpy

# For Neuron: Input Current, Output Voltage (Easy to log, easy to calculate Vpre-Vpost)
# For Synapse: Input Voltage, Output Current. 

# Neuron Type: Conductance Model (Done), Sensory, Motor (Need case-by-case solution)
# Synapse Type: Gap Junction, Chemical Synapse (A variety of), extrasynaptic 

class GRUModel(PyroModule):
    def __init__(self, hidden_size=8, num_layers=2):
        super().__init__()
        self.model=(PyroModule[nn.Sequential](PyroModule[nn.GRU](1, hidden_size, num_layers=num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](hidden_size, 1)))
    def __forward__(self, inputSignal):
        return self.model(inputSignal)

class FallbackSensory(PyroModule):
    def __init__(self, hidden_size=8, num_layers=2):
        super().__init__()
        self.model=(PyroModule[nn.Sequential](PyroModule[nn.GRU](2, hidden_size, num_layers=num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](hidden_size, 1)))
    def __forward__(self, inputSignal, ExternalInput):
        return self.model(inputSignal, ExternalInput)


# https://www.sciencedirect.com/science/article/pii/S0168010223000068
# Choose the dx/dt + d2x/dt2 model. 
class ASH(PyroModule):
    def __init__(self, initInput=0.0):
        super().__init__()
        self.pastInput = initInput
        self.pastpastInput = initInput

        self.delta_b = PyroParam(torch.zeros(1)) # Unknown Parameter
        # Original Model is b1-b2 one, hard to implement, choose arthimetic average. 
        self.Xe = PyroSample(Normal(0.9800, 0.0511)) 
        self.k1 = PyroParam(torch.tensor(1.),) 
        self.k2 = PyroParam(torch.tensor(1.),)
        
    def forward(self, inputCurrent, input):
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
        return self.X
            
class conductanceModel (PyroModule):
    def __init__(self, E=None, G=None, neuronCapacity=None):
        super().__init__()
        self.V = torch.zeros(1)
        if E is None:
            self.E = PyroSample(dist.Normal(0., 1.))
            self.G = PyroSample(dist.Normal(0., 1.))
            self.C = PyroSample(dist.Normal(0., 1.))
        else: 
            self.E = E
            self.G = G
            self.C = neuronCapacity

    # Input Current, Output Voltage
    def forward(self, inputSignal):
        current = torch.sum(self.G*torch.expand(self.V, self.E.shape) - self.E, axis = 1)
        dv = (inputSignal - current) / self.C
        v = dv * timeStep
        return v

class neuronLayer(PyroModule):
    def __init__(self, neuronSize, neuronList):
        super().__init__()
        self._neuron_List = neuronList
        self.neuronSize = neuronSize
    def forward(self, inputSignal, externalInput):
        output = []
        for i in range(self.neuronSize):
            if inputSignal[i]!=0.0:
                output.append(self._neuron_List[i](inputSignal[i], externalInput[i]))
            else:
                output.append(self._neuron_List[i](inputSignal))
        output = torch.Tensor(output)
        return output

# A Rough Estimation for Gap Junction and Synaptic ones. 
# Refer to https://www.mdpi.com/2227-7390/11/11/2442

# Extrasynaptic Connectome Refer to https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005283#sec002
# Monoamine
# Serotonin: https://cell.com/cell/pdf/S0092-8674(23)00419-1.pdf , looks like a current input
# Dealt as a basic conductance model
class GeneralSynapse(PyroModule):
    def __init__(self, synapseInput, synapseOutput, synapseWeight, g=None):
        super().__init__()
        self.synapseInput = synapseInput
        self.synapseOutput = synapseOutput
        self.synapseWeight = synapseWeight
        if g is None:
            self.g = PyroSample(dist.Normal(0., 1.).expand([len(synapseInput)]))
        else:
            self.g = g
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(300)
        preVoltage = preVoltage[self.synapseInput]
        postVoltage = postVoltage[self.synapseOutput]
        current = self.g*(postVoltage-preVoltage)
        current = current * self.synapseWeight
        output[self.synapseOutput] = current
        return output

# Refer to https://hal.science/hal-03705452/file/new_simple_model_pg.pdf
# Wicks et al. (1999)
class WicksSynapse(PyroModule):
    inputSize = 0
    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_max=None, V_rest=None, V_slope=None):
        super().__init__()
        if V_rest is None or V_slope is None:
            # Assign
            # In the following, we set Vslope = 15 mV, gsyn = 0.6 nS and Vrest = −76 mV (Wicks et al., 1996).
            self.inputSize = len(synapseInput)
            self.synapseInput = synapseInput
            self.synapseOutput = synapseOutput
            self.synapseWeight = synapseWeight
            self.g_max = PyroSample(dist.Normal(0.6, 1.).expand([self.inputSize]))
            self.V_rest = PyroSample(dist.Normal(15, 1.).expand([self.inputSize]))
            self.V_slope = PyroSample(dist.Normal(-76, 1.).expand([self.inputSize]))
        else:
            self.g_max = g_max
            self.V_rest = V_rest
            self.V_slope = V_slope
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(300)
        preVoltage = preVoltage[self.synapseInput]
        postVoltage = postVoltage[self.synapseOutput]
        current = self.g_max * 1 / (1+torch.exp((preVoltage-self.V_rest) / self.V_slope)) * (postVoltage)
        current = current * self.synapseWeight
        output[self.synapseOutput] = current
        return output

# Using GRU for Synapse Simulation for Synapse Plasticity
class RecurrentSynapse(PyroModule):
    hidden_size = 8
    num_layers = 2
    def __init__(self, synapseInput, synapseOutput, synapseWeight):
        super().__init__()
        self.RNN = []
        for i in range (len(synapseInput)):
            self.RNN.append(PyroModule[nn.Sequential](PyroModule[nn.GRU](2, self.hidden_size, num_layers=self.num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](self.hidden_size, 1)))
        self.synapseInput=synapseInput
        self.synapseOutput=synapseOutput
        self.synapseWeight=synapseWeight
    def forward(self, postVoltage, preVoltage):
        output = torch.zeros(300)
        for i in range(self.synapseInput):
            output[self.synapseOutput[i]] += self.RNN[i](( preVoltage[i], postVoltage[i])) * self.synapseWeight[i]
        return output

class synapseLayer(PyroModule):
    def __init__(self, synapseList):
        super().__init__()
        self._synpase_list = synapseList
    
    def forward(self, inputSignal):
        for layer in self._synpase_list:
            output = output + layer(inputSignal)
        return output

class NematodeForStep(PyroModule):
    # Work for Single-Step Inference for MCMC/Sequential MC and Bayesian Filter.
#    def __init__ (self, InputSize, SensoryNeuronList, ConnectomeSize, OutputSize):
    def __init__ (self, NeuronList, synapseList):
        super().__init__()
        
        # Neuron
        self.Neuron = neuronLayer(len(NeuronList), NeuronList)
        self.NeuronSize = len(NeuronList)
        self.synapse = synapseLayer(synapseList)
        self.synapseSize = len(synapseList)
        # From Interneuron to Motor Neuron
        # self.InterneuronToMotor = synapseLayer(ConnectomeSize, OutputSize) 
        # Motor neuron
        # self.MotorNeuron = neuronLayer(OutputSize)
        # Motor Output
        # self.OutputLayer = nn.Linear(OutputSize)
        
    def forward(self, Prev, ExternalInput=None, VoltageClamp=None): 
        CurrentInput = self.synapse(Prev)
        if ExternalInput is None:
            ExternalInput = torch.zeros(self.NeuronSize)
        ConnectomeOutput = self.Neuron(CurrentInput, ExternalInput)
        for Label in range(len(VoltageClamp)):
            if VoltageClamp[Label] != 0:  
            # Force Voltage Clamp First
                ConnectomeOutput[Label] = VoltageClamp[Label]
        # Deal with Sensory Neuron with non-current input
        return ConnectomeOutput

    def step(self, state, mask, y=None):
        self.t += 1
        state["z"] = pyro.sample(
            "z_{}".format(self.t),
            dist.Normal(self.forward(state["z"]), self.SNR).to_event(300),
        )
        y = pyro.sample(
            "y_{}".format(self.t), dist.Normal(state["z"][mask], self.sigma), obs=y
        )
        return state["z"], y

class RecurrentNematode(PyroModule):
    def __init__(self, NeuronList, synapseList, batch_sizes=None):
        super().__init__()
        self.model = NematodeForStep(NeuronList, synapseList)
        self.expected_input_dim = 2 if batch_sizes is not None else 3

    def forward(self, input, ExternalInput=None, VoltageClamp=None):
        if torch.Size(input) != self.expected_input_dim :
            Exception("Shape Unmatched")
        ConnectomeOutput = torch.zeros((torch.Size(input)))
        if self.expected_input_dim == 2:
            for i in range(torch.Size(input)[0]): # Iterate in Time 
                if ExternalInput:
                    _ExternalInput = ExternalInput[i]
                else:
                    _ExternalInput = None
                if VoltageClamp:
                    _VoltageClamp = VoltageClamp[i]
                else:
                    _VoltageClamp = None

                ConnectomeOutput[i] = self.model(ConnectomeOutput[i-1], _ExternalInput, _VoltageClamp)
        else :
            for i in range(torch.Size(input)[0]): # Iterate in Batch
                for time in range(torch.Size(input)[1]):
                    if ExternalInput:
                        _ExternalInput = ExternalInput[i][time]
                    else:
                        _ExternalInput = None
                    if VoltageClamp:
                        _VoltageClamp = VoltageClamp[i][time]
                    else:
                        _VoltageClamp = None
                    ConnectomeOutput[i][time] = self.model(ConnectomeOutput[i][time-1], _ExternalInput, _VoltageClamp)
        return ConnectomeOutput

SensoryList = {"ASHL": ASH, "ASHR": ASH}

def readConnectome(path):
    Sensory = pandas.read_excel(path, sheet_name="Sensory")
    # Connectome = pandas.read_excel(path, sheet_name="Connectome")
    # Using Cook et al. (2019)
    Connectome = pandas.read_csv("./data/herm_full_edgelist.csv")
    NeuronList = []
    synapseList = []
    SensoryMask = []

    # Neuron Name, 300 neurons
    NeuronName=numpy.loadtxt("./pumpprobe/pumpprobe/aconnectome_ids.txt", dtype=object)[:, 1]
    # Which neuron is sensory neuron.
    Sensory = Sensory["Neuron"]
    for i in range(len(NeuronName)):
        if NeuronName[i] in Sensory:
            if NeuronName[i] in SensoryList:
                NeuronList.append(SensoryList[NeuronName[i]]())
            else: 
                NeuronList.append(FallbackSensory())
            SensoryMask.append(1)
        else:
#            if NeuronName[i][0:3] == "RMD": TODO: Build a spiking model, need case-by-case approach.
#                NeuronList.append(SpikingModel())
            NeuronList.append(conductanceModel())
            SensoryMask.append(0)
    SensoryMask = torch.Tensor(SensoryMask)

    Gap_Junction_SRC = []
    Gap_Junction_DST = []
    Wicks_SRC = []
    Wicks_DST = []
    Generic_SRC = []
    Generic_DST = []
    Gap_Junction_Weight = []
    Wicks_Weight = []
    Generic_Weight = []
    for _,i in Connectome.iterrows():
        Origin = i["Source"]
        Target = i["Target"]
        Type = i["Type"]
        Weight = i["Weight"]

        try:
            src = numpy.where(NeuronName == Origin)
            dst = numpy.where(NeuronName == Target)
        except:
            continue # Missing

        if Type == "electrical":
            Gap_Junction_SRC.append(src)
            Gap_Junction_DST.append(dst)
            Gap_Junction_Weight.append(Weight)
        elif Type == "chemical":
            Wicks_SRC.append(src)
            Wicks_DST.append(dst)
            Wicks_Weight.append(Weight)
    
    # Merge Neuropetitide and Monoamine here. 
    neuropetitide = pandas.read_csv("./data/edgelist_NP.csv", header=None)[[0,1]]
    monoamine = pandas.read_csv("./data/edgelist_MA.csv", header=None)[[0,1]]
    extrasynaptic = pandas.concat([neuropetitide, monoamine], ignore_index=True).drop_duplicates()

    for i in extrasynaptic.iterrows():
        # Serotonin, Dopamine, Octopamine, Tyramine is likely to be extrasynaptic. (>95%)  
        try:
            src = numpy.where(NeuronName == Origin)
            dst = numpy.where(NeuronName == Target)
        except:
            continue # Missing  
        Generic_SRC.append(src)
        Generic_DST.append(dst)
        Generic_Weight.append(1) 

    synapseList.append(GeneralSynapse(Gap_Junction_SRC, Gap_Junction_DST, Gap_Junction_Weight))
    synapseList.append(WicksSynapse(Wicks_SRC, Wicks_DST, Wicks_Weight))
#    synapseList.append(RecurrentSynapse(Generic_SRC, Generic_DST, Generic_Weight))
    model = NematodeForStep( NeuronList, synapseList)
    return model
