import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal
from utils import timeStep, scale
import numpy
import snntorch 

# For Neuron: Input Current, Output Voltage (Easy to log, easy to calculate Vpre-Vpost)
# For Synapse: Input Voltage, Output Current. 

# Neuron Type: Conductance Model (Done), Sensory, Motor (Need case-by-case solution)
# Synapse Type: Gap Junction, Chemical Synapse (A variety of), extrasynaptic 

class GRUModel(PyroModule):
    def __init__(self, hidden_size=8, num_layers=2):
        super().__init__()
        self.value = PyroSample(Normal(0.0, 1.0))
        self.model=(PyroModule[nn.Sequential](PyroModule[nn.GRU](2, hidden_size, num_layers=num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](hidden_size, 1)))
    def forward(self, inputSignal):
        self.value = self.value + self.model(self.value, inputSignal) * timeStep * scale
        return self.value

class FallbackSensory(PyroModule):
    def __init__(self, hidden_size=8, num_layers=2):
        super().__init__()
        self.model=(PyroModule[nn.Sequential](PyroModule[nn.GRU](3, hidden_size, num_layers=num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](hidden_size, 1)))
    def forward(self, inputSignal, ExternalInput):
        self.value = self.value + self.model(self.value, inputSignal, ExternalInput) * timeStep * scale
        return self.value
class SpikingModel(PyroModule):
    def __init__(self, ):
        pass
    def forward():
        pass

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
        if torch.cuda.is_available():
            self.k1 = self.k1.cuda()
            self.k2 = self.k2.cuda()
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
            
class conductanceLayer (PyroModule):
    def __init__(self, input_size):
        super().__init__()
        self.E = PyroSample(dist.Normal(0., 1.).expand([input_size]).to_event(1))
        self.G = PyroSample(dist.Normal(1., 1.).expand([input_size]).to_event(1))
        self.C = PyroSample(dist.Normal(0.05, .025).expand([input_size]).to_event(1))
        
        if torch.cuda.is_available():
                self.E = self.E.cuda()
                self.G = self.G.cuda()
                self.C = self.C.cuda()
    # Input Current, Output Voltage
    def forward(self, Prev, inputSignal):
        current = self.G*(Prev - self.E) 
        dv = (inputSignal - current) 
        dv = self.C
        Output = Prev + dv * timeStep
        return Output

class neuronLayer(PyroModule):
    def __init__(self, neuronSize, neuronList):
        super().__init__()
        self._neuron_List = neuronList
        self.neuronSize = neuronSize
        self.conductance = conductanceLayer(input_size=neuronSize)
    def forward(self, Prev, inputSignal, externalInput):
        output = self.conductance(Prev, inputSignal)
        if inputSignal.dim() == 1:
            for i in self._neuron_List:
                if inputSignal[i]!=0.0:
                    output[i]=self._neuron_List[i](inputSignal[i], externalInput[i])
                else:
                    output[i]=self._neuron_List[i](inputSignal[i])
        else: 
            for i in self._neuron_List:
                if inputSignal[i]!=0.0:
                    output[:, i]=self._neuron_List[:, i](inputSignal[:, i], externalInput[:, i])
                else:
                    output[:, i]=self._neuron_List[:, i](inputSignal[:, i])
        return output

# A Rough Estimation for Gap Junction and Synaptic ones. 
# Refer to https://www.mdpi.com/2227-7390/11/11/2442

# Extrasynaptic Connectome Refer to https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005283#sec002
# Monoamine
# Serotonin: https://cell.com/cell/pdf/S0092-8674(23)00419-1.pdf , looks like a current input
# Dealt as a basic conductance model
class GeneralSynapse(PyroModule):
    def __init__(self, synapseInput, synapseOutput, synapseWeight):
        super().__init__()
        self.synapseInput = torch.Tensor(numpy.array(synapseInput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseOutput = torch.Tensor(numpy.array(synapseOutput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseWeight = torch.Tensor(numpy.array(synapseWeight)).squeeze()
        self.g_syn = PyroSample(dist.Normal(0., 1.).expand([len(synapseInput)]).to_event(1))
        if torch.cuda.is_available():
            self.synapseInput = self.synapseInput.cuda()
            self.synapseOutput = self.synapseOutput.cuda()
            self.synapseWeight = self.synapseWeight.cuda()
            self.g_syn = self.g_syn.cuda()
    def forward(self, inputSignal):
        output = torch.zeros_like(inputSignal)
        if inputSignal.dim() == 1: # Unbatched
            preVoltage = torch.index_select(inputSignal, 0, self.synapseInput)
            postVoltage = torch.index_select(inputSignal, 0, self.synapseOutput)
        elif inputSignal.dim() == 2: # Batched, selected on neuron
            preVoltage = torch.index_select(inputSignal, 1, self.synapseInput)
            postVoltage = torch.index_select(inputSignal, 1, self.synapseOutput)
        current = self.g_syn*(postVoltage-preVoltage)
        current = current * self.synapseWeight
        if inputSignal.dim() == 1: # Unbatched
            output[self.synapseOutput] = current
        elif inputSignal.dim() == 2:
            output[:, self.synapseOutput] = current
        return output

# Refer to https://hal.science/hal-03705452/file/new_simple_model_pg.pdf
# Wicks et al. (1999)
class WicksSynapse(PyroModule):
    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_max=None, V_rest=None, V_slope=None):
        super().__init__()
        if V_rest is None or V_slope is None:
            # Assign
            # In the following, we set Vslope = 15 mV, gsyn = 0.6 nS and Vrest = −76 mV (Wicks et al., 1996).
            self.inputSize = len(synapseInput)
            self.synapseInput = torch.Tensor(numpy.array(synapseInput, dtype=numpy.int64)).to(torch.int64).squeeze()
            self.synapseOutput = torch.Tensor(numpy.array(synapseOutput, dtype=numpy.int64)).to(torch.int64).squeeze()
            self.synapseWeight = torch.Tensor(numpy.array(synapseWeight)).squeeze()
            self.g_max = PyroSample(dist.Normal(0.6, 1.).expand([self.inputSize]).to_event(1))
            self.V_rest = PyroSample(dist.Normal(0.015, 0.01).expand([self.inputSize]).to_event(1))
            self.V_slope = PyroSample(dist.Normal(-0.076, 0.1).expand([self.inputSize]).to_event(1))
            if torch.cuda.is_available():
                self.synapseInput = self.synapseInput.cuda()
                self.synapseOutput = self.synapseOutput.cuda()
                self.synapseWeight = self.synapseWeight.cuda()
                self.g_max = self.g_max.cuda()
                self.V_rest = self.V_rest.cuda()
                self.V_slope = self.V_slope.cuda()
    def forward(self, inputSignal):
        output = torch.zeros_like(inputSignal)
        if inputSignal.dim() == 1: # Unbatched
            preVoltage = torch.index_select(inputSignal, 0, self.synapseInput)
            postVoltage = torch.index_select(inputSignal, 0, self.synapseOutput)
        elif inputSignal.dim() == 2: # Batched, selected on neuron
            preVoltage = torch.index_select(inputSignal, 1, self.synapseInput)
            postVoltage = torch.index_select(inputSignal, 1, self.synapseOutput)
        current = self.g_max * 1 / (1+torch.exp((preVoltage-self.V_rest) / self.V_slope)) 
        current = current * self.synapseWeight * (postVoltage)
        if inputSignal.dim() == 1: # Unbatched
            output[self.synapseOutput] = current
        elif inputSignal.dim() == 2:
            output[:, self.synapseOutput] = current
        return output

# Using GRU for Synapse Simulation for Synapse Plasticity
# Do not need timeStep as current is mainly calculated with pre/post voltage 
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
    def forward(self, inputSignal):
        output = torch.zeros(inputSignal)
        if inputSignal.dim() == 1:
            for i in range(self.synapseInput):
                output[self.synapseOutput[i]] += self.RNN[i]((inputSignal[self.synapseInput[i]], inputSignal[self.synapseOutput[i]])) * self.synapseWeight[i]
        else: 
            for i in range(self.synapseInput):
                output[:, self.synapseOutput[i]] += self.RNN[i]((inputSignal[:, self.synapseInput[i]], inputSignal[:, self.synapseOutput[i]])) * self.synapseWeight[i]
        return output

class synapseLayer(PyroModule):
    def __init__(self, synapseList):
        super().__init__()
        Wicks_SRC, Wicks_DST, Wicks_Weight = synapseList["Wicks"]
        Gap_Junction_SRC, Gap_Junction_DST, Gap_Junction_Weight = synapseList["General"]
        Generic_SRC, Generic_DST, Generic_Weight = synapseList["Generic"] 

        # self.generic = RecurrentSynapse(Generic_SRC, Generic_DST, Generic_Weight)
        self.wicks = WicksSynapse(Wicks_SRC, Wicks_DST, Wicks_Weight)    
        self.general = GeneralSynapse(Gap_Junction_SRC, Gap_Junction_DST, Gap_Junction_Weight)
    def forward(self, inputSignal):
        output = self.wicks(inputSignal)
        output = output + self.general(inputSignal)
        # output = output + self.generic(inputSignal)
        return output

# https://journals.physiology.org/doi/full/10.1152/jn.01176.2003
# SNR about 1000-10000 in crab in 5-7mm, C. elegans in less than 1mm. 
# Assuming Distance * SNR = Constant, 5000-50000, selecting 27500. 
# SNR = mu^2 / sigma^2 , let mu = 1, approximately sigma = 0.006

class NematodeForStep(PyroModule):
    def __init__ (self, neuronSize, NeuronList, synapseList):
        super().__init__()
        self.Neuron = neuronLayer(neuronSize, NeuronList)
        self.NeuronSize = len(NeuronList)
        self.synapse = synapseLayer(synapseList)
        self.synapseSize = len(synapseList)
    
    def init(self, state, initial):
        state["z"] = pyro.sample("z_init", dist.Delta(initial, event_dim=1))

    def forward(self, time, Prev, ExternalInput=None, VoltageClamp=None, y=None, mask=None): 
        CurrentInput = self.synapse(Prev)
        if ExternalInput is None:
            ExternalInput = torch.zeros_like(Prev)
        ConnectomeOutput = self.Neuron(Prev, CurrentInput, ExternalInput)
        ConnectomeOutput[VoltageClamp != 0.0] = VoltageClamp[VoltageClamp != 0.0]
        # Add noise to single inference.  
        self.t = time
        sigma = pyro.sample("sigma_%d" % self.t, dist.Uniform(70.71, 223.61))
        if mask is None:  
            ConnectomeOutput = pyro.sample("z_%d" % self.t, dist.Normal(ConnectomeOutput,  1 / (sigma * sigma)).to_event(2), obs=y)  
        if mask is not None:
            ConnectomeOutput = pyro.sample("z_%d" % self.t, dist.Normal(ConnectomeOutput,  1 / (sigma * sigma)).to_event(2))
            obs = pyro.sample("obs_%d" % self.t, ConnectomeOutput[mask], obs=y)
        return ConnectomeOutput

# TODO: Small Step for Simulation, Big Step for Sampling.
class RecurrentNematode(PyroModule):
    def __init__(self, model, batch_sizes=1):
        super().__init__()
        self.model = model
#        self.expected_input_dim = 2 if batch_sizes is None else 3

    def forward(self, VoltageClamp, ExternalInput, y=None):
        if VoltageClamp is None and ExternalInput is not None: 
            VoltageClamp = torch.zeros_like(ExternalInput)
        elif ExternalInput is None and VoltageClamp is not None:
            ExternalInput = torch.zeros_like(VoltageClamp)
        else: 
            Exception("Input dimension cannot be infered from two NoneType. ")
        if VoltageClamp.ndim != ExternalInput.ndim :
            Exception("Shape Unmatched")
        ConnectomeOutput = torch.zeros_like(VoltageClamp)
        if torch.cuda.is_available():
            ConnectomeOutput = ConnectomeOutput.cuda()
        if ExternalInput.ndim == 2:
            for i in range(VoltageClamp.size()[0]): # Iterate in Time 
                if y is None:
                    ConnectomeOutput[i] = self.model(i. ConnectomeOutput[i-1], ExternalInput[i], VoltageClamp[i])
                else:
                    ConnectomeOutput[i] = self.model(i, ConnectomeOutput[i-1], ExternalInput[i], VoltageClamp[i], y[i])
        else :
            for time in range(VoltageClamp.size()[1]):
                if y is None:  
                    ConnectomeOutput[:, time, :] = self.model(time, ConnectomeOutput[:, time-1, :], ExternalInput[:, time, :], VoltageClamp[:, time, :])       
                else : 
                    ConnectomeOutput[:, time, :] = self.model(time, ConnectomeOutput[:, time-1, :], ExternalInput[:, time, :], VoltageClamp[:, time, :], y[:, time, :])       
        return ConnectomeOutput

SensoryList = {"ASHL": ASH, "ASHR": ASH}

def readConnectome(path):
    Sensory = pandas.read_excel(path, sheet_name="Sensory")
    # Using Cook et al. (2019)
    Connectome = pandas.read_csv("./data/herm_full_edgelist.csv")
    NeuronList = {}
    synapseList = []
    SensoryMask = []

    # Neuron Name, 300 neurons
    NeuronName=numpy.loadtxt("./pumpprobe/pumpprobe/aconnectome_ids.txt", dtype=object)[:, 1]
    # Which neuron is sensory neuron.
    Sensory = Sensory["Neuron"]
    for i in range(len(NeuronName)):
        if NeuronName[i] in Sensory:
            if NeuronName[i] in SensoryList:
                NeuronList[i]=SensoryList[NeuronName[i]]()
            else: 
                NeuronList[i]=FallbackSensory()
            SensoryMask.append(1)
        else:
#            if NeuronName[i][0:3] == "RMD": TODO: Build a spiking model, need case-by-case approach.
#                NeuronList.append(SpikingModel())
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
        Origin = str(i["Source"]).rstrip()
        Target = str(i["Target"]).lstrip().rstrip()
        # Dealing with VC1 versus VC01
        if len(Origin) == 4 and Origin[2] == "0":
            Origin = Origin[0] + Origin[1] + Origin[3]
        if len(Target) == 4 and Target[2] == "0":
            Target = Target[0] + Target[1] + Target[3]
        Type = i["Type"]
        Weight = i["Weight"]
        dst = numpy.where(NeuronName == Target)
        src = numpy.where(NeuronName == Origin)
        src = src[0]
        dst = dst[0]

        if src.size ==0 or dst.size ==0:
           continue # Removing missing neuron.

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
    
    print("Gap Junction: ", len(Gap_Junction_SRC))
    print("Chemical Synapse: ", len(Wicks_SRC))
    print("Extrasynaptic Connection: ", len(Generic_SRC))

    synapseList = {}
    synapseList["General"] = (Gap_Junction_SRC, Gap_Junction_DST, Gap_Junction_Weight)
    synapseList["Wicks"] = (Wicks_SRC, Wicks_DST, Wicks_Weight)
    synapseList["Generic"] = (Generic_SRC, Generic_DST, Generic_Weight)
    model = NematodeForStep(NeuronName.size, NeuronList, synapseList)
    return model
