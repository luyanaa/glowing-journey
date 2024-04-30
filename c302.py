import torch
import pyro
import torch.nn as nn
import pandas
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.distributions import Normal
import numpy

timeStep = 0.05
value_scale = 200


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
        self.value = self.value + self.model(self.value, inputSignal) * timeStep * value_scale
        return self.value

class FallbackSensory(PyroModule):
    def __init__(self, hidden_size=8, num_layers=2):
        super().__init__()
        self.model=(PyroModule[nn.Sequential](PyroModule[nn.GRU](3, hidden_size, num_layers=num_layers, bidirectional=False), \
                                          PyroModule[nn.Linear](hidden_size, 1)))
    def forward(self, inputSignal, ExternalInput):
        self.value = self.value + self.model(self.value, inputSignal, ExternalInput) * timeStep * value_scale
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
    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_syn=None):
        super().__init__()
        self.synapseInput = torch.Tensor(numpy.array(synapseInput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseOutput = torch.Tensor(numpy.array(synapseOutput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseWeight = torch.Tensor(numpy.array(synapseWeight)).squeeze()
        if g_syn is None:
            self.g_syn = PyroSample(dist.Normal(0., 1.).expand([len(synapseInput)]).to_event(1))
        else:
            self.g_syn = g_syn
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
            # Assign
            # In the following, we set Vslope = 15 mV, gsyn = 0.6 nS and Vrest = −76 mV (Wicks et al., 1996).
        self.inputSize = len(synapseInput)
        self.synapseInput = torch.Tensor(numpy.array(synapseInput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseOutput = torch.Tensor(numpy.array(synapseOutput, dtype=numpy.int64)).to(torch.int64).squeeze()
        self.synapseWeight = torch.Tensor(numpy.array(synapseWeight)).squeeze()
        if g_max is None:
            self.g_max = PyroSample(dist.Normal(0.6, 1.).expand([self.inputSize]).to_event(1))
        else: 
            self.g_max = g_max
        if V_rest is None:
            self.V_rest = PyroSample(dist.Normal(0.015, 0.01).expand([self.inputSize]).to_event(1))
        else:
            self.V_rest = V_rest
        if V_slope is None:
            self.V_slope = PyroSample(dist.Normal(-0.076, 0.1).expand([self.inputSize]).to_event(1))
        else:
            self.V_slope = V_slope
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

# https://journals.physiology.org/doi/full/10.1152/jn.01176.2003
# SNR about 1000-10000 in crab in 5-7mm, C. elegans in less than 1mm. 
# Assuming Distance * SNR = Constant, 5000-50000, selecting 27500. 
# SNR = mu^2 / sigma^2 , let mu = 1, approximately sigma = 0.006
class synapseLayer(PyroModule):
    def __init__(self, synapseList):
        super().__init__()
        self.Wicks_SRC, self.Wicks_DST, self.Wicks_Weight = synapseList["Wicks"]
        self.Gap_Junction_SRC, self.Gap_Junction_DST, self.Gap_Junction_Weight = synapseList["General"]
        Generic_SRC, Generic_DST, Generic_Weight = synapseList["Generic"] 

        # self.generic = RecurrentSynapse(Generic_SRC, Generic_DST, Generic_Weight)
        self.wicks = WicksSynapse(self.Wicks_SRC, self.Wicks_DST, self.Wicks_Weight)    
        self.general = GeneralSynapse(self.Gap_Junction_SRC, self.Gap_Junction_DST, self.Gap_Junction_Weight)
    def forward(self, inputSignal):
        output = self.wicks(inputSignal)
        output = output + self.general(inputSignal)
        # output = output + self.generic(inputSignal)
        return output

# Making the single-step model as Deep-Markov as possible. 
from math import sqrt
class NematodeForStep(PyroModule):
    def __init__ (self, neuronSize, NeuronList, synapseList):
        super().__init__()
        self.Neuron = neuronLayer(neuronSize, NeuronList)
        self.synapse = synapseLayer(synapseList)
    def forward(self, Prev, ExternalInput=None, VoltageClamp=None): 
        CurrentInput = self.synapse(Prev)
        if ExternalInput is None:
            ExternalInput = torch.zeros_like(Prev)
        ConnectomeOutput = self.Neuron(Prev, CurrentInput, ExternalInput)
        ConnectomeOutput[VoltageClamp != 0.0] = VoltageClamp[VoltageClamp != 0.0]
        return ConnectomeOutput

class RecurrentNematode(PyroModule):
    def __init__(self, model, scale = 2):
        super().__init__()
        self.model = model
        self.scale = scale
        global timeStep
        timeStep = timeStep / scale
        self.sigma = pyro.sample("sigma" , dist.Uniform(70.71, 223.61).expand([300]).to_event(1))


    def forward(self, VoltageClamp=None, ExternalInput=None, mask=None, y=None):
        scale = self.scale
        if VoltageClamp is None and ExternalInput is not None: 
            VoltageClamp = torch.zeros_like(ExternalInput)
        elif ExternalInput is None and VoltageClamp is not None:
            ExternalInput = torch.zeros_like(VoltageClamp)
        elif VoltageClamp is None and ExternalInput is None and y is not None: 
            if y.dim == 2:
                VoltageClamp = torch.zeros((y.shape[0], 300))
                ExternalInput = VoltageClamp
            else: 
                VoltageClamp = torch.zeros((y.shape[0], y.shape[1], 300))
                ExternalInput = VoltageClamp
        if VoltageClamp is not None and ExternalInput is not None and VoltageClamp.ndim != ExternalInput.ndim :
            Exception("Shape Unmatched")

        # Doing interpolation in scale. 
        if VoltageClamp.ndim == 2:
            ConnectomeOutput = torch.zeros((VoltageClamp.shape[0] * scale, VoltageClamp.shape[1]))
        if VoltageClamp.ndim == 3:
            ConnectomeOutput = torch.zeros((VoltageClamp.shape[0] , VoltageClamp.shape[1] * scale, VoltageClamp.shape[2]))
        if torch.cuda.is_available():
            ConnectomeOutput = ConnectomeOutput.cuda()
        if ExternalInput.ndim == 2:
            for time in range(VoltageClamp.size()[0] *scale ): # Iterate in Time 
                if y is None or time % scale != 0: # In inference mode or with interpolation
                    ConnectomeOutput[time] = self.model(ConnectomeOutput[time-1], ExternalInput[(time // scale)], VoltageClamp[(time // scale)])
                else: 
                    ConnectomeOutput[time] = self.model(ConnectomeOutput[time-1], ExternalInput[(time // scale)], VoltageClamp[(time // scale)])
                    ConnectomeOutput[time] = pyro.sample("z_%d" % time, dist.Normal(ConnectomeOutput[time],  1 / (self.sigma * self.sigma)).to_event(1), obs= y[time // scale], obs_mask=mask)  

        else :
            for time in range(VoltageClamp.size()[1] * scale) :
                if y is None or time % scale != 0: # In inference mode or with interpolation
                    ConnectomeOutput[:, time, :] = self.model(ConnectomeOutput[:, time-1, :], ExternalInput[:, time // scale, :], VoltageClamp[:, time // scale, :])       
                else : 
                    ConnectomeOutput[:, time, :] = self.model(ConnectomeOutput[:, time-1, :], ExternalInput[:, time // scale, :], VoltageClamp[:, time // scale, :])     
                    ConnectomeOutput[:, time, :] = pyro.sample("z_%d" % time,dist.Normal(ConnectomeOutput[:, time-1, :],  1 / (self.sigma * self.sigma)).to_event(2), obs= y[:, time // scale, :], obs_mask=mask)  
        return ConnectomeOutput
        
    def assign(self, neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale):  
        self.model.Neuron.conductance.E = PyroSample(dist.Normal(neuron_loc[0], neuron_scale[0]).to_event(1))
        self.model.Neuron.conductance.G = PyroSample(dist.Normal(neuron_loc[1], neuron_scale[1]).to_event(1))
        self.model.Neuron.conductance.C = PyroSample(dist.Normal(neuron_loc[2], neuron_scale[2]).to_event(1))

        self.model.synapse.wicks = WicksSynapse(self.model.synapse.Wicks_SRC, self.model.synapse.Wicks_DST, self.model.synapse.Wicks_Weight, \
                                                    g_max = PyroSample(dist.Normal(wicks_loc[0], wicks_scale[0]).to_event(1)),
                                                    V_rest= PyroSample(dist.Normal(wicks_loc[1], wicks_scale[1]).to_event(1)), 
                                                    V_slope= PyroSample(dist.Normal(wicks_loc[2], wicks_scale[2]).to_event(1)))
        self.model.synapse.general = GeneralSynapse(self.model.synapse.Gap_Junction_SRC, self.model.synapse.Gap_Junction_DST, self.model.synapse.Gap_Junction_Weight, \
                                                        g_syn = PyroSample(dist.Normal(gap_loc, gap_scale).to_event(1)))    

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
