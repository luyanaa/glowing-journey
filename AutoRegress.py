
class AutoRegression(nn.Module):
    def __init__(self, neuron_dim, synapse_dim, synapseInput, synapseOutput, synapseWeight):
        super().__init__()
        self.synapse_dim = synapse_dim
        self.neuron_dim = neuron_dim
        self.synapse = [PyroModule[GRU](input_size=1, hidden_size=16, num_layers=3) for i in synapse_dim]
        self.synapse = ModuleList(self.synapse)
        self.neuron = [PyroModule[GRU](input_size=1, hidden_size=16, num_layers=3) for i in neuron_dim]        
        self.neuron = ModuleList(self.neuron)
        self.synapseInput = synapseInput
        self.synapseOutput = synapseOutput
        self.syanpseWeight = synapseWeight

    def forward_step(self, x):
        output = torch.zeros_like(x)
        if x.dim() == 1: 
            for i in range(self.synapseInput):
                output[self.synapseOutput[i]] += self.synapse[i]((x[self.synapseInput[i]], x[self.synapseOutput[i]])) * self.synapseWeight[i]
        else:
            for i in range(self.synapseInput):
                output[:, self.synapseOutput[i]] += self.synapse[i]((x[:, self.synapseInput[i]], x[:, self.synapseOutput[i]])) * self.synapseWeight[i]
        
        output = torch.stack([self.neuron[_i](x) for _i in self.neuron_dim])
        return output
    
    def forward(self, x):
        output = torch.zeros_like(x)
        if x.dim() == 2:
            for time in range(1, x.shape[1]):
                output[time]=self.forward_step(x[time-1])
        elif x.dim() == 3:
            for time in range(x.shape[2]):
                output[:, :, time] = self.forward_step(x[:,:, time-1])
        return output
