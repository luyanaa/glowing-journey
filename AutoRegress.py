import numpy as np
import torch
import torch.nn as nn
from pyro.nn import PyroModule


class AutoRegression(PyroModule):
    """Autoregressive GRU baseline over connectome edges and neurons."""

    def __init__(self, neuron_dim: int, synapse_dim: int, synapseInput, synapseOutput, synapseWeight):
        super().__init__()
        self.synapse_dim = int(synapse_dim)
        self.neuron_dim = int(neuron_dim)
        self.synapse = nn.ModuleList(
            [nn.GRU(input_size=2, hidden_size=16, num_layers=2, batch_first=True) for _ in range(self.synapse_dim)]
        )
        self.synapse_head = nn.ModuleList([nn.Linear(16, 1) for _ in range(self.synapse_dim)])
        self.neuron = nn.ModuleList(
            [nn.GRU(input_size=2, hidden_size=16, num_layers=2, batch_first=True) for _ in range(self.neuron_dim)]
        )
        self.neuron_head = nn.ModuleList([nn.Linear(16, 1) for _ in range(self.neuron_dim)])

        self.register_buffer("synapseInput", torch.as_tensor(np.asarray(synapseInput), dtype=torch.long))
        self.register_buffer("synapseOutput", torch.as_tensor(np.asarray(synapseOutput), dtype=torch.long))
        self.register_buffer("synapseWeight", torch.as_tensor(np.asarray(synapseWeight), dtype=torch.float32))

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        x_b = x.unsqueeze(0) if squeeze else x
        syn_cur = torch.zeros_like(x_b)

        for i in range(self.synapse_dim):
            pair = torch.stack(
                [x_b[:, self.synapseInput[i]], x_b[:, self.synapseOutput[i]]], dim=-1
            ).unsqueeze(1)
            out, _ = self.synapse[i](pair)
            cur = self.synapse_head[i](out[:, -1, :]).squeeze(-1) * self.synapseWeight[i]
            syn_cur[:, self.synapseOutput[i]] = syn_cur[:, self.synapseOutput[i]] + cur

        next_x = []
        for j in range(self.neuron_dim):
            inp = torch.stack([x_b[:, j], syn_cur[:, j]], dim=-1).unsqueeze(1)
            out, _ = self.neuron[j](inp)
            next_x.append(self.neuron_head[j](out[:, -1, :]).squeeze(-1))

        res = torch.stack(next_x, dim=-1)
        return res.squeeze(0) if squeeze else res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects (B, T, N) or (T, N)
        squeeze = x.dim() == 2
        x_b = x.unsqueeze(0) if squeeze else x
        outputs = [torch.zeros_like(x_b[:, 0, :])]
        for t in range(1, x_b.shape[1]):
            outputs.append(self.forward_step(x_b[:, t - 1, :]))
        res = torch.stack(outputs, dim=1)
        return res.squeeze(0) if squeeze else res
