import math
from typing import Optional

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroParam
import torch
import torch.nn as nn
import torch.nn.functional as F

from c302 import RecurrentNematode
from tcn import TemporalConvNet


class TCN(PyroModule):
    """Temporal Convolutional Network encoder pooling temporal features per sample."""

    n_channels = [25] * 4
    kernel_size = 3
    dropout = 0.25

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels=None,
        kernel_size: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if dropout is not None:
            self.dropout = dropout
        if kernel_size is not None:
            self.kernel_size = kernel_size
        if num_channels is not None:
            self.n_channels = num_channels
        self.tcn = TemporalConvNet(
            input_size, self.n_channels, self.kernel_size, dropout=self.dropout
        )
        self.linear = nn.Linear(self.n_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # x: (B, T, C) -> TCN expects (B, C, T)
        features = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pooled = features.mean(dim=1)  # (B, hidden)
        return self.linear(pooled)


class Encoder(PyroModule):
    """Amortized TCN encoder producing variational loc/scale for connectome parameters."""

    def __init__(self, neuron_dim: int, wicks_dim: int, gap_dim: int):
        super().__init__()
        self.neuron_dim = int(neuron_dim)
        self.wicks_dim = int(wicks_dim)
        self.gap_dim = int(gap_dim)

        # Amortized posterior EMA for open-loop evaluation when y is withheld (y=None)
        self.register_buffer("_has_amortized_post", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_ema_neuron_loc", torch.zeros(3, self.neuron_dim))
        self.register_buffer("_ema_neuron_scale", torch.ones(3, self.neuron_dim))
        self.register_buffer("_ema_wicks_loc", torch.zeros(4, self.wicks_dim))
        self.register_buffer("_ema_wicks_scale", torch.ones(4, self.wicks_dim))
        self.register_buffer("_ema_gap_loc", torch.zeros(self.gap_dim))
        self.register_buffer("_ema_gap_scale", torch.ones(self.gap_dim))

        # Neuron params: E (Normal), log G, log C (LogNormal) -> 3 loc + 3 scale
        self.neuron = TCN(input_size=self.neuron_dim, output_size=self.neuron_dim * 6)
        # Chemical synapse params: log g_max, V_rest, log V_slope, E_rev -> 4 loc + 4 scale
        self.wick = TCN(input_size=self.neuron_dim, output_size=max(1, self.wicks_dim) * 8)
        # Gap junction params: log g_syn -> 1 loc + 1 scale
        self.gap = TCN(input_size=self.neuron_dim, output_size=max(1, self.gap_dim) * 2)

        # Center initial outputs near biophysical prior means
        with torch.no_grad():
            self.neuron.linear.weight.mul_(0.05)
            self.neuron.linear.bias.zero_()
            self.wick.linear.weight.mul_(0.05)
            self.wick.linear.bias.zero_()
            self.gap.linear.weight.mul_(0.05)
            self.gap.linear.bias.zero_()

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Pool across batch so global connectome variational parameters match (D,) shapes
        neuron_raw = self.neuron(x).mean(dim=0).view(6, self.neuron_dim)
        neuron_loc = torch.stack(
            [
                -0.065 + 0.02 * torch.tanh(neuron_raw[0]),
                math.log(0.5) + 0.5 * torch.tanh(neuron_raw[1]),
                math.log(0.05) + 0.5 * torch.tanh(neuron_raw[2]),
            ],
            dim=0,
        )
        neuron_scale = torch.stack(
            [
                0.002 + 0.02 * torch.sigmoid(neuron_raw[3]),
                0.02 + 0.3 * torch.sigmoid(neuron_raw[4]),
                0.02 + 0.3 * torch.sigmoid(neuron_raw[5]),
            ],
            dim=0,
        )

        if self.wicks_dim > 0:
            wick_raw = self.wick(x).mean(dim=0).view(8, self.wicks_dim)
            wicks_loc = torch.stack(
                [
                    math.log(0.01) + 0.5 * torch.tanh(wick_raw[0]),
                    -0.042 + 0.015 * torch.tanh(wick_raw[1]),
                    math.log(0.015) + 0.3 * torch.tanh(wick_raw[2]),
                    -0.025 + 0.04 * torch.tanh(wick_raw[3]),
                ],
                dim=0,
            )
            wicks_scale = torch.stack(
                [
                    0.02 + 0.3 * torch.sigmoid(wick_raw[4]),
                    0.002 + 0.02 * torch.sigmoid(wick_raw[5]),
                    0.02 + 0.2 * torch.sigmoid(wick_raw[6]),
                    0.005 + 0.03 * torch.sigmoid(wick_raw[7]),
                ],
                dim=0,
            )
        else:
            wicks_loc = torch.empty(4, 0, device=x.device, dtype=x.dtype)
            wicks_scale = torch.empty(4, 0, device=x.device, dtype=x.dtype)

        if self.gap_dim > 0:
            gap_raw = self.gap(x).mean(dim=0).view(2, self.gap_dim)
            gap_loc = math.log(0.01) + 0.5 * torch.tanh(gap_raw[0])
            gap_scale = 0.02 + 0.3 * torch.sigmoid(gap_raw[1])
        else:
            gap_loc = torch.empty(0, device=x.device, dtype=x.dtype)
            gap_scale = torch.empty(0, device=x.device, dtype=x.dtype)

        return neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale


class VAElegans(PyroModule):
    """Amortized Variational Autoencoder pairing a TCN encoder with RecurrentNematode decoder."""

    def __init__(self, model: RecurrentNematode, neuron_dim: int, wicks_dim: int, gap_dim: int):
        super().__init__()
        self.encoder = Encoder(neuron_dim, wicks_dim, gap_dim)
        self.decoder = model
        self.neuron_dim = int(neuron_dim)
        self.wicks_dim = int(wicks_dim)
        self.gap_dim = int(gap_dim)

        # Amortized posterior EMA for open-loop evaluation when y is withheld (y=None)
        self.register_buffer("_has_amortized_post", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_ema_neuron_loc", torch.zeros(3, self.neuron_dim))
        self.register_buffer("_ema_neuron_scale", torch.ones(3, self.neuron_dim))
        self.register_buffer("_ema_wicks_loc", torch.zeros(4, self.wicks_dim))
        self.register_buffer("_ema_wicks_scale", torch.ones(4, self.wicks_dim))
        self.register_buffer("_ema_gap_loc", torch.zeros(self.gap_dim))
        self.register_buffer("_ema_gap_scale", torch.ones(self.gap_dim))

        # Variational parameters for shared/GCaMP parameters in RecurrentNematode
        self.stim_gain_loc = PyroParam(torch.tensor(math.log(0.04)))
        self.stim_gain_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.obs_sigma_loc = PyroParam(torch.tensor(math.log(0.15)))
        self.obs_sigma_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)

        self.v_th_ca_loc = PyroParam(torch.tensor(-0.048))
        self.v_th_ca_scale = PyroParam(torch.tensor(0.005), constraint=constraints.positive)
        self.v_slope_ca_loc = PyroParam(torch.tensor(math.log(0.012)))
        self.v_slope_ca_scale = PyroParam(torch.tensor(0.15), constraint=constraints.positive)
        self.alpha_ca_loc = PyroParam(torch.tensor(math.log(2.0)))
        self.alpha_ca_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.tau_ca_loc = PyroParam(torch.tensor(math.log(0.6)))
        self.tau_ca_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.k_on_loc = PyroParam(torch.tensor(math.log(4.0)))
        self.k_on_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.k_off_loc = PyroParam(torch.tensor(math.log(1.2)))
        self.k_off_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)

        self.gcamp_gain_loc = PyroParam(torch.full((self.neuron_dim,), math.log(1.5)))
        self.gcamp_gain_scale = PyroParam(
            torch.full((self.neuron_dim,), 0.2), constraint=constraints.positive
        )
        self.gcamp_offset_loc = PyroParam(torch.zeros(self.neuron_dim))
        self.gcamp_offset_scale = PyroParam(
            torch.full((self.neuron_dim,), 0.02), constraint=constraints.positive
        )

    def reconstruct_img(
        self,
        x: torch.Tensor,
        ExternalInput: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Posterior-guided reconstruction given GCaMP trace x and optional stimulus."""
        if ExternalInput is None:
            ExternalInput = torch.zeros_like(x)
        pred = pyro.infer.Predictive(
            self.forward, guide=self.guide, num_samples=1, return_sites=("_RETURN",)
        )(ExternalInput=ExternalInput, mask=None, y=x)
        return pred["_RETURN"].squeeze(0)

    def forward(
        self,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(
            ExternalInput=ExternalInput,
            VoltageClamp=VoltageClamp,
            mask=mask,
            y=y,
            noise_scale=noise_scale,
        )

    def guide(
        self,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
    ) -> None:
        pyro.module("vae_guide", self)
        if y is not None:
            neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = self.encoder(y)
            with torch.no_grad():
                mom = 0.9 if bool(self._has_amortized_post.item()) else 0.0
                self._ema_neuron_loc.mul_(mom).add_(neuron_loc.detach(), alpha=1.0 - mom)
                self._ema_neuron_scale.mul_(mom).add_(neuron_scale.detach(), alpha=1.0 - mom)
                if self.wicks_dim > 0:
                    self._ema_wicks_loc.mul_(mom).add_(wicks_loc.detach(), alpha=1.0 - mom)
                    self._ema_wicks_scale.mul_(mom).add_(wicks_scale.detach(), alpha=1.0 - mom)
                if self.gap_dim > 0:
                    self._ema_gap_loc.mul_(mom).add_(gap_loc.detach(), alpha=1.0 - mom)
                    self._ema_gap_scale.mul_(mom).add_(gap_scale.detach(), alpha=1.0 - mom)
                self._has_amortized_post.fill_(True)
        elif bool(self._has_amortized_post.item()):
            neuron_loc, neuron_scale = self._ema_neuron_loc, self._ema_neuron_scale
            wicks_loc, wicks_scale = self._ema_wicks_loc, self._ema_wicks_scale
            gap_loc, gap_scale = self._ema_gap_loc, self._ema_gap_scale
        else:
            enc_in = ExternalInput if ExternalInput is not None else VoltageClamp
            neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = self.encoder(enc_in)

        # Match exact PyroSample site names registered in RecurrentNematode
        pyro.sample(
            "decoder.model.Neuron.conductance.E",
            dist.Normal(neuron_loc[0], neuron_scale[0]).to_event(1),
        )
        pyro.sample(
            "decoder.model.Neuron.conductance.G",
            dist.LogNormal(neuron_loc[1], neuron_scale[1]).to_event(1),
        )
        pyro.sample(
            "decoder.model.Neuron.conductance.C",
            dist.LogNormal(neuron_loc[2], neuron_scale[2]).to_event(1),
        )

        if self.wicks_dim > 0:
            pyro.sample(
                "decoder.model.synapse.wicks.g_max",
                dist.LogNormal(wicks_loc[0], wicks_scale[0]).to_event(1),
            )
            pyro.sample(
                "decoder.model.synapse.wicks.V_rest",
                dist.Normal(wicks_loc[1], wicks_scale[1]).to_event(1),
            )
            pyro.sample(
                "decoder.model.synapse.wicks.V_slope",
                dist.LogNormal(wicks_loc[2], wicks_scale[2]).to_event(1),
            )
            pyro.sample(
                "decoder.model.synapse.wicks.E_rev",
                dist.Normal(wicks_loc[3], wicks_scale[3]).to_event(1),
            )

        if self.gap_dim > 0:
            pyro.sample(
                "decoder.model.synapse.general.g_syn",
                dist.LogNormal(gap_loc, gap_scale).to_event(1),
            )

        pyro.sample("decoder.model.stim_gain", dist.LogNormal(self.stim_gain_loc, self.stim_gain_scale))
        pyro.sample("decoder.obs_sigma", dist.LogNormal(self.obs_sigma_loc, self.obs_sigma_scale))

        pyro.sample("decoder.gcamp.v_th_ca", dist.Normal(self.v_th_ca_loc, self.v_th_ca_scale))
        pyro.sample("decoder.gcamp.v_slope_ca", dist.LogNormal(self.v_slope_ca_loc, self.v_slope_ca_scale))
        pyro.sample("decoder.gcamp.alpha_ca", dist.LogNormal(self.alpha_ca_loc, self.alpha_ca_scale))
        pyro.sample("decoder.gcamp.tau_ca", dist.LogNormal(self.tau_ca_loc, self.tau_ca_scale))
        pyro.sample("decoder.gcamp.k_on", dist.LogNormal(self.k_on_loc, self.k_on_scale))
        pyro.sample("decoder.gcamp.k_off", dist.LogNormal(self.k_off_loc, self.k_off_scale))
        pyro.sample(
            "decoder.gcamp.gain",
            dist.LogNormal(self.gcamp_gain_loc, self.gcamp_gain_scale).to_event(1),
        )
        pyro.sample(
            "decoder.gcamp.offset",
            dist.Normal(self.gcamp_offset_loc, self.gcamp_offset_scale).to_event(1),
        )
