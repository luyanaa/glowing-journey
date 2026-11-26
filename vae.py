import math
from typing import Optional

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroParam
import torch
import torch.nn as nn
import torch.nn.functional as F

from c302 import CausalPreStimEncoder, RecurrentNematode
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

    def __init__(
        self,
        neuron_dim: int,
        wicks_dim: int,
        gap_dim: int,
        e_rev_prior: Optional[torch.Tensor] = None,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
    ):
        super().__init__()
        self.neuron_dim = int(neuron_dim)
        self.wicks_dim = int(wicks_dim)
        self.gap_dim = int(gap_dim)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)

        if e_rev_prior is not None and self.wicks_dim > 0:
            e_prior = torch.as_tensor(e_rev_prior, dtype=torch.float32).detach().clone().reshape(self.wicks_dim)
        else:
            e_prior = torch.full((self.wicks_dim,), -0.025, dtype=torch.float32)
        self.register_buffer("e_rev_prior", e_prior)

        # Neuron + presynaptic release params: E, log G, log C, wicks.V_rest, log wicks.V_slope -> 5 loc + 5 scale
        self.neuron = TCN(input_size=self.neuron_dim, output_size=self.neuron_dim * 10)
        # Edge-wise chemical synapse params: log g_max, E_rev -> 2 loc + 2 scale
        self.wick = TCN(input_size=self.neuron_dim, output_size=max(1, self.wicks_dim) * 4)
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
        neuron_raw = self.neuron(x).mean(dim=0).view(10, self.neuron_dim)
        v_rest_center = -0.042
        v_slope_center = math.log(0.008) if self.evoked_syn_priors else math.log(0.015)
        neuron_loc = torch.stack(
            [
                -0.065 + 0.02 * torch.tanh(neuron_raw[0]),
                math.log(0.5) + 0.8 * torch.tanh(neuron_raw[1]),
                math.log(0.05) + 0.8 * torch.tanh(neuron_raw[2]),
                v_rest_center + 0.02 * torch.tanh(neuron_raw[3]),
                v_slope_center + 0.5 * torch.tanh(neuron_raw[4]),
            ],
            dim=0,
        )
        neuron_scale = torch.stack(
            [
                0.002 + 0.02 * torch.sigmoid(neuron_raw[5]),
                0.02 + 0.35 * torch.sigmoid(neuron_raw[6]),
                0.02 + 0.35 * torch.sigmoid(neuron_raw[7]),
                0.002 + 0.02 * torch.sigmoid(neuron_raw[8]),
                0.02 + 0.25 * torch.sigmoid(neuron_raw[9]),
            ],
            dim=0,
        )

        if self.wicks_dim > 0:
            gmax_center = math.log(0.015) if self.evoked_syn_priors else math.log(0.01)
            gmax_span = 2.8 if self.wide_amp_priors else 2.2
            erev_span = 0.075 if self.wide_amp_priors else 0.045
            wick_raw = self.wick(x).mean(dim=0).view(4, self.wicks_dim)
            wicks_loc = torch.stack(
                [
                    # Centered on 0.01 nS (or 0.015 nS with evoked_syn_priors) with wide dynamic range
                    gmax_center + gmax_span * torch.tanh(wick_raw[0]),
                    # Centered per-edge on neurotransmitter prior (expanded span when wide_amp_priors)
                    self.e_rev_prior.to(device=x.device, dtype=x.dtype) + erev_span * torch.tanh(wick_raw[1]),
                ],
                dim=0,
            )
            wicks_scale = torch.stack(
                [
                    0.02 + 0.5 * torch.sigmoid(wick_raw[2]),
                    0.005 + 0.035 * torch.sigmoid(wick_raw[3]),
                ],
                dim=0,
            )
        else:
            wicks_loc = torch.empty(2, 0, device=x.device, dtype=x.dtype)
            wicks_scale = torch.empty(2, 0, device=x.device, dtype=x.dtype)

        if self.gap_dim > 0:
            gap_span = 2.6 if self.wide_amp_priors else 2.2
            gap_raw = self.gap(x).mean(dim=0).view(2, self.gap_dim)
            # Centered on 0.015 nS prior with wide dynamic range
            gap_loc = math.log(0.015) + gap_span * torch.tanh(gap_raw[0])
            gap_scale = 0.02 + 0.5 * torch.sigmoid(gap_raw[1])
        else:
            gap_loc = torch.empty(0, device=x.device, dtype=x.dtype)
            gap_scale = torch.empty(0, device=x.device, dtype=x.dtype)

        return neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale


class PreStimVariationalEncoder(PyroModule):
    """Strictly causal recognition network q_phi(z | y_pre) for local/worm latent state."""

    def __init__(self, neuron_dim: int, rank: int = 16, hidden: int = 64, oscillatory: bool = False):
        super().__init__()
        self.neuron_dim = int(neuron_dim)
        self.rank = int(rank)
        self.oscillatory = bool(oscillatory)
        in_mult = 4 if self.oscillatory else 3
        self.fc1 = PyroModule[nn.Linear](self.neuron_dim * in_mult, hidden)
        self.ln = PyroModule[nn.LayerNorm](hidden)
        self.fc2 = PyroModule[nn.Linear](hidden, self.rank * 2)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, y_pre: torch.Tensor, mask_pre: Optional[torch.Tensor] = None):
        if self.oscillatory:
            global_in, _, _, _, _, _ = CausalPreStimEncoder.extract_oscillatory_features(y_pre, mask_pre)
        else:
            global_in, _, _, _, _ = CausalPreStimEncoder.extract_global_features(y_pre, mask_pre)
        h = F.silu(self.ln(self.fc1(global_in)))
        raw = self.fc2(h)
        z_loc = 3.0 * torch.tanh(raw[:, :self.rank])
        z_scale = 0.05 + 0.95 * torch.sigmoid(raw[:, self.rank:])
        return z_loc, z_scale


class VAEGuide(PyroModule):
    """Standalone PyroModule guide owning global connectome/GCaMP posteriors and optional local q_phi(z | y_pre)."""

    def __init__(
        self,
        neuron_dim: int,
        wicks_dim: int,
        gap_dim: int,
        e_rev_prior: Optional[torch.Tensor] = None,
        per_neuron_kinetics: bool = False,
        per_edge_release: bool = False,
        per_neuron_stim_gain: bool = False,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
        direct_global_guide: bool = False,
        use_prestim_encoder: bool = False,
        pre_vols: int = 10,
        prestim_rank: int = 16,
        variational_prestim: bool = False,
        prestim_oscillatory: bool = False,
        use_extrasynaptic: bool = False,
        extra_dim: int = 0,
        dimensionless: bool = False,
    ):
        super().__init__()
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        self.direct_global_guide = bool(direct_global_guide)
        self.dimensionless = bool(dimensionless)
        cond_scale = 2.0 if self.dimensionless else 1.0
        self.neuron_dim = int(neuron_dim)
        self.wicks_dim = int(wicks_dim)
        self.gap_dim = int(gap_dim)
        self.per_neuron_kinetics = bool(per_neuron_kinetics)
        self.per_edge_release = bool(per_edge_release)
        self.per_neuron_stim_gain = bool(per_neuron_stim_gain)
        self.pre_vols = int(pre_vols)
        self.prestim_rank = int(prestim_rank)
        self.variational_prestim = bool(use_prestim_encoder and variational_prestim)
        self.use_extrasynaptic = bool(use_extrasynaptic)
        self.extra_dim = int(extra_dim)

        if e_rev_prior is not None and self.wicks_dim > 0:
            e_prior = torch.as_tensor(e_rev_prior, dtype=torch.float32).detach().clone().reshape(self.wicks_dim)
        else:
            e_prior = torch.full((self.wicks_dim,), -0.025, dtype=torch.float32)
        self.register_buffer("e_rev_prior", e_prior)

        self.register_buffer("_has_amortized_post", torch.tensor(False, dtype=torch.bool))
        if self.direct_global_guide:
            self.encoder = None
            v_slope_init = math.log(0.008) if self.evoked_syn_priors else math.log(0.015)

            self.neuron_E_z = PyroParam(torch.zeros(self.neuron_dim))
            self.neuron_E_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.008), constraint=constraints.interval(1e-4, 0.04)
            )
            if self.dimensionless:
                self.register_buffer("neuron_G_loc", torch.zeros(self.neuron_dim))
                self.register_buffer("neuron_G_scale", torch.full((self.neuron_dim,), 0.15))
                c_init = math.log(0.10)
            else:
                self.neuron_G_loc = PyroParam(torch.full((self.neuron_dim,), math.log(0.5)))
                self.neuron_G_scale = PyroParam(
                    torch.full((self.neuron_dim,), 0.15), constraint=constraints.interval(1e-3, 1.0)
                )
                c_init = math.log(0.05)
            self.neuron_C_loc = PyroParam(torch.full((self.neuron_dim,), c_init))
            self.neuron_C_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.15), constraint=constraints.interval(1e-3, 1.0)
            )
            self.wicks_V_rest_z = PyroParam(torch.zeros(self.neuron_dim))
            self.wicks_V_rest_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.008), constraint=constraints.interval(1e-4, 0.04)
            )
            self.wicks_V_slope_loc = PyroParam(torch.full((self.neuron_dim,), v_slope_init))
            self.wicks_V_slope_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.12), constraint=constraints.interval(1e-3, 0.8)
            )
            if self.wicks_dim > 0:
                self.wicks_g_max_global_z = PyroParam(torch.tensor(0.0))
                self.wicks_g_max_z = PyroParam(torch.zeros(self.wicks_dim))
                self.wicks_g_max_scale = PyroParam(
                    torch.full((self.wicks_dim,), 0.25), constraint=constraints.interval(1e-3, 1.5)
                )
                self.wicks_E_rev_z = PyroParam(torch.zeros(self.wicks_dim))
                self.wicks_E_rev_scale = PyroParam(
                    torch.full((self.wicks_dim,), 0.015), constraint=constraints.interval(1e-4, 0.06)
                )
            if self.gap_dim > 0:
                self.gap_g_syn_global_z = PyroParam(torch.tensor(0.0))
                self.gap_g_syn_z = PyroParam(torch.zeros(self.gap_dim))
                self.gap_g_syn_scale = PyroParam(
                    torch.full((self.gap_dim,), 0.25), constraint=constraints.interval(1e-3, 1.5)
                )
            self.register_buffer("_ema_neuron_loc", torch.zeros(0))
            self.register_buffer("_ema_neuron_scale", torch.ones(0))
            self.register_buffer("_ema_wicks_loc", torch.zeros(0))
            self.register_buffer("_ema_wicks_scale", torch.ones(0))
            self.register_buffer("_ema_gap_loc", torch.zeros(0))
            self.register_buffer("_ema_gap_scale", torch.ones(0))
        else:
            self.encoder = Encoder(
                neuron_dim, wicks_dim, gap_dim,
                e_rev_prior=e_rev_prior,
                wide_amp_priors=self.wide_amp_priors,
                evoked_syn_priors=self.evoked_syn_priors,
            )
            # Amortized posterior EMA for open-loop evaluation when y is withheld (y=None)
            self.register_buffer("_ema_neuron_loc", torch.zeros(5, self.neuron_dim))
            self.register_buffer("_ema_neuron_scale", torch.ones(5, self.neuron_dim))
            self.register_buffer("_ema_wicks_loc", torch.zeros(2, self.wicks_dim))
            self.register_buffer("_ema_wicks_scale", torch.ones(2, self.wicks_dim))
            self.register_buffer("_ema_gap_loc", torch.zeros(self.gap_dim))
            self.register_buffer("_ema_gap_scale", torch.ones(self.gap_dim))

        if self.use_extrasynaptic and self.extra_dim > 0:
            self.extrasyn_gain_z = PyroParam(torch.zeros(self.extra_dim))
            self.extrasyn_gain_scale = PyroParam(
                torch.full((self.extra_dim,), 0.0005), constraint=constraints.interval(1e-5, 0.01)
            )

        self.prestim_q = (
            PreStimVariationalEncoder(
                self.neuron_dim, rank=self.prestim_rank, oscillatory=bool(prestim_oscillatory)
            )
            if self.variational_prestim
            else None
        )

        # Variational parameters for shared/GCaMP parameters in RecurrentNematode
        stim_init = math.log((2.0 if self.wide_amp_priors else 1.2) * cond_scale)
        self.stim_gain_loc = PyroParam(torch.tensor(stim_init))
        self.stim_gain_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.obs_sigma_loc = PyroParam(torch.tensor(math.log(0.15)))
        self.obs_sigma_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)

        # Unconstrained logit params mapped into (V_TH_CA_LOW, V_TH_CA_HIGH)
        self.v_th_ca_loc = PyroParam(torch.tensor(-0.05))
        self.v_th_ca_scale = PyroParam(torch.tensor(0.3), constraint=constraints.positive)
        self.v_slope_ca_loc = PyroParam(torch.tensor(math.log(0.008)))
        self.v_slope_ca_scale = PyroParam(torch.tensor(0.15), constraint=constraints.positive)
        self.alpha_ca_loc = PyroParam(torch.tensor(math.log(2.5)))
        self.alpha_ca_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        self.k_on_loc = PyroParam(torch.tensor(math.log(0.06)))
        self.k_on_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
        if self.per_neuron_kinetics:
            self.tau_ca_loc = PyroParam(
                torch.full((self.neuron_dim,), math.log(7.0))
            )
            self.tau_ca_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.35), constraint=constraints.positive
            )
            self.k_off_loc = PyroParam(
                torch.full((self.neuron_dim,), math.log(0.045))
            )
            self.k_off_scale = PyroParam(
                torch.full((self.neuron_dim,), 0.35), constraint=constraints.positive
            )
        else:
            self.tau_ca_loc = PyroParam(torch.tensor(math.log(7.0)))
            self.tau_ca_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)
            self.k_off_loc = PyroParam(torch.tensor(math.log(0.045)))
            self.k_off_scale = PyroParam(torch.tensor(0.2), constraint=constraints.positive)

        if self.per_neuron_stim_gain:
            stim_vec_scale = 0.35 if self.wide_amp_priors else 0.3
            self.stim_gain_vec_loc = PyroParam(
                torch.full((self.neuron_dim,), stim_init)
            )
            self.stim_gain_vec_scale = PyroParam(
                torch.full((self.neuron_dim,), stim_vec_scale), constraint=constraints.positive
            )

        if self.per_edge_release and self.wicks_dim > 0:
            self.v_rest_edge_loc = PyroParam(torch.zeros(self.wicks_dim))
            self.v_rest_edge_scale = PyroParam(
                torch.full((self.wicks_dim,), 0.005), constraint=constraints.positive
            )
            self.v_slope_edge_loc = PyroParam(torch.zeros(self.wicks_dim))
            self.v_slope_edge_scale = PyroParam(
                torch.full((self.wicks_dim,), 0.1), constraint=constraints.positive
            )

        self.gcamp_gain_loc = PyroParam(torch.full((self.neuron_dim,), math.log(1.4)))
        self.gcamp_gain_scale = PyroParam(
            torch.full((self.neuron_dim,), 0.2), constraint=constraints.positive
        )
        self.gcamp_offset_loc = PyroParam(torch.zeros(self.neuron_dim))
        self.gcamp_offset_scale = PyroParam(
            torch.full((self.neuron_dim,), 0.02), constraint=constraints.positive
        )

    @property
    def wicks_g_max_loc(self) -> torch.Tensor:
        gmax_med = (0.015 if self.evoked_syn_priors else 0.01) * (2.0 if self.dimensionless else 1.0)
        return math.log(gmax_med) + 1.5 * self.wicks_g_max_global_z + 3.0 * self.wicks_g_max_z

    @property
    def gap_g_syn_loc(self) -> torch.Tensor:
        return math.log(0.015 * (2.0 if self.dimensionless else 1.0)) + 1.5 * self.gap_g_syn_global_z + 3.0 * self.gap_g_syn_z

    def _direct_global_params(self):
        neuron_loc = torch.stack(
            [
                -0.065 + 0.01 * self.neuron_E_z,
                self.neuron_G_loc,
                self.neuron_C_loc,
                -0.042 + 0.01 * self.wicks_V_rest_z,
                self.wicks_V_slope_loc,
            ],
            dim=0,
        )
        neuron_scale = torch.stack(
            [
                self.neuron_E_scale,
                self.neuron_G_scale,
                self.neuron_C_scale,
                self.wicks_V_rest_scale,
                self.wicks_V_slope_scale,
            ],
            dim=0,
        )
        if self.wicks_dim > 0:
            wicks_loc = torch.stack(
                [
                    self.wicks_g_max_loc,
                    self.e_rev_prior + 0.04 * self.wicks_E_rev_z,
                ],
                dim=0,
            )
            wicks_scale = torch.stack(
                [self.wicks_g_max_scale, self.wicks_E_rev_scale],
                dim=0,
            )
        else:
            dev, dt = self.neuron_E_z.device, self.neuron_E_z.dtype
            wicks_loc = torch.empty(2, 0, device=dev, dtype=dt)
            wicks_scale = torch.empty(2, 0, device=dev, dtype=dt)
        if self.gap_dim > 0:
            gap_loc = self.gap_g_syn_loc
            gap_scale = self.gap_g_syn_scale
        else:
            dev, dt = self.neuron_E_z.device, self.neuron_E_z.dtype
            gap_loc = torch.empty(0, device=dev, dtype=dt)
            gap_scale = torch.empty(0, device=dev, dtype=dt)
        return neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale

    def forward(
        self,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        dataset_size: Optional[int] = None,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
        drift_gate: Optional[torch.Tensor] = None,
    ) -> None:
        if self.direct_global_guide:
            neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale = self._direct_global_params()
        else:
            # Amortized TCN encoder path (legacy default when direct_global_guide is False)
            encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())
            if encoder_frozen and bool(self._has_amortized_post.item()):
                neuron_loc, neuron_scale = self._ema_neuron_loc, self._ema_neuron_scale
                wicks_loc, wicks_scale = self._ema_wicks_loc, self._ema_wicks_scale
                gap_loc, gap_scale = self._ema_gap_loc, self._ema_gap_scale
            elif y is not None:
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
        if not self.dimensionless:
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
                dist.Normal(neuron_loc[3], neuron_scale[3]).to_event(1),
            )
            pyro.sample(
                "decoder.model.synapse.wicks.V_slope",
                dist.LogNormal(neuron_loc[4], neuron_scale[4]).to_event(1),
            )
            pyro.sample(
                "decoder.model.synapse.wicks.E_rev",
                dist.Normal(wicks_loc[1], wicks_scale[1]).to_event(1),
            )
            if self.per_edge_release:
                pyro.sample(
                    "decoder.model.synapse.wicks.v_rest_edge_delta",
                    dist.Normal(self.v_rest_edge_loc, self.v_rest_edge_scale).to_event(1),
                )
                pyro.sample(
                    "decoder.model.synapse.wicks.v_slope_edge_logdelta",
                    dist.Normal(self.v_slope_edge_loc, self.v_slope_edge_scale).to_event(1),
                )

        if self.gap_dim > 0:
            pyro.sample(
                "decoder.model.synapse.general.g_syn",
                dist.LogNormal(gap_loc, gap_scale).to_event(1),
            )
        if self.use_extrasynaptic and self.extra_dim > 0:
            ex_scale = 0.006 if self.dimensionless else 0.003
            pyro.sample(
                "decoder.model.synapse.extrasyn_gain",
                dist.Normal(ex_scale * self.extrasyn_gain_z, self.extrasyn_gain_scale).to_event(1),
            )

        if self.per_neuron_stim_gain:
            pyro.sample(
                "decoder.model.stim_gain_vec",
                dist.LogNormal(self.stim_gain_vec_loc, self.stim_gain_vec_scale).to_event(1),
            )
        else:
            pyro.sample("decoder.model.stim_gain", dist.LogNormal(self.stim_gain_loc, self.stim_gain_scale))
        pyro.sample("decoder.obs_sigma", dist.LogNormal(self.obs_sigma_loc, self.obs_sigma_scale))

        pyro.sample("decoder.gcamp.v_th_ca_logit", dist.Normal(self.v_th_ca_loc, self.v_th_ca_scale))
        pyro.sample("decoder.gcamp.v_slope_ca", dist.LogNormal(self.v_slope_ca_loc, self.v_slope_ca_scale))
        pyro.sample("decoder.gcamp.alpha_ca", dist.LogNormal(self.alpha_ca_loc, self.alpha_ca_scale))
        if self.per_neuron_kinetics:
            pyro.sample(
                "decoder.gcamp.tau_ca",
                dist.LogNormal(self.tau_ca_loc, self.tau_ca_scale).to_event(1),
            )
            pyro.sample(
                "decoder.gcamp.k_off",
                dist.LogNormal(self.k_off_loc, self.k_off_scale).to_event(1),
            )
        else:
            pyro.sample("decoder.gcamp.tau_ca", dist.LogNormal(self.tau_ca_loc, self.tau_ca_scale))
            pyro.sample("decoder.gcamp.k_off", dist.LogNormal(self.k_off_loc, self.k_off_scale))
        pyro.sample("decoder.gcamp.k_on", dist.LogNormal(self.k_on_loc, self.k_on_scale))
        pyro.sample(
            "decoder.gcamp.gain",
            dist.LogNormal(self.gcamp_gain_loc, self.gcamp_gain_scale).to_event(1),
        )
        pyro.sample(
            "decoder.gcamp.offset",
            dist.Normal(self.gcamp_offset_loc, self.gcamp_offset_scale).to_event(1),
        )

        if self.prestim_q is not None:
            ref = ExternalInput if ExternalInput is not None else (y if y is not None else VoltageClamp)
            batch_size = 1 if (ref is None or ref.dim() == 2) else ref.shape[0]
            dev = self.v_th_ca_loc.device
            dt = self.v_th_ca_loc.dtype
            if dataset_size is not None and int(dataset_size) > batch_size:
                batch_plate = pyro.plate("batch", size=int(dataset_size), subsample_size=batch_size)
            else:
                batch_plate = pyro.plate("batch", batch_size)
            y_pre = pre_stim_obs
            m_pre = pre_stim_mask
            if y_pre is None and y is not None and self.pre_vols > 0:
                t_cut = min(self.pre_vols, y.shape[1])
                y_pre = y[:, :t_cut, :]
                m_pre = None if mask is None else mask[:, :t_cut, :]
            if y_pre is not None and y_pre.shape[1] > 0:
                if y_pre.dim() == 2:
                    y_pre = y_pre.unsqueeze(0)
                    if m_pre is not None and m_pre.dim() == 2:
                        m_pre = m_pre.unsqueeze(0)
                z_loc, z_scale = self.prestim_q(y_pre, m_pre)
            else:
                z_loc = torch.zeros(batch_size, self.prestim_rank, device=dev, dtype=dt)
                z_scale = torch.ones_like(z_loc)
            with batch_plate:
                pyro.sample("prestim_z", dist.Normal(z_loc, z_scale).to_event(1))


class VAElegans(PyroModule):
    """Variational connectome model pairing VAEGuide with RecurrentNematode decoder."""

    def __init__(
        self,
        model: RecurrentNematode,
        neuron_dim: int,
        wicks_dim: int,
        gap_dim: int,
        direct_global_guide: bool = False,
    ):
        super().__init__()
        e_rev_prior = None
        if (
            hasattr(model, "model")
            and hasattr(model.model, "synapse")
            and hasattr(model.model.synapse, "wicks")
            and hasattr(model.model.synapse.wicks, "e_rev_loc_buf")
        ):
            e_rev_prior = model.model.synapse.wicks.e_rev_loc_buf
        per_neuron_kinetics = bool(getattr(getattr(model, "gcamp", None), "per_neuron_kinetics", False))
        per_edge_release = bool(
            getattr(getattr(getattr(model, "model", None), "synapse", None), "per_edge_release", False)
        )
        per_neuron_stim_gain = bool(
            getattr(getattr(model, "model", None), "per_neuron_stim_gain", False)
        )
        wide_amp_priors = bool(
            getattr(getattr(model, "model", None), "wide_amp_priors", False)
        )
        evoked_syn_priors = bool(
            getattr(getattr(model, "model", None), "evoked_syn_priors", False)
        )
        use_prestim_encoder = bool(getattr(model, "use_prestim_encoder", False))
        pre_vols = int(getattr(model, "pre_vols", 10))
        prestim_rank = int(getattr(model, "prestim_rank", 16))
        variational_prestim = bool(getattr(model, "variational_prestim", False))
        prestim_oscillatory = bool(getattr(model, "prestim_oscillatory", False))
        use_extrasynaptic = bool(getattr(getattr(getattr(model, "model", None), "synapse", None), "use_extrasynaptic", False))
        extra_dim = len(getattr(getattr(getattr(model, "model", None), "synapse", None), "Generic_SRC", []))
        dimensionless = bool(
            getattr(getattr(getattr(getattr(model, "model", None), "Neuron", None), "conductance", None), "dimensionless", False)
        )
        self.direct_global_guide = bool(direct_global_guide)
        self.dimensionless = dimensionless
        self.guide = VAEGuide(
            neuron_dim, wicks_dim, gap_dim,
            e_rev_prior=e_rev_prior,
            per_neuron_kinetics=per_neuron_kinetics,
            per_edge_release=per_edge_release,
            per_neuron_stim_gain=per_neuron_stim_gain,
            wide_amp_priors=wide_amp_priors,
            evoked_syn_priors=evoked_syn_priors,
            direct_global_guide=self.direct_global_guide,
            use_prestim_encoder=use_prestim_encoder,
            pre_vols=pre_vols,
            prestim_rank=prestim_rank,
            variational_prestim=variational_prestim,
            prestim_oscillatory=prestim_oscillatory,
            use_extrasynaptic=use_extrasynaptic,
            extra_dim=extra_dim,
            dimensionless=dimensionless,
        )
        self.decoder = model
        self.neuron_dim = int(neuron_dim)
        self.wicks_dim = int(wicks_dim)
        self.gap_dim = int(gap_dim)
        self.per_neuron_kinetics = per_neuron_kinetics
        self.per_edge_release = per_edge_release
        self.per_neuron_stim_gain = per_neuron_stim_gain
        self.wide_amp_priors = wide_amp_priors
        self.evoked_syn_priors = evoked_syn_priors

    @property
    def encoder(self) -> Encoder:
        return self.guide.encoder

    @property
    def _has_amortized_post(self) -> torch.Tensor:
        return self.guide._has_amortized_post

    @property
    def _ema_neuron_loc(self) -> torch.Tensor:
        return self.guide._ema_neuron_loc

    @property
    def _ema_neuron_scale(self) -> torch.Tensor:
        return self.guide._ema_neuron_scale

    @property
    def _ema_wicks_loc(self) -> torch.Tensor:
        return self.guide._ema_wicks_loc

    @property
    def _ema_wicks_scale(self) -> torch.Tensor:
        return self.guide._ema_wicks_scale

    @property
    def _ema_gap_loc(self) -> torch.Tensor:
        return self.guide._ema_gap_loc

    @property
    def _ema_gap_scale(self) -> torch.Tensor:
        return self.guide._ema_gap_scale

    @property
    def stim_gain_loc(self) -> torch.Tensor:
        return self.guide.stim_gain_loc

    @property
    def obs_sigma_loc(self) -> torch.Tensor:
        return self.guide.obs_sigma_loc

    @property
    def v_th_ca_loc(self) -> torch.Tensor:
        return self.guide.v_th_ca_loc

    @property
    def v_slope_ca_loc(self) -> torch.Tensor:
        return self.guide.v_slope_ca_loc

    @property
    def alpha_ca_loc(self) -> torch.Tensor:
        return self.guide.alpha_ca_loc

    @property
    def tau_ca_loc(self) -> torch.Tensor:
        return self.guide.tau_ca_loc

    @property
    def k_on_loc(self) -> torch.Tensor:
        return self.guide.k_on_loc

    @property
    def k_off_loc(self) -> torch.Tensor:
        return self.guide.k_off_loc

    @property
    def gcamp_gain_loc(self) -> torch.Tensor:
        return self.guide.gcamp_gain_loc

    @property
    def gcamp_offset_loc(self) -> torch.Tensor:
        return self.guide.gcamp_offset_loc

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
        dataset_size: Optional[int] = None,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
        drift_gate: Optional[torch.Tensor] = None,
        sample_obs: bool = True,
        ssm_eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(
            ExternalInput=ExternalInput,
            VoltageClamp=VoltageClamp,
            mask=mask,
            y=y,
            noise_scale=noise_scale,
            dataset_size=dataset_size,
            pre_stim_obs=pre_stim_obs,
            pre_stim_mask=pre_stim_mask,
            drift_gate=drift_gate,
            sample_obs=sample_obs,
            ssm_eps=ssm_eps,
        )
