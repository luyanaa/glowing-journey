import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroParam, PyroSample

timeStep = 0.05
value_scale = 200


PyroModuleDict = PyroModule[nn.ModuleDict]
PyroModuleList = PyroModule[nn.ModuleList]


def _sample_normal(ref_mod: nn.Module, loc: float, scale: float, shape=(), to_event: int = 0) -> PyroSample:
    """Create a device-agnostic Normal PyroSample anchored to a registered buffer."""
    def _prior(_mod):
        ref = ref_mod._dev_ref
        l = torch.full(shape, float(loc), device=ref.device, dtype=ref.dtype)
        s = torch.full(shape, float(scale), device=ref.device, dtype=ref.dtype)
        d = dist.Normal(l, s)
        return d.to_event(to_event) if to_event > 0 else d

    return PyroSample(_prior)


def _sample_lognormal(ref_mod: nn.Module, loc: float, scale: float, shape=(), to_event: int = 0) -> PyroSample:
    """Create a device-agnostic LogNormal PyroSample anchored to a registered buffer."""
    def _prior(_mod):
        ref = ref_mod._dev_ref
        l = torch.full(shape, float(loc), device=ref.device, dtype=ref.dtype)
        s = torch.full(shape, float(scale), device=ref.device, dtype=ref.dtype)
        d = dist.LogNormal(l, s)
        return d.to_event(to_event) if to_event > 0 else d

    return PyroSample(_prior)


class GRUModel(PyroModule):
    def __init__(self, hidden_size: int = 8, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = PyroModule[nn.GRU](1, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = PyroModule[nn.Linear](hidden_size, 1)

    def forward(
        self,
        inputSignal: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = inputSignal.dim() == 1
        if squeeze_batch:
            x = inputSignal.view(1, 1, 1)
        elif inputSignal.dim() == 2:
            x = inputSignal.unsqueeze(1)
        else:
            x = inputSignal
        out, next_hx = self.gru(x, hx)
        val = self.head(out[:, -1, :]).squeeze(-1)
        if squeeze_batch:
            val = val.squeeze(0)
        return val, next_hx


class FallbackSensory(PyroModule):
    """Batched GRU-based sensory current modulator."""

    def __init__(self, hidden_size: int = 8, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = PyroModule[nn.GRU](2, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = PyroModule[nn.Linear](hidden_size, 1)

    def forward(
        self,
        inputSignal: torch.Tensor,
        ExternalInput: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ExternalInput is None:
            ExternalInput = torch.zeros_like(inputSignal)
        squeeze_batch = inputSignal.dim() == 0
        if squeeze_batch:
            x = torch.stack([inputSignal, ExternalInput], dim=-1).view(1, 1, 2)
        else:
            x = torch.stack([inputSignal, ExternalInput], dim=-1).unsqueeze(1)
        out, next_state = self.gru(x, state)
        current = self.head(out[:, -1, :]).squeeze(-1)
        if squeeze_batch:
            current = current.squeeze(0)
        return current, next_state


class SpikingModel(PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# https://www.sciencedirect.com/science/article/pii/S0168010223000068
class ASH(PyroModule):
    """Sensory ASH adaptation dynamics (batched, pure functional state passing)."""

    def __init__(self, initInput: float = 0.0):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.initInput = float(initInput)
        self.delta_b = PyroParam(torch.zeros(1))
        self.Xe = _sample_normal(self, 0.9800, 0.0511)
        self.k1 = PyroParam(torch.tensor(1.0))
        self.k2 = PyroParam(torch.tensor(1.0))

    def forward(
        self,
        inputCurrent: torch.Tensor,
        externalInput: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        dt: float = timeStep,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if externalInput is None:
            externalInput = torch.zeros_like(inputCurrent)
        if state is None:
            x_prev = torch.full_like(externalInput, self.initInput)
            past_1 = torch.full_like(externalInput, self.initInput)
            past_2 = torch.full_like(externalInput, self.initInput)
        else:
            x_prev, past_1, past_2 = state

        i1 = -self.k1 * (externalInput - past_1) / dt
        d_cur = (externalInput - past_1) / dt
        d_prev = (past_1 - past_2) / dt
        i2 = torch.clamp(-self.k2 * (d_cur - d_prev) / dt, min=0.14)

        tau = torch.clamp(0.02 * torch.log(torch.clamp(x_prev, min=0.1)) + 0.11, min=0.02)
        dx = i1 + i2 - (x_prev - self.Xe) / tau
        x_next = x_prev + dx * dt
        return x_next, (x_next, externalInput, past_1)


class conductanceLayer(PyroModule):
    """Single-compartment conductance-based non-spiking neuron layer.

    Units:
        V, E: Volts (resting ~ -0.065 V)
        G: nanoSiemens (nS, > 0)
        C: nanoFarads (nF, > 0; tau_m = C / G ~ 0.1 s)
        inputSignal: nanoAmperes (nA)
    Update uses unconditionally stable semi-implicit Euler:
        V_{t+dt} = (V_t + (dt / C) * (I_t + G * E)) / (1 + dt * G / C)
    """

    def __init__(self, input_size: int, dimensionless: bool = False, leak_gauge: str = "unit"):
        super().__init__()
        if leak_gauge not in ("unit", "zero_mean_log"):
            raise ValueError(f"Unsupported leak_gauge={leak_gauge!r}; expected 'unit' or 'zero_mean_log'.")
        self.input_size = input_size
        self.dimensionless = bool(dimensionless)
        self.leak_gauge = str(leak_gauge)
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.E = _sample_normal(self, -0.065, 0.005, shape=(input_size,), to_event=1)
        if self.dimensionless:
            self.register_buffer("G", torch.ones(input_size, dtype=torch.float32))
            if self.leak_gauge == "zero_mean_log":
                self.log_G_raw = _sample_normal(self, 0.0, 0.4, shape=(input_size,), to_event=1)
            self.C = _sample_lognormal(self, math.log(0.10), 0.4, shape=(input_size,), to_event=1)
        else:
            self.G = _sample_lognormal(self, math.log(0.5), 0.4, shape=(input_size,), to_event=1)
            self.C = _sample_lognormal(self, math.log(0.05), 0.4, shape=(input_size,), to_event=1)

    def effective_G(self) -> torch.Tensor:
        """Return strictly positive effective leak conductance vector (N,)."""
        if self.dimensionless and self.leak_gauge == "zero_mean_log":
            raw = self.log_G_raw
            return torch.clamp(torch.exp(raw - raw.mean()), min=1e-4)
        return torch.clamp(self.G, min=1e-4)

    def build_step_cache(self, dt: float = timeStep) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute positive conductance parameters once per rollout."""
        g_leak = self.effective_G()
        c = torch.clamp(self.C, min=1e-4)
        alpha = float(dt) / c
        return g_leak, alpha, g_leak * self.E

    def forward(
        self,
        Prev: torch.Tensor,
        inputSignal: torch.Tensor,
        dt: float = timeStep,
        extraConductance: Optional[torch.Tensor] = None,
        cached: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if cached is None:
            g_leak, alpha, g_leak_e = self.build_step_cache(dt=dt)
        else:
            g_leak, alpha, g_leak_e = cached
        g_total = g_leak if extraConductance is None else (g_leak + torch.clamp(extraConductance, min=0.0))
        return (Prev + alpha * (inputSignal + g_leak_e)) / (1.0 + alpha * g_total)


class neuronLayer(PyroModule):
    """Neuron population layer combining conductance dynamics and optional sensory models."""

    def __init__(
        self,
        neuronSize: int,
        neuronList: Optional[Dict[int, PyroModule]] = None,
        dimensionless: bool = False,
        leak_gauge: str = "unit",
    ):
        super().__init__()
        self.neuronSize = neuronSize
        self.conductance = conductanceLayer(
            input_size=neuronSize, dimensionless=dimensionless, leak_gauge=leak_gauge
        )
        self._neuron_List = PyroModuleDict(
            {str(int(k)): v for k, v in (neuronList or {}).items()}
        )

    def forward(
        self,
        Prev: torch.Tensor,
        inputSignal: torch.Tensor,
        externalInput: Optional[torch.Tensor] = None,
        dt: float = timeStep,
        sensory_states: Optional[Dict[str, object]] = None,
        extraConductance: Optional[torch.Tensor] = None,
        rawExternal: Optional[torch.Tensor] = None,
        conductance_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if externalInput is None:
            total_current = inputSignal
        else:
            total_current = inputSignal + externalInput

        output = self.conductance(
            Prev, total_current, dt=dt, extraConductance=extraConductance, cached=conductance_cache
        )

        if len(self._neuron_List) > 0:
            is_1d = output.dim() == 1
            sensory_ext = rawExternal if rawExternal is not None else externalInput
            ext = torch.zeros_like(total_current) if sensory_ext is None else sensory_ext
            for key, mod in self._neuron_List.items():
                idx = int(key)
                cur_slice = total_current[idx] if is_1d else total_current[:, idx]
                ext_slice = ext[idx] if is_1d else ext[:, idx]
                prev_st = None if sensory_states is None else sensory_states.get(key)
                if isinstance(mod, ASH):
                    mod_out, next_st = mod(cur_slice, ext_slice, state=prev_st, dt=dt)
                    if sensory_states is not None:
                        sensory_states[key] = next_st
                    # Couple ASH normalized adaptation variable gently into current/voltage
                    delta = 0.01 * (mod_out - mod.Xe)
                elif isinstance(mod, FallbackSensory):
                    mod_out, next_st = mod(cur_slice, ext_slice, state=prev_st)
                    if sensory_states is not None:
                        sensory_states[key] = next_st
                    delta = 0.005 * torch.tanh(mod_out)
                else:
                    delta = 0.0
                if is_1d:
                    output[idx] = output[idx] + delta
                else:
                    output = output.clone()
                    output[:, idx] = output[:, idx] + delta
        return output


class GeneralSynapse(PyroModule):
    """Ohmic electrical gap-junction layer: I_{post} = sum_{pre} w * g_syn * (V_pre - V_post)."""

    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_syn=None, symmetric: bool = True, cond_scale: float = 1.0):
        super().__init__()
        self.symmetric = bool(symmetric)
        self.cond_scale = float(cond_scale)
        src = np.asarray(synapseInput, dtype=np.int64).reshape(-1)
        dst = np.asarray(synapseOutput, dtype=np.int64).reshape(-1)
        w = np.asarray(synapseWeight, dtype=np.float32).reshape(-1)
        self.inputSize = int(len(src))
        self.register_buffer("synapseInput", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("synapseOutput", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("synapseWeight", torch.as_tensor(w, dtype=torch.float32))

        self.register_buffer("_dev_ref", torch.tensor(0.0))
        if self.inputSize > 0:
            if g_syn is None:
                # Gap junctions draw zero tonic current at uniform rest; 0.015 median + 0.85 log-std
                # preserves quiet baseline while letting strong homologous pairs reach ~0.08-0.15 nS.
                self.g_syn = _sample_lognormal(
                    self, math.log(0.015 * self.cond_scale), 0.85, shape=(self.inputSize,), to_event=1
                )
            else:
                self.g_syn = g_syn
        else:
            self.register_buffer("g_syn", torch.empty(0, dtype=torch.float32))

    def build_weight_matrix(
        self, n_neurons: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Assemble differentiable dense (N, N) gap-junction conductance matrix and column sums."""
        if self.inputSize == 0:
            return None, None
        g_eff = (torch.clamp(self.g_syn, min=0.0) * self.synapseWeight).to(dtype=dtype)
        w_gap = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype).index_put(
            (self.synapseInput, self.synapseOutput), g_eff, accumulate=True
        )
        if self.symmetric:
            # One row per unordered electrical pair; add the reverse direction so
            # I_i + I_j = 0 for an isolated pair (no external input/dissipation).
            w_gap = w_gap + w_gap.T
        cond_1d = w_gap.sum(dim=0)
        return w_gap, cond_1d

    def forward_terms(
        self,
        inputSignal: torch.Tensor,
        cached: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (synaptic_drive_current, postsynaptic_conductance_sum) for semi-implicit integration."""
        if self.inputSize == 0:
            return torch.zeros_like(inputSignal), torch.zeros_like(inputSignal)
        if cached is None:
            w_gap, cond_1d = self.build_weight_matrix(
                inputSignal.shape[-1], inputSignal.device, inputSignal.dtype
            )
        else:
            w_gap, cond_1d = cached
        if w_gap is None:
            return torch.zeros_like(inputSignal), torch.zeros_like(inputSignal)
        drive = inputSignal @ w_gap
        cond = cond_1d if inputSignal.dim() == 1 else cond_1d.unsqueeze(0).expand_as(inputSignal)
        return drive, cond

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        drive, cond = self.forward_terms(inputSignal)
        return drive - cond * inputSignal


# Graded chemical synapse based on Wicks et al. (1996) / HAL-03705452.
# Per-contact conductance prior (g_max ~ 0.01 nS) and threshold (V_rest ~ -42 mV)
# are scaled for whole-connectome degree (~3,638 chemical synapses across 300 neurons).
class WicksSynapse(PyroModule):
    """Graded chemical synapse layer (Wicks et al. 1996).

    I_{post} = sum_{pre} w * g_max * sigmoid((V_pre - V_rest) / V_slope) * (E_rev - V_post)
    """

    def __init__(
        self,
        synapseInput,
        synapseOutput,
        synapseWeight,
        e_rev_loc=None,
        e_rev_scale=None,
        g_max=None,
        V_rest=None,
        V_slope=None,
        E_rev=None,
        neuron_size: Optional[int] = None,
        per_edge_release: bool = False,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
        cond_scale: float = 1.0,
        use_syn_filter: bool = False,
    ):
        super().__init__()
        src = np.asarray(synapseInput, dtype=np.int64).reshape(-1)
        dst = np.asarray(synapseOutput, dtype=np.int64).reshape(-1)
        w = np.asarray(synapseWeight, dtype=np.float32).reshape(-1)
        self.inputSize = int(len(src))
        self.neuron_size = None if neuron_size is None else int(neuron_size)
        self.share_presynaptic = self.neuron_size is not None
        self.per_edge_release = bool(per_edge_release)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        self.cond_scale = float(cond_scale)
        self.use_syn_filter = bool(use_syn_filter)
        self.register_buffer("synapseInput", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("synapseOutput", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("synapseWeight", torch.as_tensor(w, dtype=torch.float32))

        self.register_buffer("_dev_ref", torch.tensor(0.0))
        if self.inputSize > 0:
            if e_rev_loc is None:
                e_rev_loc_t = torch.full((self.inputSize,), -0.025, dtype=torch.float32)
            else:
                e_rev_loc_t = torch.as_tensor(e_rev_loc, dtype=torch.float32).reshape(-1)
            if e_rev_scale is None:
                e_rev_scale_t = torch.full((self.inputSize,), 0.035, dtype=torch.float32)
            else:
                e_rev_scale_t = torch.as_tensor(e_rev_scale, dtype=torch.float32).reshape(-1)
            self.register_buffer("e_rev_loc_buf", e_rev_loc_t)
            self.register_buffer("e_rev_scale_buf", e_rev_scale_t)

            pre_param_dim = self.neuron_size if self.share_presynaptic else self.inputSize
            # Keep prior median calm (0.01 nS, or 0.015 nS with sharper V_slope=8 mV for evoked_syn_priors)
            # so high-degree hubs avoid resting tonic depolarization while evoked release remains responsive.
            g_max_med = (0.015 if self.evoked_syn_priors else 0.01) * self.cond_scale
            g_max_scale = 0.95 if self.evoked_syn_priors else (1.1 if self.wide_amp_priors else 0.85)
            v_rest_loc = -0.042
            v_slope_loc = math.log(0.008) if self.evoked_syn_priors else math.log(0.015)
            v_slope_scale = 0.35 if self.evoked_syn_priors else 0.3
            self.g_max = (
                _sample_lognormal(self, math.log(g_max_med), g_max_scale, shape=(self.inputSize,), to_event=1)
                if g_max is None
                else g_max
            )
            self.V_rest = (
                _sample_normal(self, v_rest_loc, 0.008, shape=(pre_param_dim,), to_event=1)
                if V_rest is None
                else V_rest
            )
            self.V_slope = (
                _sample_lognormal(self, v_slope_loc, v_slope_scale, shape=(pre_param_dim,), to_event=1)
                if V_slope is None
                else V_slope
            )
            self.E_rev = (
                PyroSample(lambda mod: dist.Normal(mod.e_rev_loc_buf, mod.e_rev_scale_buf).to_event(1))
                if E_rev is None
                else E_rev
            )
            if self.per_edge_release:
                # Zero-mean residual release threshold/slope on top of the shared
                # per-presynaptic-neuron values; allows individual edges to lead/lag.
                self.v_rest_edge_delta = _sample_normal(
                    self, 0.0, 0.005, shape=(self.inputSize,), to_event=1
                )
                self.v_slope_edge_logdelta = _sample_normal(
                    self, 0.0, 0.1, shape=(self.inputSize,), to_event=1
                )
            else:
                self.register_buffer("v_rest_edge_delta", torch.empty(0, dtype=torch.float32))
                self.register_buffer("v_slope_edge_logdelta", torch.empty(0, dtype=torch.float32))
            if self.use_syn_filter:
                # Per-presynaptic-neuron chemical vesicle/receptor first-order filter tau_syn (s)
                self.tau_syn = _sample_lognormal(
                    self, math.log(0.6), 0.7, shape=(pre_param_dim,), to_event=1
                )
            else:
                self.register_buffer("tau_syn", torch.empty(0, dtype=torch.float32))
        else:
            self.register_buffer("g_max", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_rest", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_slope", torch.empty(0, dtype=torch.float32))
            self.register_buffer("E_rev", torch.empty(0, dtype=torch.float32))
            self.register_buffer("v_rest_edge_delta", torch.empty(0, dtype=torch.float32))
            self.register_buffer("v_slope_edge_logdelta", torch.empty(0, dtype=torch.float32))
            self.register_buffer("tau_syn", torch.empty(0, dtype=torch.float32))

    def build_weight_matrices(
        self, n_neurons: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Assemble differentiable dense (N, N) drive/conductance matrices for shared presynaptic params."""
        if self.inputSize == 0 or not self.share_presynaptic or self.per_edge_release:
            return None
        g_w = (torch.clamp(self.g_max, min=0.0) * self.synapseWeight).to(dtype=dtype)
        # Physiological reversal potential guardrail [-95 mV, +30 mV] prevents unphysical +/-200 mV runaway
        e_rev = torch.clamp(self.E_rev, min=-0.095, max=0.030).to(dtype=dtype)
        zeros = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype)
        w_cond = zeros.index_put((self.synapseInput, self.synapseOutput), g_w, accumulate=True)
        w_drive = zeros.index_put((self.synapseInput, self.synapseOutput), g_w * e_rev, accumulate=True)
        v_rest = self.V_rest.to(dtype=dtype)
        v_slope = torch.clamp(self.V_slope, min=1e-3).to(dtype=dtype)
        return w_drive, w_cond, v_rest, v_slope

    def forward_terms(
        self,
        inputSignal: torch.Tensor,
        cached: Optional[Tuple[torch.Tensor, ...]] = None,
        pre_act: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (synaptic_drive_current, postsynaptic_conductance_sum) for semi-implicit integration."""
        if self.inputSize == 0:
            return torch.zeros_like(inputSignal), torch.zeros_like(inputSignal)
        if self.share_presynaptic and not self.per_edge_release:
            mats = (
                cached
                if cached is not None
                else self.build_weight_matrices(
                    inputSignal.shape[-1], inputSignal.device, inputSignal.dtype
                )
            )
            if len(mats) >= 5 and mats[4] is not None:
                v_rest, v_slope, w_both = mats[2], mats[3], mats[4]
                act = pre_act if pre_act is not None else torch.sigmoid((inputSignal - v_rest) / v_slope)
                return (act @ w_both).chunk(2, dim=-1)
            w_drive, w_cond, v_rest, v_slope = mats[:4]
            act = pre_act if pre_act is not None else torch.sigmoid((inputSignal - v_rest) / v_slope)
            return act @ w_drive, act @ w_cond

        drive = torch.zeros_like(inputSignal)
        cond = torch.zeros_like(inputSignal)
        dim = 0 if inputSignal.dim() == 1 else 1
        if pre_act is not None:
            act = pre_act
        else:
            pre_v = torch.index_select(inputSignal, dim, self.synapseInput)
            if self.per_edge_release:
                v_rest = self.V_rest[self.synapseInput] + self.v_rest_edge_delta
                slope = torch.clamp(
                    self.V_slope[self.synapseInput] * torch.exp(self.v_slope_edge_logdelta),
                    min=1e-3,
                )
            else:
                v_rest = self.V_rest
                slope = torch.clamp(self.V_slope, min=1e-3)
            act = torch.sigmoid((pre_v - v_rest) / slope)
        g = torch.clamp(self.g_max, min=0.0)
        e_rev = torch.clamp(self.E_rev, min=-0.095, max=0.030)
        g_eff = g * self.synapseWeight * act
        drive.index_add_(dim, self.synapseOutput, g_eff * e_rev)
        cond.index_add_(dim, self.synapseOutput, g_eff)
        return drive, cond

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        drive, cond = self.forward_terms(inputSignal)
        return drive - cond * inputSignal


class RecurrentSynapse(PyroModule):
    """Optional GRU-based extrasynaptic / plastic connection layer."""

    hidden_size = 8
    num_layers = 2

    def __init__(self, synapseInput, synapseOutput, synapseWeight):
        super().__init__()
        src = np.asarray(synapseInput, dtype=np.int64).reshape(-1)
        dst = np.asarray(synapseOutput, dtype=np.int64).reshape(-1)
        w = np.asarray(synapseWeight, dtype=np.float32).reshape(-1)
        self.register_buffer("synapseInput", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("synapseOutput", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("synapseWeight", torch.as_tensor(w, dtype=torch.float32))
        self.RNN = PyroModuleList(
            [
                PyroModule[nn.Sequential](
                    PyroModule[nn.GRU](2, self.hidden_size, num_layers=self.num_layers, batch_first=True),
                    PyroModule[nn.Linear](self.hidden_size, 1),
                )
                for _ in range(len(src))
            ]
        )

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(inputSignal)
        if len(self.RNN) == 0:
            return output
        is_1d = inputSignal.dim() == 1
        for i, rnn in enumerate(self.RNN):
            pre = inputSignal[self.synapseInput[i]] if is_1d else inputSignal[:, self.synapseInput[i]]
            post = inputSignal[self.synapseOutput[i]] if is_1d else inputSignal[:, self.synapseOutput[i]]
            pair = torch.stack([pre, post], dim=-1)
            pair = pair.view(1, 1, 2) if is_1d else pair.unsqueeze(1)
            gru_out, _ = rnn[0](pair)
            cur = rnn[1](gru_out[:, -1, :]).squeeze(-1) * self.synapseWeight[i]
            if is_1d:
                output[self.synapseOutput[i]] = output[self.synapseOutput[i]] + cur.squeeze(0)
            else:
                output[:, self.synapseOutput[i]] = output[:, self.synapseOutput[i]] + cur
        return output


class synapseLayer(PyroModule):
    """Aggregates chemical (Wicks), electrical (gap junction), and extrasynaptic (NP/MA) currents."""

    def __init__(
        self,
        synapseList: dict,
        use_extrasynaptic: bool = False,
        neuron_size: Optional[int] = None,
        per_edge_release: bool = False,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
        dimensionless: bool = False,
        split_extrasynaptic: bool = False,
        use_syn_filter: bool = False,
        tau_np_prior: float = 4.0,
        tau_ma_prior: float = 1.0,
        tau_mod_prior_std: float = 0.5,
        genotype_conditioned: bool = False,
    ):
        super().__init__()
        self.per_edge_release = bool(per_edge_release)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        self.dimensionless = bool(dimensionless)
        self.split_extrasynaptic = bool(split_extrasynaptic)
        self.use_syn_filter = bool(use_syn_filter)
        self.tau_np_prior = float(tau_np_prior)
        self.tau_ma_prior = float(tau_ma_prior)
        self.tau_mod_prior_std = float(tau_mod_prior_std)
        self.genotype_conditioned = bool(genotype_conditioned)
        cond_scale = 2.0 if self.dimensionless else 1.0
        wicks_tuple = synapseList["Wicks"]
        self.Wicks_SRC, self.Wicks_DST, self.Wicks_Weight = wicks_tuple[:3]
        e_rev_loc = wicks_tuple[3] if len(wicks_tuple) > 3 else None
        e_rev_scale = wicks_tuple[4] if len(wicks_tuple) > 4 else None

        self.Gap_Junction_SRC, self.Gap_Junction_DST, self.Gap_Junction_Weight = synapseList["General"]
        self.Generic_SRC, self.Generic_DST, self.Generic_Weight = synapseList.get("Generic", ([], [], []))
        self.NP_SRC, self.NP_DST, self.NP_Weight = synapseList.get("NP", ([], [], []))
        self.MA_SRC, self.MA_DST, self.MA_Weight = synapseList.get("MA", ([], [], []))
        self.neuron_size = neuron_size

        self.wicks = WicksSynapse(
            self.Wicks_SRC,
            self.Wicks_DST,
            self.Wicks_Weight,
            e_rev_loc=e_rev_loc,
            e_rev_scale=e_rev_scale,
            neuron_size=neuron_size,
            per_edge_release=per_edge_release,
            wide_amp_priors=wide_amp_priors,
            evoked_syn_priors=evoked_syn_priors,
            cond_scale=cond_scale,
            use_syn_filter=self.use_syn_filter,
        )
        self.general = GeneralSynapse(
            self.Gap_Junction_SRC,
            self.Gap_Junction_DST,
            self.Gap_Junction_Weight,
            cond_scale=cond_scale,
        )
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.use_extrasynaptic = bool(use_extrasynaptic)
        if self.use_extrasynaptic and self.split_extrasynaptic:
            if len(self.NP_SRC) > 0:
                self.extrasyn_np_gain = _sample_normal(
                    self, 0.0, 0.0025 * cond_scale, shape=(len(self.NP_SRC),), to_event=1
                )
                self.tau_np = _sample_lognormal(
                    self, math.log(max(self.tau_np_prior, 0.05)), self.tau_mod_prior_std
                )
                self.register_buffer("np_src_t", torch.as_tensor(self.NP_SRC, dtype=torch.long))
                self.register_buffer("np_dst_t", torch.as_tensor(self.NP_DST, dtype=torch.long))
            if len(self.MA_SRC) > 0:
                self.extrasyn_ma_gain = _sample_normal(
                    self, 0.0, 0.0025 * cond_scale, shape=(len(self.MA_SRC),), to_event=1
                )
                self.tau_ma = _sample_lognormal(
                    self, math.log(max(self.tau_ma_prior, 0.05)), self.tau_mod_prior_std
                )
                self.register_buffer("ma_src_t", torch.as_tensor(self.MA_SRC, dtype=torch.long))
                self.register_buffer("ma_dst_t", torch.as_tensor(self.MA_DST, dtype=torch.long))
        elif self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            self.extrasyn_gain = _sample_normal(
                self, 0.0, 0.0025 * cond_scale, shape=(len(self.Generic_SRC),), to_event=1
            )
            self.register_buffer("generic_src_t", torch.as_tensor(self.Generic_SRC, dtype=torch.long))
            self.register_buffer("generic_dst_t", torch.as_tensor(self.Generic_DST, dtype=torch.long))

        if self.genotype_conditioned:
            self.gamma_np_mut = _sample_lognormal(self, 0.0, 0.8)
            self.gamma_ma_mut = _sample_lognormal(self, 0.0, 0.8)
            self.gamma_wired_mut = _sample_lognormal(self, 0.0, 0.25)

    def build_step_cache(
        self, n_neurons: int, device: torch.device, dtype: torch.dtype, dt: float = timeStep
    ) -> dict:
        """Build per-rollout dense (N, N) synaptic matrices and filter decays once per rollout."""
        w_extra = None
        w_extra_np = None
        w_extra_ma = None
        alpha_np = None
        alpha_ma = None
        if self.use_extrasynaptic and self.split_extrasynaptic:
            if len(self.NP_SRC) > 0:
                w_extra_np = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype).index_put(
                    (self.np_src_t, self.np_dst_t),
                    self.extrasyn_np_gain.to(dtype=dtype),
                    accumulate=True,
                )
                tau_np = torch.clamp(self.tau_np.to(device=device, dtype=dtype), min=0.05, max=30.0)
                alpha_np = torch.exp(-float(dt) / tau_np)
            if len(self.MA_SRC) > 0:
                w_extra_ma = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype).index_put(
                    (self.ma_src_t, self.ma_dst_t),
                    self.extrasyn_ma_gain.to(dtype=dtype),
                    accumulate=True,
                )
                tau_ma = torch.clamp(self.tau_ma.to(device=device, dtype=dtype), min=0.05, max=30.0)
                alpha_ma = torch.exp(-float(dt) / tau_ma)
        elif self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            w_extra = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype).index_put(
                (self.generic_src_t, self.generic_dst_t),
                self.extrasyn_gain.to(dtype=dtype),
                accumulate=True,
            )
        wicks_mats = self.wicks.build_weight_matrices(n_neurons, device, dtype)
        if wicks_mats is not None:
            wicks_mats = (*wicks_mats, torch.cat([wicks_mats[0], wicks_mats[1]], dim=1))
        alpha_syn = None
        if self.use_syn_filter and self.wicks.inputSize > 0:
            tau_syn = torch.clamp(self.wicks.tau_syn.to(device=device, dtype=dtype), min=0.02, max=15.0)
            alpha_syn = torch.exp(-float(dt) / tau_syn)
        return {
            "wicks": wicks_mats,
            "general": self.general.build_weight_matrix(n_neurons, device, dtype),
            "extra": w_extra,
            "extra_np": w_extra_np,
            "extra_ma": w_extra_ma,
            "alpha_np": alpha_np,
            "alpha_ma": alpha_ma,
            "alpha_syn": alpha_syn,
        }

    def init_syn_states(
        self, v_rest_eq: torch.Tensor, syn_cache: Optional[dict] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Initialize slow chemical/NP/MA filter states at exact coupled resting equilibrium."""
        states: Dict[str, torch.Tensor] = {}
        if self.use_syn_filter and self.wicks.inputSize > 0:
            w_mats = None if syn_cache is None else syn_cache.get("wicks")
            if w_mats is not None and self.wicks.share_presynaptic and not self.wicks.per_edge_release:
                v_rest, v_slope = w_mats[2], w_mats[3]
                states["chem"] = torch.sigmoid((v_rest_eq - v_rest) / v_slope)
            elif self.wicks.share_presynaptic and not self.wicks.per_edge_release:
                v_rest = self.wicks.V_rest.to(device=v_rest_eq.device, dtype=v_rest_eq.dtype)
                v_slope = torch.clamp(self.wicks.V_slope, min=1e-3).to(device=v_rest_eq.device, dtype=v_rest_eq.dtype)
                states["chem"] = torch.sigmoid((v_rest_eq - v_rest) / v_slope)
            else:
                dim = 0 if v_rest_eq.dim() == 1 else 1
                pre_v = torch.index_select(v_rest_eq, dim, self.wicks.synapseInput)
                if self.wicks.per_edge_release:
                    vr = self.wicks.V_rest[self.wicks.synapseInput] + self.wicks.v_rest_edge_delta
                    vs = torch.clamp(
                        self.wicks.V_slope[self.wicks.synapseInput] * torch.exp(self.wicks.v_slope_edge_logdelta),
                        min=1e-3,
                    )
                else:
                    vr = self.wicks.V_rest
                    vs = torch.clamp(self.wicks.V_slope, min=1e-3)
                states["chem"] = torch.sigmoid((pre_v - vr.to(dtype=v_rest_eq.dtype)) / vs.to(dtype=v_rest_eq.dtype))
        if self.use_extrasynaptic and self.split_extrasynaptic:
            mod_eq = torch.sigmoid((v_rest_eq + 0.042) / 0.012)
            if len(self.NP_SRC) > 0:
                states["np"] = mod_eq
            if len(self.MA_SRC) > 0:
                states["ma"] = mod_eq
        return states if states else None

    def forward_terms(
        self,
        inputSignal: torch.Tensor,
        syn_cache: Optional[dict] = None,
        syn_states: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = timeStep,
        genotype_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wicks_cache = None if syn_cache is None else syn_cache.get("wicks")
        gen_cache = None if syn_cache is None else syn_cache.get("general")

        pre_act = None
        if self.use_syn_filter and self.wicks.inputSize > 0 and syn_states is not None and "chem" in syn_states:
            alpha_syn = None if syn_cache is None else syn_cache.get("alpha_syn")
            if alpha_syn is None:
                tau_syn = torch.clamp(
                    self.wicks.tau_syn.to(device=inputSignal.device, dtype=inputSignal.dtype),
                    min=0.02, max=15.0,
                )
                alpha_syn = torch.exp(-float(dt) / tau_syn)
            if self.wicks.share_presynaptic and not self.wicks.per_edge_release:
                if wicks_cache is not None:
                    v_rest, v_slope = wicks_cache[2], wicks_cache[3]
                else:
                    v_rest = self.wicks.V_rest.to(dtype=inputSignal.dtype)
                    v_slope = torch.clamp(self.wicks.V_slope, min=1e-3).to(dtype=inputSignal.dtype)
                act_inst = torch.sigmoid((inputSignal - v_rest) / v_slope)
            else:
                dim = 0 if inputSignal.dim() == 1 else 1
                pre_v = torch.index_select(inputSignal, dim, self.wicks.synapseInput)
                if self.wicks.per_edge_release:
                    v_rest = self.wicks.V_rest[self.wicks.synapseInput] + self.wicks.v_rest_edge_delta
                    v_slope = torch.clamp(
                        self.wicks.V_slope[self.wicks.synapseInput] * torch.exp(self.wicks.v_slope_edge_logdelta),
                        min=1e-3,
                    )
                else:
                    v_rest = self.wicks.V_rest
                    v_slope = torch.clamp(self.wicks.V_slope, min=1e-3)
                act_inst = torch.sigmoid((pre_v - v_rest.to(dtype=inputSignal.dtype)) / v_slope.to(dtype=inputSignal.dtype))
                if self.wicks.share_presynaptic and alpha_syn.numel() != act_inst.shape[-1]:
                    alpha_syn = alpha_syn[self.wicks.synapseInput]
            pre_act = alpha_syn * syn_states["chem"] + (1.0 - alpha_syn) * act_inst
            syn_states["chem"] = pre_act

        drive_w, cond_w = self.wicks.forward_terms(inputSignal, cached=wicks_cache, pre_act=pre_act)
        drive_g, cond_g = self.general.forward_terms(inputSignal, cached=gen_cache)
        drive = drive_w + drive_g
        cond = cond_w + cond_g

        g_np = 1.0
        g_ma = 1.0
        if self.genotype_conditioned and genotype_id is not None:
            is_mut = (genotype_id.to(device=inputSignal.device) > 0).to(dtype=inputSignal.dtype)
            if inputSignal.dim() >= 2 and is_mut.dim() == 1:
                is_mut = is_mut.unsqueeze(-1)
            g_wired = 1.0 + is_mut * (
                torch.clamp(self.gamma_wired_mut.to(device=inputSignal.device, dtype=inputSignal.dtype), min=0.05, max=10.0) - 1.0
            )
            drive = drive * g_wired
            cond = cond * g_wired
            g_np = 1.0 + is_mut * (
                torch.clamp(self.gamma_np_mut.to(device=inputSignal.device, dtype=inputSignal.dtype), min=0.0, max=10.0) - 1.0
            )
            g_ma = 1.0 + is_mut * (
                torch.clamp(self.gamma_ma_mut.to(device=inputSignal.device, dtype=inputSignal.dtype), min=0.0, max=10.0) - 1.0
            )

        if self.use_extrasynaptic and self.split_extrasynaptic:
            mod_inst = torch.sigmoid((inputSignal + 0.042) / 0.012)
            if len(self.NP_SRC) > 0:
                if syn_states is not None and "np" in syn_states:
                    alpha_np = None if syn_cache is None else syn_cache.get("alpha_np")
                    if alpha_np is None:
                        tau_np = torch.clamp(self.tau_np.to(device=inputSignal.device, dtype=inputSignal.dtype), min=0.05, max=30.0)
                        alpha_np = torch.exp(-float(dt) / tau_np)
                    s_np = alpha_np * syn_states["np"] + (1.0 - alpha_np) * mod_inst
                    syn_states["np"] = s_np
                else:
                    s_np = mod_inst
                w_np = None if syn_cache is None else syn_cache.get("extra_np")
                if w_np is not None:
                    d_np = s_np @ w_np
                else:
                    dim = 0 if s_np.dim() == 1 else 1
                    d_np = torch.zeros_like(drive)
                    d_np.index_add_(dim, self.np_dst_t, self.extrasyn_np_gain * torch.index_select(s_np, dim, self.np_src_t))
                drive = drive + d_np * g_np
            if len(self.MA_SRC) > 0:
                if syn_states is not None and "ma" in syn_states:
                    alpha_ma = None if syn_cache is None else syn_cache.get("alpha_ma")
                    if alpha_ma is None:
                        tau_ma = torch.clamp(self.tau_ma.to(device=inputSignal.device, dtype=inputSignal.dtype), min=0.05, max=30.0)
                        alpha_ma = torch.exp(-float(dt) / tau_ma)
                    s_ma = alpha_ma * syn_states["ma"] + (1.0 - alpha_ma) * mod_inst
                    syn_states["ma"] = s_ma
                else:
                    s_ma = mod_inst
                w_ma = None if syn_cache is None else syn_cache.get("extra_ma")
                if w_ma is not None:
                    d_ma = s_ma @ w_ma
                else:
                    dim = 0 if s_ma.dim() == 1 else 1
                    d_ma = torch.zeros_like(drive)
                    d_ma.index_add_(dim, self.ma_dst_t, self.extrasyn_ma_gain * torch.index_select(s_ma, dim, self.ma_src_t))
                drive = drive + d_ma * g_ma
        elif self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            w_extra = None if syn_cache is None else syn_cache.get("extra")
            if w_extra is not None:
                drive = drive + torch.sigmoid((inputSignal + 0.042) / 0.012) @ w_extra
            else:
                dim = 0 if inputSignal.dim() == 1 else 1
                pre_v = torch.index_select(inputSignal, dim, self.generic_src_t)
                mod_cur = self.extrasyn_gain * torch.sigmoid((pre_v + 0.042) / 0.012)
                drive.index_add_(dim, self.generic_dst_t, mod_cur)
        return drive, cond

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        drive, cond = self.forward_terms(inputSignal)
        return drive - cond * inputSignal

    def adjacency_matrix(self, neuron_size: int) -> np.ndarray:
        """Return boolean (N, N) adjacency A[src, dst] for chemical + gap-junction connectome."""
        adj = np.zeros((neuron_size, neuron_size), dtype=bool)
        if len(self.Wicks_SRC) > 0:
            adj[np.asarray(self.Wicks_SRC, dtype=np.int64), np.asarray(self.Wicks_DST, dtype=np.int64)] = True
        if len(self.Gap_Junction_SRC) > 0:
            g_src = np.asarray(self.Gap_Junction_SRC, dtype=np.int64)
            g_dst = np.asarray(self.Gap_Junction_DST, dtype=np.int64)
            adj[g_src, g_dst] = True
            adj[g_dst, g_src] = True
        np.fill_diagonal(adj, False)
        return adj

    def np_adjacency_matrix(self, neuron_size: int) -> np.ndarray:
        """Return boolean (N, N) adjacency A_np[src, dst] for neuropeptide (NP) links only."""
        adj = np.zeros((neuron_size, neuron_size), dtype=bool)
        if len(self.NP_SRC) > 0:
            adj[np.asarray(self.NP_SRC, dtype=np.int64), np.asarray(self.NP_DST, dtype=np.int64)] = True
        np.fill_diagonal(adj, False)
        return adj

    def ma_adjacency_matrix(self, neuron_size: int) -> np.ndarray:
        """Return boolean (N, N) adjacency A_ma[src, dst] for monoamine (MA) links only."""
        adj = np.zeros((neuron_size, neuron_size), dtype=bool)
        if len(self.MA_SRC) > 0:
            adj[np.asarray(self.MA_SRC, dtype=np.int64), np.asarray(self.MA_DST, dtype=np.int64)] = True
        np.fill_diagonal(adj, False)
        return adj

    def extrasynaptic_adjacency_matrix(self, neuron_size: int) -> np.ndarray:
        """Return boolean (N, N) adjacency A_extra[src, dst] for extrasynaptic NP/MA links."""
        adj = np.zeros((neuron_size, neuron_size), dtype=bool)
        if len(self.Generic_SRC) > 0:
            adj[np.asarray(self.Generic_SRC, dtype=np.int64), np.asarray(self.Generic_DST, dtype=np.int64)] = True
        np.fill_diagonal(adj, False)
        return adj


class BayesianNNResidual(PyroModule):
    """Low-rank Bayesian neural network current residual for unmodeled dynamics."""

    def __init__(self, neuron_size: int, hidden_dim: int = 16, prior_scale: float = 0.03):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuron_size = neuron_size
        self.hidden_dim = hidden_dim
        self.fc1 = PyroModule[nn.Linear](neuron_size, hidden_dim)
        self.fc2 = PyroModule[nn.Linear](hidden_dim, neuron_size)
        self.fc1.weight = _sample_normal(
            self, 0.0, prior_scale, shape=(hidden_dim, neuron_size), to_event=2
        )
        self.fc1.bias = _sample_normal(self, 0.0, prior_scale, shape=(hidden_dim,), to_event=1)
        self.fc2.weight = _sample_normal(
            self, 0.0, prior_scale, shape=(neuron_size, hidden_dim), to_event=2
        )
        self.fc2.bias = _sample_normal(
            self, 0.0, prior_scale * 0.2, shape=(neuron_size,), to_event=1
        )

    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        # Center voltage near physiological rest (-0.065 V) before tanh activation
        h = torch.tanh(self.fc1((voltage + 0.065) * 20.0))
        return self.fc2(h)


class CalciumGCaMPLayer(PyroModule):
    """Differentiable forward conversion from membrane voltage V(t) to GCaMP ΔF/F0.

    Models two biophysical stages per neuron:
      1. Voltage-gated Ca2+ influx & extrusion (semi-implicit Euler):
         d[Ca]/dt = ([Ca]_rest - [Ca]) / tau_ca + alpha_ca * sigmoid((V - V_th) / V_slope)
      2. Cooperative GCaMP6s binding kinetics (bounded fraction b in [0, 1]):
         db/dt = k_on * [Ca]^n * (1 - b) - k_off * b
         y_hat = gain * (b - b_rest) + offset
    """

    V_TH_CA_LOW = -0.060
    V_TH_CA_HIGH = -0.015

    @classmethod
    def logit_to_v_th_ca(cls, logit: Union[float, torch.Tensor]) -> torch.Tensor:
        """Map an unconstrained logit smoothly into strictly bounded (V_TH_CA_LOW, V_TH_CA_HIGH) V."""
        if not isinstance(logit, torch.Tensor):
            logit = torch.as_tensor(logit, dtype=torch.float32)
        s = torch.sigmoid(logit)
        return (1.0 - s) * cls.V_TH_CA_LOW + s * cls.V_TH_CA_HIGH

    @classmethod
    def v_th_ca_to_logit(cls, v_th_ca: Union[float, torch.Tensor]) -> torch.Tensor:
        """Inverse map from voltage in (V_TH_CA_LOW, V_TH_CA_HIGH) V to unconstrained logit."""
        if not isinstance(v_th_ca, torch.Tensor):
            v = torch.as_tensor(v_th_ca, dtype=torch.float32)
        elif not v_th_ca.is_floating_point():
            v = v_th_ca.to(dtype=torch.float32)
        else:
            v = v_th_ca
        p = torch.clamp((v - cls.V_TH_CA_LOW) / (cls.V_TH_CA_HIGH - cls.V_TH_CA_LOW), 1e-6, 1.0 - 1e-6)
        return torch.logit(p)

    @property
    def v_th_ca(self) -> torch.Tensor:
        return self.logit_to_v_th_ca(self.v_th_ca_logit)

    def __init__(
        self,
        neuron_size: int,
        ca_rest: float = 0.05,
        hill_n: float = 2.0,
        per_neuron_kinetics: bool = False,
    ):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuron_size = neuron_size
        self.ca_rest = float(ca_rest)
        self.hill_n = float(hill_n)
        self.per_neuron_kinetics = bool(per_neuron_kinetics)

        # Shared nuclear GCaMP6s indicator kinetics + neuron-specific gain/offset.
        # Sample v_th_ca_logit as an unconstrained Normal and map deterministically via sigmoid
        # into (-60 mV, -15 mV), below excitatory E_rev = 0 V. Using an unconstrained site avoids
        # PyTorch's AffineTransform.codomain == constraints.real pitfall under Pyro AutoGuides,
        # guaranteeing strict physiological bounds under AutoDiagonalNormal, VAE, and NUTS/MCMC.
        # At logit = -0.05: -0.060 + 0.045 * sigmoid(-0.05) ≈ -0.0381 V, keeping resting
        # indicator saturation low (b0 ~ 0.17) while matching nuclear GCaMP6s rise (~8-10 s)
        # and slow decay (~15-25 s) observed in Randi et al. optogenetic recordings.
        self.v_th_ca_logit = _sample_normal(self, -0.05, 0.8)
        self.v_slope_ca = _sample_lognormal(self, math.log(0.008), 0.3)
        self.alpha_ca = _sample_lognormal(self, math.log(2.5), 0.4)
        if self.per_neuron_kinetics:
            # Hierarchical per-neuron Ca2+ extrusion / GCaMP unbinding.  The prior is
            # deliberately tight around the v3 global values so a warm start cannot
            # collapse into arbitrarily fast/slow kinetics.
            self.tau_ca = _sample_lognormal(
                self, math.log(7.0), 0.35, shape=(neuron_size,), to_event=1
            )
            self.k_off = _sample_lognormal(
                self, math.log(0.045), 0.35, shape=(neuron_size,), to_event=1
            )
        else:
            self.tau_ca = _sample_lognormal(self, math.log(7.0), 0.45)
            self.k_off = _sample_lognormal(self, math.log(0.045), 0.45)
        self.k_on = _sample_lognormal(self, math.log(0.06), 0.45)

        self.gain = _sample_lognormal(self, math.log(1.4), 0.5, shape=(neuron_size,), to_event=1)
        self.offset = _sample_normal(self, 0.0, 0.05, shape=(neuron_size,), to_event=1)

    def resting_state(
        self,
        v_rest: torch.Tensor,
        v_th_ca: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute analytical steady-state ([Ca]_ss, b_ss, b_ss) at resting voltage."""
        v_th = self.v_th_ca if v_th_ca is None else v_th_ca
        influx_ss = self.alpha_ca * torch.sigmoid((v_rest - v_th) / self.v_slope_ca)
        ca_ss = self.ca_rest + self.tau_ca * influx_ss
        ca_n = torch.pow(torch.clamp(ca_ss, min=1e-6), self.hill_n)
        rate_on = self.k_on * ca_n
        b_ss = rate_on / (rate_on + self.k_off)
        return ca_ss, b_ss, b_ss

    def build_step_cache(
        self,
        dt: float,
        v_th_ca: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute Ca2+/GCaMP parameters once per rollout instead of per substep."""
        v_th = self.v_th_ca if v_th_ca is None else v_th_ca
        tau_ca = self.tau_ca
        dt_f = float(dt)
        return (
            v_th,
            self.alpha_ca,
            1.0 / self.v_slope_ca,
            self.ca_rest / tau_ca,
            1.0 / (1.0 + dt_f / tau_ca),
            self.k_on,
            self.k_off,
            self.gain,
            self.offset,
        )

    def step(
        self,
        voltage: torch.Tensor,
        ca_prev: torch.Tensor,
        b_prev: torch.Tensor,
        dt: float,
        v_th_ca: Optional[torch.Tensor] = None,
        cached: Optional[tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_f = float(dt)
        if cached is None:
            v_th = self.v_th_ca if v_th_ca is None else v_th_ca
            alpha_ca = self.alpha_ca
            inv_v_slope = 1.0 / self.v_slope_ca
            ca_rest_over_tau = self.ca_rest / self.tau_ca
            inv_denom = 1.0 / (1.0 + dt_f / self.tau_ca)
            k_on = self.k_on
            k_off = self.k_off
        else:
            v_th, alpha_ca, inv_v_slope, ca_rest_over_tau, inv_denom, k_on, k_off = cached[:7]
        influx = alpha_ca * torch.sigmoid((voltage - v_th) * inv_v_slope)
        ca_next = (ca_prev + dt_f * (ca_rest_over_tau + influx)) * inv_denom
        ca_c = torch.clamp(ca_next, min=1e-6)
        ca_n = (ca_c * ca_c) if self.hill_n == 2.0 else torch.pow(ca_c, self.hill_n)
        rate_on = k_on * ca_n
        b_next = (b_prev + dt_f * rate_on) / (1.0 + dt_f * (rate_on + k_off))
        return ca_next, b_next

    def readout(self, b_bound: torch.Tensor, b_baseline: torch.Tensor) -> torch.Tensor:
        return self.gain * (b_bound - b_baseline) + self.offset

    def forward(
        self,
        v_seq: torch.Tensor,
        v_rest: Optional[torch.Tensor] = None,
        dt: float = timeStep,
    ) -> torch.Tensor:
        """Convert a voltage trajectory (B, T, N) or (T, N) into GCaMP ΔF/F0."""
        squeeze = v_seq.dim() == 2
        if squeeze:
            v_seq = v_seq.unsqueeze(0)
        if v_rest is None:
            v_rest = torch.full_like(v_seq[:, 0, :], -0.065)
        v_th_ca = self.v_th_ca
        ca, b, b_base = self.resting_state(v_rest, v_th_ca=v_th_ca)
        outs = []
        for t in range(v_seq.shape[1]):
            ca, b = self.step(v_seq[:, t, :], ca, b, dt=dt, v_th_ca=v_th_ca)
            outs.append(self.readout(b, b_base))
        res = torch.stack(outs, dim=1)
        return res.squeeze(0) if squeeze else res


class CausalPreStimEncoder(PyroModule):
    """Strictly causal pre-stimulus encoder mapping y[:, :pre_vols, :] to initial state & slow drift.

    Never reads any frame at t >= pre_vols or stimulus ExternalInput. Combines:
      1. Shared per-neuron local encoder (4 -> 16 -> 5, or 5 -> 16 -> 6 with oscillatory=True)
         over masked summary features producing initial (delta_v0, delta_ca0, i_bg_local) and
         fluorescence relaxation modes (drift_fast0, drift_slow0[, drift_vel0]).
      2. Zero-initialized low-rank whole-brain bottleneck (3*N or 4*N -> 32 -> rank -> 2*N [+ osc])
         capturing cross-neuron spontaneous behavioral modes (tonic current, slow drift, and optional
         damped oscillatory modes).
      3. Learnable positive decay timescales (tau_bg, tau_fast, tau_slow[, tau_vel, osc_omega, osc_gamma]).
    """

    def __init__(self, neuron_size: int, rank: int = 16, local_hidden: int = 16, oscillatory: bool = False):
        super().__init__()
        self.neuron_size = int(neuron_size)
        self.rank = int(rank)
        self.oscillatory = bool(oscillatory)

        local_in_dim = 5 if self.oscillatory else 4
        local_out_dim = 6 if self.oscillatory else 5
        global_in_mult = 4 if self.oscillatory else 3

        self.local_fc1 = PyroModule[nn.Linear](local_in_dim, local_hidden)
        self.local_fc2 = PyroModule[nn.Linear](local_hidden, local_out_dim)

        self.global_fc1 = PyroModule[nn.Linear](self.neuron_size * global_in_mult, 32)
        self.global_ln = PyroModule[nn.LayerNorm](32)
        self.global_fc2 = PyroModule[nn.Linear](32, self.rank)
        self.global_out = PyroModule[nn.Linear](self.rank, self.neuron_size * 2)

        # Residual local persistence gains and decay timescales (seconds)
        self.fast_gain = PyroParam(torch.tensor(0.45))
        self.slow_gain = PyroParam(torch.tensor(0.30))
        self.tau_bg = PyroParam(torch.tensor(12.0), constraint=constraints.greater_than(0.5))
        self.tau_fast = PyroParam(torch.tensor(2.5), constraint=constraints.greater_than(0.5))
        self.tau_slow = PyroParam(torch.tensor(20.0), constraint=constraints.greater_than(1.0))

        with torch.no_grad():
            self.local_fc2.weight.mul_(0.05)
            self.local_fc2.bias.zero_()
            self.global_out.weight.zero_()
            self.global_out.bias.zero_()

        if self.oscillatory:
            self.vel_gain = PyroParam(torch.tensor(0.25))
            self.tau_vel = PyroParam(torch.tensor(5.0), constraint=constraints.greater_than(0.5))
            self.n_osc = max(1, min(6, self.rank // 2)) if self.rank >= 2 else 0
            self.n_real = self.rank - 2 * self.n_osc
            if self.n_osc > 0:
                self.osc_omega = PyroParam(
                    torch.linspace(0.12, 0.52, self.n_osc),
                    constraint=constraints.interval(0.02, 1.5),
                )
                self.osc_gamma = PyroParam(
                    torch.linspace(0.05, 0.14, self.n_osc),
                    constraint=constraints.interval(0.015, 1.0),
                )
            else:
                self.register_buffer("osc_omega", torch.empty(0, dtype=torch.float32))
                self.register_buffer("osc_gamma", torch.empty(0, dtype=torch.float32))
            if self.n_real > 0:
                self.real_gamma = PyroParam(
                    torch.linspace(0.04, 0.25, self.n_real),
                    constraint=constraints.interval(0.01, 2.0),
                )
            else:
                self.register_buffer("real_gamma", torch.empty(0, dtype=torch.float32))
            self.osc_out = PyroModule[nn.Linear](self.rank, self.neuron_size)
            with torch.no_grad():
                self.osc_out.weight.zero_()
                self.osc_out.bias.zero_()
        else:
            self.n_osc = 0
            self.n_real = 0

    @staticmethod
    def _prepare_masked_pre(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_pre.dim() == 2:
            y_pre = y_pre.unsqueeze(0)
            if mask_pre is not None and mask_pre.dim() == 2:
                mask_pre = mask_pre.unsqueeze(0)
        dtype, device = y_pre.dtype, y_pre.device
        m = torch.ones_like(y_pre, dtype=dtype) if mask_pre is None else mask_pre.to(device=device, dtype=dtype)
        y_raw_masked = torch.where(
            m > 0.5, torch.nan_to_num(y_pre, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(y_pre)
        )
        clean_pre = (y_raw_masked.abs().max(dim=1).values <= 2.0).to(dtype=dtype)
        y_safe = torch.where(m > 0.5, 0.8 * torch.tanh(y_raw_masked / 0.8), torch.zeros_like(y_pre))
        obs_any = (m.sum(dim=1) > 0.5).to(dtype=dtype) * clean_pre
        return y_safe, m, clean_pre, obs_any

    @staticmethod
    def _extract_features_impl(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_safe, m, clean_pre, obs_any = CausalPreStimEncoder._prepare_masked_pre(y_pre, mask_pre)
        t_pre = y_safe.shape[1]
        dtype, device = y_safe.dtype, y_safe.device

        t_idx = torch.arange(t_pre, device=device, dtype=torch.long).view(1, t_pre, 1)
        valid_idx = torch.where(m > 0.5, t_idx, torch.full_like(t_idx, -1))
        last_idx = valid_idx.max(dim=1).values.clamp(min=0)
        y_last = y_safe.gather(1, last_idx.unsqueeze(1)).squeeze(1) * obs_any

        w_rec = torch.exp(-0.5 * (t_pre - 1 - t_idx.to(dtype=dtype))) * m
        y_recent = (y_safe * w_rec).sum(dim=1) / torch.clamp(w_rec.sum(dim=1), min=1e-6)
        y_recent = y_recent * obs_any

        if t_pre >= 2:
            mid = t_pre // 2
            m1, m2 = m[:, :mid, :], m[:, mid:, :]
            c1, c2 = m1.sum(dim=1), m2.sum(dim=1)
            mu1 = (y_safe[:, :mid, :] * m1).sum(dim=1) / torch.clamp(c1, min=1.0)
            mu2 = (y_safe[:, mid:, :] * m2).sum(dim=1) / torch.clamp(c2, min=1.0)
            has_both = ((c1 > 0.5) & (c2 > 0.5)).to(dtype=dtype) * clean_pre
            y_slope = torch.clamp(mu2 - mu1, -1.5, 1.5) * has_both
        else:
            y_slope = torch.zeros_like(y_last)

        if t_pre >= 3:
            t1, t2 = t_pre // 3, (2 * t_pre) // 3
            ma, mb, mc = m[:, :t1, :], m[:, t1:t2, :], m[:, t2:, :]
            ca, cb, cc = ma.sum(dim=1), mb.sum(dim=1), mc.sum(dim=1)
            mua = (y_safe[:, :t1, :] * ma).sum(dim=1) / torch.clamp(ca, min=1.0)
            mub = (y_safe[:, t1:t2, :] * mb).sum(dim=1) / torch.clamp(cb, min=1.0)
            muc = (y_safe[:, t2:, :] * mc).sum(dim=1) / torch.clamp(cc, min=1.0)
            has_all = ((ca > 0.5) & (cb > 0.5) & (cc > 0.5)).to(dtype=dtype) * clean_pre
            y_accel = torch.clamp(muc - 2.0 * mub + mua, -1.5, 1.5) * has_all
        else:
            y_accel = torch.zeros_like(y_last)

        return y_last, y_recent, y_slope, y_accel, obs_any

    @staticmethod
    def extract_global_features(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly causal artifact-safe summary features extracted solely from ``y[:, :pre_vols]``."""
        y_last, y_recent, y_slope, _, obs_any = CausalPreStimEncoder._extract_features_impl(y_pre, mask_pre)
        global_in = torch.cat([y_last, y_slope, obs_any], dim=-1)  # (B, 3*N)
        return global_in, y_last, y_recent, y_slope, obs_any

    @staticmethod
    def extract_oscillatory_features(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly causal summary features including pre-stim curvature ``y_accel``."""
        y_last, y_recent, y_slope, y_accel, obs_any = CausalPreStimEncoder._extract_features_impl(y_pre, mask_pre)
        global_in = torch.cat([y_last, y_slope, y_accel, obs_any], dim=-1)  # (B, 4*N)
        return global_in, y_last, y_recent, y_slope, y_accel, obs_any

    def forward(
        self,
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
        z_bg: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Return initial state / drift tuple from pre-stimulus observations only."""
        n_neu = self.neuron_size
        if self.oscillatory:
            global_in, y_last, y_recent, y_slope, y_accel, obs_any = self.extract_oscillatory_features(y_pre, mask_pre)
            feat_local = torch.stack([y_last, y_recent, y_slope, y_accel, obs_any], dim=-1)  # (B, N, 5)
        else:
            global_in, y_last, y_recent, y_slope, obs_any = self.extract_global_features(y_pre, mask_pre)
            feat_local = torch.stack([y_last, y_recent, y_slope, obs_any], dim=-1)  # (B, N, 4)

        h_local = F.silu(self.local_fc1(feat_local))
        raw_local = self.local_fc2(h_local) * obs_any.unsqueeze(-1)

        delta_v0 = 0.025 * torch.tanh(raw_local[..., 0])
        delta_ca0 = 0.5 * torch.tanh(raw_local[..., 1])
        i_bg_local = 0.025 * torch.tanh(raw_local[..., 2])
        drift_fast0 = torch.clamp(self.fast_gain, -1.5, 1.5) * y_last + 0.5 * torch.tanh(raw_local[..., 3]) * obs_any
        drift_slow_local = torch.clamp(self.slow_gain, -1.5, 1.5) * y_recent + 0.5 * torch.tanh(raw_local[..., 4]) * obs_any

        if z_bg is None:
            z_h = F.silu(self.global_ln(self.global_fc1(global_in)))
            z_bg = self.global_fc2(z_h)
        global_out = self.global_out(z_bg)
        i_bg_global = 0.015 * torch.tanh(global_out[:, :n_neu])
        drift_slow_global = 0.25 * torch.tanh(global_out[:, n_neu:])

        i_bg0 = i_bg_local + i_bg_global
        drift_slow0 = drift_slow_local + drift_slow_global
        base = (
            delta_v0,
            delta_ca0,
            i_bg0,
            drift_fast0,
            drift_slow0,
            self.tau_bg,
            self.tau_fast,
            self.tau_slow,
        )
        if not self.oscillatory:
            return base
        drift_vel0 = torch.clamp(self.vel_gain, -1.5, 1.5) * y_slope + 0.5 * torch.tanh(raw_local[..., 5]) * obs_any
        return (*base, drift_vel0, self.tau_vel, z_bg)

    def compute_post_drift(
        self,
        prestim_state: tuple,
        n_post: int,
        dt_obs: float,
        drift_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute vectorized (B, n_post, N) fluorescence drift for t >= inject_t."""
        drift_fast0, drift_slow0 = prestim_state[3], prestim_state[4]
        tau_fast, tau_slow = prestim_state[6], prestim_state[7]
        bsz, n_neu = drift_fast0.shape
        device, dtype = drift_fast0.device, drift_fast0.dtype
        t_rel = torch.arange(n_post, device=device, dtype=dtype).view(1, n_post, 1) * float(dt_obs)
        post = (
            drift_fast0.unsqueeze(1) * torch.exp(-t_rel / tau_fast)
            + drift_slow0.unsqueeze(1) * torch.exp(-t_rel / tau_slow)
        )
        if not self.oscillatory or len(prestim_state) < 11:
            return post
        drift_vel0, tau_vel, z_bg = prestim_state[8], prestim_state[9], prestim_state[10]
        u_vel = t_rel / tau_vel
        post = post + drift_vel0.unsqueeze(1) * u_vel * torch.exp(-u_vel)

        t_col = t_rel.squeeze(-1).unsqueeze(-1)  # (1, n_post, 1)
        parts = []
        if self.n_osc > 0:
            omega = self.osc_omega.view(1, 1, self.n_osc)
            gamma = self.osc_gamma.view(1, 1, self.n_osc)
            theta = t_col * omega
            decay = torch.exp(-t_col * gamma)
            c, s = torch.cos(theta), torch.sin(theta)
            z_osc = z_bg[:, : 2 * self.n_osc].view(bsz, 1, self.n_osc, 2)
            u0, v0 = z_osc[..., 0], z_osc[..., 1]
            u_t = decay * (u0 * c - v0 * s)
            v_t = decay * (u0 * s + v0 * c)
            parts.append(torch.stack([u_t, v_t], dim=-1).reshape(bsz, n_post, 2 * self.n_osc))
        if self.n_real > 0:
            r_gamma = self.real_gamma.view(1, 1, self.n_real)
            r0 = z_bg[:, 2 * self.n_osc :].unsqueeze(1)
            parts.append(r0 * torch.exp(-t_col * r_gamma))
        z_t = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        osc_term = 0.45 * torch.tanh(self.osc_out(z_t))
        if drift_gate is not None:
            g = drift_gate.to(device=device, dtype=dtype)
            if g.dim() == 3:
                g = g[4]
            elif g.dim() == 1:
                g = g.unsqueeze(0)
            osc_term = osc_term * g.unsqueeze(1)
        return post + osc_term


class LatentNeuralSSM(PyroModule):
    """Low-dimensional stochastic neural state-space residual coupled to membrane dynamics.

    Supports:
      - ``encoder_type="pooling"`` (4 causal summary stats) or ``"gru"`` (mask-aware sequential GRU).
      - ``split_obs_latent=True`` separating ``z_neural`` (drives ``latent_current`` only, never
        direct ``obs_residual``) from autonomous ``z_artifact`` (drives ``obs_residual`` only,
        zero stimulus/voltage input).
      - ``variational_traj=True`` enabling explicit per-step variational trajectory posterior
        ``q_phi(z_t | z_{t-1}, y_{<=t})`` (and optional backward ``use_smoother=True`` during training),
        as well as causal online ``filter_step`` / ``filter_posterior`` for Task 2.
    """

    def __init__(
        self,
        neuron_size: int,
        z_dim: int = 16,
        encoder_type: str = "pooling",
        split_obs_latent: bool = False,
        artifact_dim: int = 8,
        variational_traj: bool = False,
        use_smoother: bool = False,
    ):
        super().__init__()
        if encoder_type not in ("pooling", "gru"):
            raise ValueError(f"Unsupported ssm encoder_type={encoder_type!r}; expected 'pooling' or 'gru'.")
        self.neuron_size = int(neuron_size)
        self.z_dim = int(z_dim)
        self.encoder_type = str(encoder_type)
        self.split_obs_latent = bool(split_obs_latent)
        self.artifact_dim = int(artifact_dim) if self.split_obs_latent else 0
        self.total_z_dim = self.z_dim + self.artifact_dim
        self.variational_traj = bool(variational_traj)
        self.use_smoother = bool(use_smoother)

        # Scale-invariant multi-window causal summary encoder (works for pre_vols >= 1)
        self.pre_global_fc1 = PyroModule[nn.Linear](self.neuron_size * 4, 64)
        self.pre_global_ln = PyroModule[nn.LayerNorm](64)
        self.pre_global_fc2 = PyroModule[nn.Linear](64, self.z_dim * 2)

        self.local_fc1 = PyroModule[nn.Linear](5, 16)
        self.local_fc2 = PyroModule[nn.Linear](16, 2)  # (delta_v0, delta_ca0)

        # Autonomous stable transition & closed-loop membrane feedback + residual readouts
        self.A_decay_logit = PyroParam(torch.full((self.z_dim,), 2.2))
        self.S_skew = PyroParam(torch.zeros(self.z_dim, self.z_dim))
        self.sigma_eta_log = PyroParam(torch.full((self.z_dim,), math.log(0.04)))

        self.v_proj = PyroModule[nn.Linear](self.neuron_size, 32, bias=False)
        self.v_to_z = PyroModule[nn.Linear](32, self.z_dim, bias=False)
        self.z_to_current = PyroModule[nn.Linear](self.z_dim, self.neuron_size, bias=False)
        self.z_to_obs = PyroModule[nn.Linear](self.z_dim, self.neuron_size, bias=False)
        # Task-2 strictly causal innovation gain: z_{t|t} = z_{t|t-1} + K_obs(masked_innov)
        self.K_obs = PyroModule[nn.Linear](self.neuron_size, self.z_dim, bias=False)

        with torch.no_grad():
            self.pre_global_fc2.weight.zero_()
            self.pre_global_fc2.bias.zero_()
            self.local_fc2.weight.mul_(0.05)
            self.local_fc2.bias.zero_()
            self.v_to_z.weight.zero_()
            self.z_to_current.weight.zero_()
            self.z_to_obs.weight.zero_()
            self.K_obs.weight.mul_(0.02)

        if self.encoder_type == "gru":
            self.gru_in_proj = PyroModule[nn.Linear](self.neuron_size * 3, 64)
            self.gru_ln = PyroModule[nn.LayerNorm](64)
            self.pre_gru = PyroModule[nn.GRU](64, 64, batch_first=True)
            self.gru_z_head = PyroModule[nn.Linear](64, self.total_z_dim * 2)
            with torch.no_grad():
                self.gru_z_head.weight.zero_()
                self.gru_z_head.bias.zero_()

        if self.split_obs_latent:
            self.A_art_decay_logit = PyroParam(torch.full((self.artifact_dim,), 3.0))
            self.S_art_skew = PyroParam(torch.zeros(self.artifact_dim, self.artifact_dim))
            self.sigma_art_log = PyroParam(torch.full((self.artifact_dim,), math.log(0.02)))
            self.art_to_obs = PyroModule[nn.Linear](self.artifact_dim, self.neuron_size, bias=False)
            if self.encoder_type == "pooling":
                self.pre_art_fc2 = PyroModule[nn.Linear](64, self.artifact_dim * 2)
                with torch.no_grad():
                    self.pre_art_fc2.weight.zero_()
                    self.pre_art_fc2.bias.zero_()
            with torch.no_grad():
                self.art_to_obs.weight.zero_()

        if self.variational_traj:
            self.q_traj_in = PyroModule[nn.Linear](self.neuron_size * 3, 48)
            self.q_traj_ln = PyroModule[nn.LayerNorm](48)
            self.q_traj_gru = PyroModule[nn.GRU](48, 48, batch_first=True)
            if self.use_smoother:
                self.q_smooth_gru = PyroModule[nn.GRU](48, 48, batch_first=True)
            head_in = 96 if self.use_smoother else 48
            self.q_traj_head = PyroModule[nn.Linear](head_in, self.z_dim * 2)
            self.q_filter_sigma_log = PyroParam(torch.full((self.z_dim,), math.log(0.05)))
            with torch.no_grad():
                self.q_traj_head.weight.zero_()
                self.q_traj_head.bias.zero_()

    @staticmethod
    def extract_causal_pre_features(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
        dt_obs: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract length-invariant causal features from arbitrary pre_vols."""
        y_safe, m, clean_pre, obs_any = CausalPreStimEncoder._prepare_masked_pre(y_pre, mask_pre)
        B, t_pre, N = y_safe.shape
        dtype, device = y_safe.dtype, y_safe.device

        t_idx = torch.arange(t_pre, device=device, dtype=dtype).view(1, t_pre, 1)
        valid_idx = torch.where(m > 0.5, torch.arange(t_pre, device=device).view(1, t_pre, 1), torch.full((1, t_pre, 1), -1, device=device))
        last_idx = valid_idx.max(dim=1).values.clamp(min=0)
        y_last = y_safe.gather(1, last_idx.unsqueeze(1)).squeeze(1) * obs_any

        tau_rec = max(dt_obs, 10.0)
        w_rec = torch.exp(-(t_pre - 1 - t_idx) * dt_obs / tau_rec) * m
        y_slow = (y_safe * w_rec).sum(dim=1) / torch.clamp(w_rec.sum(dim=1), min=1e-6) * obs_any
        y_all = (y_safe * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0) * obs_any

        _, _, y_slope, _, _ = CausalPreStimEncoder._extract_features_impl(y_safe, m)
        global_feat = torch.cat([y_last, y_slow, y_all, y_slope], dim=-1)  # (B, 4*N)
        local_feat = torch.stack([y_last, y_slow, y_all, y_slope, obs_any], dim=-1)  # (B, N, 5)
        return global_feat, local_feat

    def encode_initial(
        self,
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
        dt_obs: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode pre_stim slice to (z0_loc, z0_scale, delta_v0, delta_ca0)."""
        global_feat, local_feat = self.extract_causal_pre_features(y_pre, mask_pre, dt_obs=dt_obs)
        if self.encoder_type == "gru":
            y_safe, m, _, _ = CausalPreStimEncoder._prepare_masked_pre(y_pre, mask_pre)
            B, t_pre, N = y_safe.shape
            gaps = []
            g = torch.ones((B, N), device=y_safe.device, dtype=y_safe.dtype)
            inv_t = 1.0 / float(max(t_pre, 1))
            for t_i in range(t_pre):
                m_i = m[:, t_i, :]
                g = torch.where(m_i > 0.5, torch.zeros_like(g), torch.clamp(g + inv_t, max=1.0))
                gaps.append(g)
            gap_seq = torch.stack(gaps, dim=1)
            seq_in = torch.cat([y_safe * m, m, gap_seq], dim=-1)
            h_seq = F.silu(self.gru_ln(self.gru_in_proj(seq_in)))
            gru_out, _ = self.pre_gru(h_seq)
            raw = self.gru_z_head(gru_out[:, -1, :])
            z0_loc = 2.0 * torch.tanh(raw[:, :self.total_z_dim])
            z0_scale = 0.05 + 0.45 * torch.sigmoid(raw[:, self.total_z_dim:])
        else:
            h = F.silu(self.pre_global_ln(self.pre_global_fc1(global_feat)))
            raw = self.pre_global_fc2(h)
            z0_loc_n = 2.0 * torch.tanh(raw[:, :self.z_dim])
            z0_scale_n = 0.05 + 0.45 * torch.sigmoid(raw[:, self.z_dim:])
            if self.split_obs_latent:
                raw_a = self.pre_art_fc2(h)
                z0_loc = torch.cat([z0_loc_n, 2.0 * torch.tanh(raw_a[:, :self.artifact_dim])], dim=-1)
                z0_scale = torch.cat([z0_scale_n, 0.05 + 0.45 * torch.sigmoid(raw_a[:, self.artifact_dim:])], dim=-1)
            else:
                z0_loc, z0_scale = z0_loc_n, z0_scale_n

        loc_out = self.local_fc2(F.silu(self.local_fc1(local_feat)))
        delta_v0 = 0.015 * torch.tanh(loc_out[..., 0])
        delta_ca0 = 0.5 * torch.tanh(loc_out[..., 1])
        return z0_loc, z0_scale, delta_v0, delta_ca0

    def transition_matrix(self) -> torch.Tensor:
        """Build stable discrete-time transition matrix A (spectral radius < 0.98)."""
        decay = 0.98 * torch.sigmoid(self.A_decay_logit)
        skew = 0.5 * (self.S_skew - self.S_skew.T)
        rot = torch.linalg.matrix_exp(skew)
        return decay.unsqueeze(1) * rot

    def artifact_transition_matrix(self) -> Optional[torch.Tensor]:
        """Build slow autonomous artifact transition matrix A_art (spectral radius < 0.99)."""
        if not self.split_obs_latent:
            return None
        decay = 0.99 * torch.sigmoid(self.A_art_decay_logit)
        skew = 0.5 * (self.S_art_skew - self.S_art_skew.T)
        rot = torch.linalg.matrix_exp(skew)
        return decay.unsqueeze(1) * rot

    def sigma_eta(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.sigma_eta_log), min=1e-4, max=0.5)

    def sigma_art(self) -> Optional[torch.Tensor]:
        if not self.split_obs_latent:
            return None
        return torch.clamp(torch.exp(self.sigma_art_log), min=1e-4, max=0.5)

    def latent_current(self, z: torch.Tensor) -> torch.Tensor:
        return 0.05 * torch.tanh(self.z_to_current(z))

    def obs_residual(self, z_art_or_z: torch.Tensor) -> torch.Tensor:
        if self.split_obs_latent:
            return self.art_to_obs(z_art_or_z)
        return self.z_to_obs(z_art_or_z)

    def step_latent(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        v_rest: torch.Tensor,
        A_mat: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A = self.transition_matrix() if A_mat is None else A_mat
        v_feat = torch.tanh(self.v_proj(torch.tanh((v - v_rest) / 0.025)))
        fb = 0.15 * torch.tanh(self.v_to_z(v_feat)) if self.split_obs_latent else self.v_to_z(v_feat)
        z_next = z @ A.T + fb
        if eps is not None:
            sig = self.sigma_eta() if sigma is None else sigma
            z_next = z_next + eps * sig
        return z_next

    def step_artifact(
        self,
        z_art: torch.Tensor,
        A_art: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Strictly autonomous slow artifact evolution with zero voltage/stimulus dependence."""
        A = self.artifact_transition_matrix() if A_art is None else A_art
        z_next = z_art @ A.T
        if eps is not None:
            sig = self.sigma_art() if sigma is None else sigma
            z_next = z_next + eps * sig
        return z_next

    def encode_traj_posterior(
        self,
        y_post: torch.Tensor,
        mask_post: Optional[torch.Tensor],
        stim_post: torch.Tensor,
        allow_smoother: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variational posterior q_phi(z_t | z_{t-1}, y_{<=t} [or y_{1:T} if smoother])."""
        if not self.variational_traj:
            raise RuntimeError("encode_traj_posterior called when variational_traj=False")
        y_safe, m, _, _ = CausalPreStimEncoder._prepare_masked_pre(y_post, mask_post)
        s_safe = torch.tanh(torch.nan_to_num(stim_post.to(dtype=y_safe.dtype, device=y_safe.device), nan=0.0))
        feat = torch.cat([y_safe * m, m, s_safe], dim=-1)
        h_in = F.silu(self.q_traj_ln(self.q_traj_in(feat)))
        h_fwd, _ = self.q_traj_gru(h_in)
        if self.use_smoother:
            if allow_smoother:
                h_bwd_rev, _ = self.q_smooth_gru(torch.flip(h_in, dims=[1]))
                h_bwd = torch.flip(h_bwd_rev, dims=[1])
            else:
                h_bwd = torch.zeros_like(h_fwd)
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h_cat = h_fwd
        raw = self.q_traj_head(h_cat)
        q_loc = 2.0 * torch.tanh(raw[..., :self.z_dim])
        q_scale = 0.02 + 0.48 * torch.sigmoid(raw[..., self.z_dim:])
        return q_loc, q_scale

    def filter_step(
        self,
        z_pred: torch.Tensor,
        y_obs_t: torch.Tensor,
        mask_t: torch.Tensor,
        y_pred_t: torch.Tensor,
    ) -> torch.Tensor:
        """Task-2 causal innovation update: strictly conditioned on t <= current."""
        m = mask_t.to(device=z_pred.device, dtype=z_pred.dtype)
        innov = torch.where(m > 0.5, y_obs_t - y_pred_t, torch.zeros_like(y_pred_t))
        return z_pred + torch.tanh(self.K_obs(innov))

    def filter_posterior(
        self,
        z_pred: torch.Tensor,
        y_obs_t: torch.Tensor,
        mask_t: torch.Tensor,
        y_pred_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian filtering posterior (loc, scale) conditioned on y_obs_t."""
        loc = self.filter_step(z_pred, y_obs_t, mask_t, y_pred_t)
        if hasattr(self, "q_filter_sigma_log"):
            sig = torch.clamp(torch.exp(self.q_filter_sigma_log), min=1e-3, max=0.5).unsqueeze(0).expand_as(loc)
        else:
            sig = self.sigma_eta().unsqueeze(0).expand_as(loc)
        return loc, sig


class NematodeForStep(PyroModule):
    """Single-step continuous-time connectome + optional BNN residual current."""

    def __init__(
        self,
        neuronSize: int,
        NeuronList: Optional[dict],
        synapseList: dict,
        use_extrasynaptic: bool = False,
        use_bnn_residual: bool = False,
        bnn_hidden_dim: int = 16,
        per_edge_release: bool = False,
        per_neuron_stim_gain: bool = False,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
        dimensionless: bool = False,
        leak_gauge: str = "unit",
        split_extrasynaptic: bool = False,
        use_syn_filter: bool = False,
        tau_np_prior: float = 4.0,
        tau_ma_prior: float = 1.0,
        tau_mod_prior_std: float = 0.5,
        genotype_conditioned: bool = False,
    ):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuronSize = neuronSize
        self.per_neuron_stim_gain = bool(per_neuron_stim_gain)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        self.dimensionless = bool(dimensionless)
        self.leak_gauge = str(leak_gauge)
        self.split_extrasynaptic = bool(split_extrasynaptic)
        self.use_syn_filter = bool(use_syn_filter)
        cond_scale = 2.0 if self.dimensionless else 1.0
        self.Neuron = neuronLayer(
            neuronSize, NeuronList, dimensionless=self.dimensionless, leak_gauge=self.leak_gauge
        )
        self.synapse = synapseLayer(
            synapseList,
            use_extrasynaptic=use_extrasynaptic,
            neuron_size=neuronSize,
            per_edge_release=per_edge_release,
            wide_amp_priors=wide_amp_priors,
            evoked_syn_priors=evoked_syn_priors,
            dimensionless=self.dimensionless,
            split_extrasynaptic=self.split_extrasynaptic,
            use_syn_filter=self.use_syn_filter,
            tau_np_prior=tau_np_prior,
            tau_ma_prior=tau_ma_prior,
            tau_mod_prior_std=tau_mod_prior_std,
            genotype_conditioned=genotype_conditioned,
        )
        # Optogenetic Chrimson cation-channel conductance (nS, or dimensionless kappa=g/G when dimensionless=True)
        self.stim_e_rev = 0.0
        self.stim_e_inh = -0.085
        stim_loc = math.log((2.0 if self.wide_amp_priors else 1.2) * cond_scale)
        if self.per_neuron_stim_gain:
            stim_scale = 0.6 if self.wide_amp_priors else 0.3
            self.stim_gain_vec = _sample_lognormal(
                self, stim_loc, stim_scale, shape=(neuronSize,), to_event=1
            )
        else:
            stim_scale = 0.6 if self.wide_amp_priors else 0.5
            self.stim_gain = _sample_lognormal(self, stim_loc, stim_scale)
        self.bnn_residual = (
            BayesianNNResidual(neuronSize, hidden_dim=bnn_hidden_dim) if use_bnn_residual else None
        )

    def forward(
        self,
        Prev: torch.Tensor,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        dt: float = timeStep,
        sensory_states: Optional[Dict[str, object]] = None,
        syn_cache: Optional[dict] = None,
        BackgroundCurrent: Optional[torch.Tensor] = None,
        neuron_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        stim_gain_cached: Optional[torch.Tensor] = None,
        syn_states: Optional[Dict[str, torch.Tensor]] = None,
        genotype_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        drive_cur, syn_cond = self.synapse.forward_terms(
            Prev, syn_cache=syn_cache, syn_states=syn_states, dt=dt, genotype_id=genotype_id
        )
        if self.bnn_residual is not None:
            drive_cur = drive_cur + self.bnn_residual(Prev)
        if ExternalInput is None:
            ext_current = torch.zeros_like(Prev)
            total_extra_cond = syn_cond
        else:
            gain = (
                stim_gain_cached
                if stim_gain_cached is not None
                else (self.stim_gain_vec if self.per_neuron_stim_gain else self.stim_gain)
            )
            g_opt_pos = gain * torch.clamp(ExternalInput, min=0.0)
            g_opt_neg = gain * torch.clamp(-ExternalInput, min=0.0)
            ext_current = g_opt_pos * self.stim_e_rev + g_opt_neg * self.stim_e_inh
            total_extra_cond = syn_cond + g_opt_pos + g_opt_neg
        if BackgroundCurrent is not None:
            ext_current = ext_current + BackgroundCurrent
        ConnectomeOutput = self.Neuron(
            Prev,
            drive_cur,
            ext_current,
            dt=dt,
            sensory_states=sensory_states,
            extraConductance=total_extra_cond,
            rawExternal=ExternalInput,
            conductance_cache=neuron_cache,
        )
        if VoltageClamp is not None:
            clamp_mask = VoltageClamp != 0.0
            if torch.any(clamp_mask):
                ConnectomeOutput = torch.where(clamp_mask, VoltageClamp, ConnectomeOutput)
        return ConnectomeOutput


class RNNematode(PyroModule):
    def __init__(self, neuronSize: int, NeuronList=None, synapseList=None):
        super().__init__()
        self.neuronSize = neuronSize
        self.Neuron = PyroModule[nn.GRU](
            input_size=neuronSize, hidden_size=neuronSize, batch_first=True
        )
        self.synapse = PyroModule[nn.GRU](
            input_size=neuronSize, hidden_size=neuronSize, batch_first=True
        )

    def forward(
        self,
        Prev: torch.Tensor,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        dt: float = timeStep,
        sensory_states=None,
        syn_cache=None,
        BackgroundCurrent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze = Prev.dim() == 1
        x = Prev.view(1, 1, -1) if squeeze else Prev.unsqueeze(1)
        if ExternalInput is not None:
            ext = ExternalInput.view(1, 1, -1) if squeeze else ExternalInput.unsqueeze(1)
            x = x + ext
        cur, _ = self.synapse(x)
        out, _ = self.Neuron(cur)
        out = out[:, -1, :]
        if squeeze:
            out = out.squeeze(0)
        if VoltageClamp is not None:
            clamp_mask = VoltageClamp != 0.0
            if torch.any(clamp_mask):
                out = torch.where(clamp_mask, VoltageClamp, out)
        return out


def _advance_stim_actuator(
    stim_act: torch.Tensor,
    ext_t: torch.Tensor,
    dt_sim: float,
    tau_rise: float,
    tau_decay: float,
    decay_rise: Optional[float] = None,
    decay_decay: Optional[float] = None,
) -> torch.Tensor:
    """Exact first-order rise/decay update for the optogenetic actuator.

    The input is held constant over one simulation substep.  Comparing the
    current state with the target avoids overshoot even when ``dt_sim``
    exceeds the rise time, and keeps negative (inhibitory) commands instead of
    clamping them to zero.  For a positive step from zero this is
    ``1 - exp(-dt_sim / tau_rise)``; for a decay toward zero it is
    ``stim_act * exp(-dt_sim / tau_decay)``.
    """
    if decay_rise is None:
        decay_rise = math.exp(-float(dt_sim) / max(float(tau_rise), 1e-6))
    if decay_decay is None:
        decay_decay = math.exp(-float(dt_sim) / max(float(tau_decay), 1e-6))
    decay = torch.where(
        ext_t > stim_act,
        stim_act.new_full((), decay_rise),
        stim_act.new_full((), decay_decay),
    )
    return ext_t + (stim_act - ext_t) * decay


class RecurrentNematode(PyroModule):
    """Multi-step biophysical connectome ODE + Ca2+ + GCaMP generative model for Pyro SVI/MCMC."""

    def __init__(
        self,
        model: NematodeForStep,
        dt_obs: float = 0.5,
        substeps: int = 10,
        likelihood: str = "student_t",
        student_t_df: float = 4.0,
        scale: Optional[int] = None,
        use_prestim_encoder: bool = False,
        pre_vols: int = 10,
        prestim_rank: int = 16,
        variational_prestim: bool = False,
        prestim_oscillatory: bool = False,
        per_neuron_kinetics: bool = False,
        per_edge_release: bool = False,
        stim_actuator: bool = False,
        stim_tau_rise: float = 0.08,
        stim_tau_decay: float = 0.4,
        use_latent_ssm: bool = False,
        latent_ssm_dim: int = 16,
        ssm_encoder: str = "pooling",
        ssm_split_latent: bool = False,
        ssm_artifact_dim: int = 8,
        ssm_variational_traj: bool = False,
        ssm_smoother: bool = False,
        history_vols: int = 0,
    ):
        super().__init__()
        self.model = model
        self.per_edge_release = bool(per_edge_release)
        self.stim_actuator = bool(stim_actuator)
        self.stim_tau_rise = float(stim_tau_rise)
        self.stim_tau_decay = float(stim_tau_decay)
        # Keep backward compatibility with callers passing `scale=` while never mutating globals
        self.substeps = int(scale if scale is not None else substeps)
        self.dt_obs = float(dt_obs)
        self.likelihood = likelihood
        self.student_t_df = float(student_t_df)
        self.use_prestim_encoder = bool(use_prestim_encoder)
        self.pre_vols = int(pre_vols)
        self.prestim_rank = int(prestim_rank)
        self.variational_prestim = bool(variational_prestim)
        self.prestim_oscillatory = bool(prestim_oscillatory)
        self.use_latent_ssm = bool(use_latent_ssm)
        self.latent_ssm_dim = int(latent_ssm_dim)
        self.ssm_encoder = str(ssm_encoder)
        self.ssm_split_latent = bool(ssm_split_latent)
        self.ssm_artifact_dim = int(ssm_artifact_dim)
        self.ssm_variational_traj = bool(ssm_variational_traj)
        self.ssm_smoother = bool(ssm_smoother)
        self.history_vols = int(history_vols)

        self.register_buffer("_dev_ref", torch.tensor(0.0))
        neuron_size = getattr(model, "neuronSize", 300)
        self.per_neuron_kinetics = bool(per_neuron_kinetics)
        self.gcamp = CalciumGCaMPLayer(neuron_size, per_neuron_kinetics=self.per_neuron_kinetics)
        self.prestim_encoder = (
            CausalPreStimEncoder(neuron_size, rank=self.prestim_rank, oscillatory=self.prestim_oscillatory)
            if self.use_prestim_encoder
            else None
        )
        self.latent_ssm = (
            LatentNeuralSSM(
                neuron_size,
                z_dim=self.latent_ssm_dim,
                encoder_type=self.ssm_encoder,
                split_obs_latent=self.ssm_split_latent,
                artifact_dim=self.ssm_artifact_dim,
                variational_traj=self.ssm_variational_traj,
                use_smoother=self.ssm_smoother,
            )
            if self.use_latent_ssm
            else None
        )
        # Dimensionless observation noise multiplier (when noise_scale is provided) or base std
        self.obs_sigma = _sample_lognormal(self, math.log(0.15), 0.5)

    def _rollout(
        self,
        stim_input: torch.Tensor,
        voltage_clamp: Optional[torch.Tensor],
        dt_obs: float,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
        drift_gate: Optional[torch.Tensor] = None,
        prestim_z: Optional[torch.Tensor] = None,
        return_trajectories: bool = True,
        use_step_cache: bool = True,
        ssm_eps: Optional[torch.Tensor] = None,
        genotype_id: Optional[torch.Tensor] = None,
        ssm_z0_sample: Optional[torch.Tensor] = None,
        ssm_a0_sample: Optional[torch.Tensor] = None,
        traj_batch_plate=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, n_t, n_neurons = stim_input.shape
        dt_sim = dt_obs / max(1, self.substeps)

        if hasattr(self.model, "Neuron") and hasattr(self.model.Neuron, "conductance"):
            e_rest = self.model.Neuron.conductance.E.unsqueeze(0)
            neuron_cache = self.model.Neuron.conductance.build_step_cache(dt=dt_sim) if use_step_cache else None
            has_sensory = (not use_step_cache) or len(getattr(self.model.Neuron, "_neuron_List", {})) > 0
        else:
            e_rest = torch.full((1, n_neurons), -0.065, device=stim_input.device, dtype=stim_input.dtype)
            neuron_cache = None
            has_sensory = False

        stim_gain_cached = None
        if use_step_cache and (hasattr(self.model, "stim_gain_vec") or hasattr(self.model, "stim_gain")):
            stim_gain_cached = self.model.stim_gain_vec if getattr(self.model, "per_neuron_stim_gain", False) else self.model.stim_gain

        sensory_states: Dict[str, object] = {}
        syn_cache = (
            self.model.synapse.build_step_cache(n_neurons, stim_input.device, stim_input.dtype, dt=dt_sim)
            if (use_step_cache and hasattr(self.model, "synapse") and hasattr(self.model.synapse, "build_step_cache"))
            else None
        )
        has_geno = (
            genotype_id is not None
            and hasattr(self.model, "synapse")
            and getattr(self.model.synapse, "genotype_conditioned", False)
        )
        # Settle coupled network resting equilibrium once before anchoring F0 baseline.
        settle_steps = max(8, self.substeps * 4)
        v_settle = e_rest.expand(batch_size, n_neurons) if (has_sensory or has_geno) else e_rest
        step_kwargs = {"ExternalInput": None, "VoltageClamp": None, "dt": dt_sim, "sensory_states": sensory_states, "syn_cache": syn_cache}
        if has_geno:
            step_kwargs["genotype_id"] = genotype_id
        if neuron_cache is not None:
            step_kwargs["neuron_cache"] = neuron_cache
            step_kwargs["stim_gain_cached"] = stim_gain_cached
        for _ in range(settle_steps):
            v_settle = self.model(v_settle, **step_kwargs)
        v = v_settle if (has_sensory or has_geno) else v_settle.expand(batch_size, n_neurons)
        syn_states = (
            self.model.synapse.init_syn_states(v, syn_cache=syn_cache)
            if (hasattr(self.model, "synapse") and hasattr(self.model.synapse, "init_syn_states"))
            else None
        )

        gcamp_cache = self.gcamp.build_step_cache(dt=dt_sim) if use_step_cache else None
        v_th_ca = gcamp_cache[0] if gcamp_cache is not None else self.gcamp.v_th_ca
        ca, b, b_baseline = self.gcamp.resting_state(v, v_th_ca=v_th_ca)
        gcamp_gain = gcamp_cache[7] if gcamp_cache is not None else self.gcamp.gain
        gcamp_offset = gcamp_cache[8] if gcamp_cache is not None else self.gcamp.offset

        prestim_state = None
        inject_t = max(0, min(self.pre_vols, n_t))
        if self.prestim_encoder is not None and pre_stim_obs is not None and inject_t > 0:
            # Structural causality guarantee: never read any frame at t >= pre_vols
            y_pre_slice = pre_stim_obs[:, :inject_t, :]
            m_pre_slice = None if pre_stim_mask is None else pre_stim_mask[:, :inject_t, :]
            prestim_state = self.prestim_encoder(y_pre_slice, m_pre_slice, z_bg=prestim_z)
            if drift_gate is not None:
                g = drift_gate.to(device=stim_input.device, dtype=stim_input.dtype)
                values = list(prestim_state)
                if g.dim() == 3:
                    for i in range(5):
                        values[i] = values[i] * g[i]
                    if len(values) > 8:
                        values[8] = values[8] * g[4]
                else:
                    if g.dim() == 1:
                        g = g.unsqueeze(0)
                    if g.shape[0] == 1 and batch_size > 1:
                        g = g.expand(batch_size, -1)
                    for i in range(5):
                        values[i] = values[i] * g
                    if len(values) > 8:
                        values[8] = values[8] * g
                prestim_state = tuple(values)

        y_preds = []
        v_traj = [] if return_trajectories else None
        ca_traj = [] if return_trajectories else None
        stim_act = torch.zeros_like(v)
        decay_rise = math.exp(-dt_sim / max(self.stim_tau_rise, 1e-6))
        decay_decay = math.exp(-dt_sim / max(self.stim_tau_decay, 1e-6))

        bg_decay = None
        if prestim_state is not None and n_t > inject_t:
            i_bg0 = prestim_state[2]
            tau_bg = prestim_state[5]
            sub_idx = torch.arange(self.substeps, device=stim_input.device, dtype=stim_input.dtype)
            rel_t_obs = torch.arange(n_t - inject_t, device=stim_input.device, dtype=stim_input.dtype).unsqueeze(1) * dt_obs
            rel_t_sim = rel_t_obs + (sub_idx.unsqueeze(0) + 0.5) * dt_sim
            bg_decay = torch.exp(-rel_t_sim / tau_bg)  # (n_post, substeps)

        ssm_z = None
        ssm_art = None
        ssm_dv0 = None
        ssm_dca0 = None
        ssm_A = None
        ssm_sigma = None
        ssm_A_art = None
        ssm_sig_art = None
        if self.latent_ssm is not None:
            ssm_A = self.latent_ssm.transition_matrix()
            ssm_sigma = self.latent_ssm.sigma_eta()
            z_dim = self.latent_ssm.z_dim
            total_z_dim = self.latent_ssm.total_z_dim
            if ssm_eps is None:
                ssm_eps = torch.randn(
                    (batch_size, n_t + 1, total_z_dim),
                    device=stim_input.device,
                    dtype=stim_input.dtype,
                )
            has_pre_data = pre_stim_obs is not None and inject_t > 0
            if has_pre_data:
                hist_k = inject_t if self.history_vols <= 0 else max(1, min(inject_t, int(self.history_vols)))
                h_start = inject_t - hist_k
                y_pre_ssm = pre_stim_obs[:, h_start:inject_t, :]
                m_pre_ssm = None if pre_stim_mask is None else pre_stim_mask[:, h_start:inject_t, :]
                z0_loc, z0_sc, ssm_dv0, ssm_dca0 = self.latent_ssm.encode_initial(
                    y_pre_ssm, m_pre_ssm, dt_obs=dt_obs
                )
            if ssm_z0_sample is not None:
                ssm_z = ssm_z0_sample
            elif has_pre_data:
                ssm_z = z0_loc[:, :z_dim] + z0_sc[:, :z_dim] * ssm_eps[:, 0, :z_dim]
            else:
                ssm_z = torch.zeros(
                    (batch_size, z_dim),
                    device=stim_input.device,
                    dtype=stim_input.dtype,
                )
            if self.latent_ssm.split_obs_latent:
                ssm_A_art = self.latent_ssm.artifact_transition_matrix()
                ssm_sig_art = self.latent_ssm.sigma_art()
                if ssm_a0_sample is not None:
                    ssm_art = ssm_a0_sample
                elif has_pre_data:
                    ssm_art = z0_loc[:, z_dim:] + z0_sc[:, z_dim:] * ssm_eps[:, 0, z_dim:]
                else:
                    ssm_art = torch.zeros(
                        (batch_size, self.latent_ssm.artifact_dim),
                        device=stim_input.device,
                        dtype=stim_input.dtype,
                    )

        call_cache = {"neuron_cache": neuron_cache, "stim_gain_cached": stim_gain_cached} if neuron_cache is not None else {}
        if has_geno:
            call_cache["genotype_id"] = genotype_id

        for t in range(n_t):
            if t == inject_t:
                if prestim_state is not None:
                    delta_v0, delta_ca0 = prestim_state[0], prestim_state[1]
                    v = v + delta_v0
                    ca = torch.clamp(ca + delta_ca0, min=1e-4)
                if ssm_dv0 is not None:
                    v = v + ssm_dv0
                    ca = torch.clamp(ca + ssm_dca0, min=1e-4)

            ssm_cur = None
            if ssm_z is not None and t >= inject_t:
                if traj_batch_plate is not None and self.latent_ssm.variational_traj:
                    p_loc = self.latent_ssm.step_latent(
                        ssm_z, v, e_rest, A_mat=ssm_A, eps=None
                    )
                    with traj_batch_plate:
                        ssm_z = pyro.sample(
                            f"ssm_z_t{t - inject_t}",
                            dist.Normal(p_loc, ssm_sigma).to_event(1),
                        )
                else:
                    ssm_z = self.latent_ssm.step_latent(
                        ssm_z, v, e_rest, A_mat=ssm_A, eps=ssm_eps[:, t + 1, :self.latent_ssm.z_dim], sigma=ssm_sigma
                    )
                ssm_cur = self.latent_ssm.latent_current(ssm_z)
                if ssm_art is not None:
                    ssm_art = self.latent_ssm.step_artifact(
                        ssm_art, A_art=ssm_A_art, eps=ssm_eps[:, t + 1, self.latent_ssm.z_dim:], sigma=ssm_sig_art
                    )

            ext_t = stim_input[:, t, :]
            clamp_t = None if voltage_clamp is None else voltage_clamp[:, t, :]
            for sub in range(self.substeps):
                bg_cur = None
                if bg_decay is not None and t >= inject_t:
                    bg_cur = prestim_state[2] * bg_decay[t - inject_t, sub]
                if ssm_cur is not None:
                    bg_cur = ssm_cur if bg_cur is None else (bg_cur + ssm_cur)
                if self.stim_actuator:
                    stim_act = _advance_stim_actuator(
                        stim_act, ext_t, dt_sim, self.stim_tau_rise, self.stim_tau_decay,
                        decay_rise=decay_rise, decay_decay=decay_decay,
                    )
                    ext_eff = stim_act
                else:
                    ext_eff = ext_t
                v = self.model(
                    v,
                    ExternalInput=ext_eff,
                    VoltageClamp=clamp_t,
                    dt=dt_sim,
                    sensory_states=sensory_states,
                    syn_cache=syn_cache,
                    BackgroundCurrent=bg_cur,
                    syn_states=syn_states,
                    **call_cache,
                )
                ca, b = self.gcamp.step(v, ca, b, dt=dt_sim, v_th_ca=v_th_ca, cached=gcamp_cache)
            y_t = gcamp_gain * (b - b_baseline) + gcamp_offset
            if ssm_art is not None and t >= inject_t:
                y_t = y_t + self.latent_ssm.obs_residual(ssm_art)
            elif ssm_z is not None and t >= inject_t and not self.latent_ssm.split_obs_latent:
                y_t = y_t + self.latent_ssm.obs_residual(ssm_z)
            y_preds.append(y_t)
            if return_trajectories:
                v_traj.append(v)
                ca_traj.append(ca)

        y_pred = torch.stack(y_preds, dim=1)
        if prestim_state is not None and n_t > inject_t:
            post_drift = self.prestim_encoder.compute_post_drift(
                prestim_state, n_t - inject_t, dt_obs, drift_gate=drift_gate
            )
            y_pred = torch.cat([y_pred[:, :inject_t, :], y_pred[:, inject_t:, :] + post_drift], dim=1)

        if not return_trajectories:
            return y_pred, None, None
        return y_pred, torch.stack(v_traj, dim=1), torch.stack(ca_traj, dim=1)

    def simulate(
        self,
        stim_input: torch.Tensor,
        voltage_clamp: Optional[torch.Tensor] = None,
        dt_obs: Optional[float] = None,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
        drift_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward rollout and return (y_gcamp_pred, V_traj, Ca_traj)."""
        squeezed = stim_input.dim() == 2
        if squeezed:
            stim_input = stim_input.unsqueeze(0)
            if voltage_clamp is not None:
                voltage_clamp = voltage_clamp.unsqueeze(0)
            if pre_stim_obs is not None and pre_stim_obs.dim() == 2:
                pre_stim_obs = pre_stim_obs.unsqueeze(0)
            if pre_stim_mask is not None and pre_stim_mask.dim() == 2:
                pre_stim_mask = pre_stim_mask.unsqueeze(0)
            if drift_gate is not None and drift_gate.dim() == 2:
                drift_gate = drift_gate.unsqueeze(0)
        y_pred, v_traj, ca_traj = self._rollout(
            stim_input,
            voltage_clamp,
            self.dt_obs if dt_obs is None else float(dt_obs),
            pre_stim_obs=pre_stim_obs,
            pre_stim_mask=pre_stim_mask,
            drift_gate=drift_gate,
        )
        if squeezed:
            return y_pred.squeeze(0), v_traj.squeeze(0), ca_traj.squeeze(0)
        return y_pred, v_traj, ca_traj

    def forward(
        self,
        ExternalInput: Optional[torch.Tensor] = None,
        VoltageClamp: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        noise_scale: Optional[torch.Tensor] = None,
        dt_obs: Optional[float] = None,
        dataset_size: Optional[int] = None,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
        drift_gate: Optional[torch.Tensor] = None,
        sample_obs: bool = True,
        use_step_cache: bool = True,
        ssm_eps: Optional[torch.Tensor] = None,
        genotype_id: Optional[torch.Tensor] = None,
        filter_online: bool = False,
    ) -> torch.Tensor:
        if filter_online:
            if ExternalInput is None or y is None:
                raise ValueError("filter_online=True requires ExternalInput and y.")
            m_eff = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
            return self.rollout_online_filter(
                ExternalInput,
                y,
                m_eff,
                dt_obs=dt_obs,
                pre_stim_obs=pre_stim_obs,
                pre_stim_mask=pre_stim_mask,
            )
        if ExternalInput is None and VoltageClamp is None:
            if y is None:
                raise ValueError("Provide at least one of ExternalInput, VoltageClamp, or y.")
            ExternalInput = torch.zeros_like(y)

        ref = ExternalInput if ExternalInput is not None else VoltageClamp
        squeezed = ref.dim() == 2
        if squeezed:
            ext = None if ExternalInput is None else ExternalInput.unsqueeze(0)
            v_clamp = None if VoltageClamp is None else VoltageClamp.unsqueeze(0)
            y_obs = None if y is None else y.unsqueeze(0)
            obs_mask = None if mask is None else mask.unsqueeze(0)
            n_scale = None if noise_scale is None else noise_scale.unsqueeze(0)
            pre_obs = None if pre_stim_obs is None else (pre_stim_obs.unsqueeze(0) if pre_stim_obs.dim() == 2 else pre_stim_obs)
            pre_msk = None if pre_stim_mask is None else (pre_stim_mask.unsqueeze(0) if pre_stim_mask.dim() == 2 else pre_stim_mask)
            d_gate = None if drift_gate is None else (drift_gate.unsqueeze(0) if drift_gate.dim() == 2 else drift_gate)
        else:
            ext = ExternalInput
            v_clamp = VoltageClamp
            y_obs = y
            obs_mask = mask
            n_scale = noise_scale
            pre_obs = pre_stim_obs
            pre_msk = pre_stim_mask
            d_gate = drift_gate

        if ext is None:
            ext = torch.zeros_like(v_clamp)

        if (self.prestim_encoder is not None or self.latent_ssm is not None) and pre_obs is None and y_obs is not None:
            t_cut = min(self.pre_vols, y_obs.shape[1])
            pre_obs = y_obs[:, :t_cut, :]
            pre_msk = None if obs_mask is None else obs_mask[:, :t_cut, :]

        batch_size = ext.shape[0]
        if dataset_size is not None and int(dataset_size) > batch_size:
            batch_plate = pyro.plate("batch", size=int(dataset_size), subsample_size=batch_size)
        else:
            batch_plate = pyro.plate("batch", batch_size)

        # Sample per-trial/worm latent state inside `batch_plate` BEFORE `_rollout` so
        # global connectome/GCaMP parameters inside `_rollout` remain outside the plate.
        prestim_z = None
        ssm_z0_sample = None
        ssm_a0_sample = None
        if self.prestim_encoder is not None and self.variational_prestim:
            with batch_plate:
                z_zeros = torch.zeros(batch_size, self.prestim_rank, device=ext.device, dtype=ext.dtype)
                prestim_z = pyro.sample(
                    "prestim_z",
                    dist.Normal(z_zeros, torch.ones_like(z_zeros)).to_event(1),
                )
        if self.latent_ssm is not None and self.latent_ssm.variational_traj:
            with batch_plate:
                zz = torch.zeros(batch_size, self.latent_ssm.z_dim, device=ext.device, dtype=ext.dtype)
                ssm_z0_sample = pyro.sample("ssm_z0", dist.Normal(zz, torch.ones_like(zz)).to_event(1))
                if self.latent_ssm.split_obs_latent:
                    az = torch.zeros(batch_size, self.latent_ssm.artifact_dim, device=ext.device, dtype=ext.dtype)
                    ssm_a0_sample = pyro.sample("ssm_a0", dist.Normal(az, torch.ones_like(az)).to_event(1))

        y_pred, _, _ = self._rollout(
            ext,
            v_clamp,
            self.dt_obs if dt_obs is None else float(dt_obs),
            pre_stim_obs=pre_obs,
            pre_stim_mask=pre_msk,
            drift_gate=d_gate,
            prestim_z=prestim_z,
            return_trajectories=False,
            use_step_cache=use_step_cache,
            ssm_eps=ssm_eps,
            genotype_id=genotype_id,
            ssm_z0_sample=ssm_z0_sample,
            ssm_a0_sample=ssm_a0_sample,
            traj_batch_plate=(batch_plate if y_obs is not None else None),
        )
        if not sample_obs and y_obs is None:
            return y_pred.squeeze(0) if squeezed else y_pred

        obs_sigma = torch.clamp(self.obs_sigma, min=1e-3)
        n_t = y_pred.shape[1]

        if n_scale is not None:
            # When empirical per-neuron noise_scale (B, N) or (B, T, N) is passed,
            # obs_sigma acts as a global calibration floor.
            scale_t = torch.sqrt((obs_sigma * 0.5) ** 2 + n_scale ** 2)
        else:
            scale_t = obs_sigma

        with batch_plate:
            for t in range(n_t):
                loc_t = y_pred[:, t, :]
                if isinstance(scale_t, torch.Tensor) and scale_t.dim() == 3:
                    s_t = scale_t[:, t, :]
                else:
                    s_t = scale_t
                if self.likelihood == "student_t":
                    base_dist = dist.StudentT(self.student_t_df, loc_t, s_t)
                else:
                    base_dist = dist.Normal(loc_t, s_t)

                if obs_mask is not None:
                    base_dist = base_dist.mask(obs_mask[:, t, :])
                obs_t = None if y_obs is None else y_obs[:, t, :]
                pyro.sample(f"obs_{t}", base_dist.to_event(1), obs=obs_t)

        return y_pred.squeeze(0) if squeezed else y_pred

    def assign(self, neuron_loc, neuron_scale, wicks_loc, wicks_scale, gap_loc, gap_scale):
        self.model.Neuron.conductance.E = PyroSample(
            dist.Normal(neuron_loc[0], torch.clamp(neuron_scale[0], min=1e-4)).to_event(1)
        )
        if not getattr(self.model.Neuron.conductance, "dimensionless", False):
            self.model.Neuron.conductance.G = PyroSample(
                dist.LogNormal(neuron_loc[1], torch.clamp(neuron_scale[1], min=1e-4)).to_event(1)
            )
        self.model.Neuron.conductance.C = PyroSample(
            dist.LogNormal(neuron_loc[2], torch.clamp(neuron_scale[2], min=1e-4)).to_event(1)
        )

        prev_wicks = self.model.synapse.wicks
        e_rev_loc = getattr(prev_wicks, "e_rev_loc_buf", None)
        e_rev_scale = getattr(prev_wicks, "e_rev_scale_buf", None)
        self.model.synapse.wicks = WicksSynapse(
            self.model.synapse.Wicks_SRC,
            self.model.synapse.Wicks_DST,
            self.model.synapse.Wicks_Weight,
            e_rev_loc=e_rev_loc,
            e_rev_scale=e_rev_scale,
            g_max=PyroSample(dist.LogNormal(wicks_loc[0], torch.clamp(wicks_scale[0], min=1e-4)).to_event(1)),
            V_rest=PyroSample(dist.Normal(wicks_loc[1], torch.clamp(wicks_scale[1], min=1e-4)).to_event(1)),
            V_slope=PyroSample(dist.LogNormal(wicks_loc[2], torch.clamp(wicks_scale[2], min=1e-4)).to_event(1)),
            neuron_size=getattr(self.model.synapse, "neuron_size", None),
            per_edge_release=getattr(prev_wicks, "per_edge_release", False),
        )
        self.model.synapse.general = GeneralSynapse(
            self.model.synapse.Gap_Junction_SRC,
            self.model.synapse.Gap_Junction_DST,
            self.model.synapse.Gap_Junction_Weight,
            g_syn=PyroSample(dist.LogNormal(gap_loc, torch.clamp(gap_scale, min=1e-4)).to_event(1)),
        )

    def rollout_online_filter(
        self,
        stim_input: torch.Tensor,
        y_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        dt_obs: Optional[float] = None,
        pre_stim_obs: Optional[torch.Tensor] = None,
        pre_stim_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Task-2 strictly causal online filtering: predicts y_t before assimilating y_obs_t."""
        if self.latent_ssm is None:
            raise RuntimeError("rollout_online_filter requires latent_ssm != None")
        squeezed = stim_input.dim() == 2
        if squeezed:
            stim_input = stim_input.unsqueeze(0)
            y_obs = y_obs.unsqueeze(0)
            obs_mask = obs_mask.unsqueeze(0)
        batch_size, n_t, n_neurons = stim_input.shape
        dt_obs_f = self.dt_obs if dt_obs is None else float(dt_obs)
        dt_sim = dt_obs_f / max(1, self.substeps)

        e_rest = self.model.Neuron.conductance.E.unsqueeze(0)
        neuron_cache = self.model.Neuron.conductance.build_step_cache(dt=dt_sim)
        syn_cache = self.model.synapse.build_step_cache(n_neurons, stim_input.device, stim_input.dtype, dt=dt_sim)
        stim_gain_cached = (
            self.model.stim_gain_vec if getattr(self.model, "per_neuron_stim_gain", False) else self.model.stim_gain
        )
        sensory_states: Dict[str, object] = {}
        v_settle = e_rest
        for _ in range(max(8, self.substeps * 4)):
            v_settle = self.model(
                v_settle, ExternalInput=None, dt=dt_sim, sensory_states=sensory_states,
                syn_cache=syn_cache, neuron_cache=neuron_cache, stim_gain_cached=stim_gain_cached,
            )
        v = v_settle.expand(batch_size, n_neurons)
        syn_states = self.model.synapse.init_syn_states(v, syn_cache=syn_cache)
        gcamp_cache = self.gcamp.build_step_cache(dt=dt_sim)
        v_th_ca = gcamp_cache[0]
        ca, b, b_baseline = self.gcamp.resting_state(v, v_th_ca=v_th_ca)
        gcamp_gain, gcamp_offset = gcamp_cache[7], gcamp_cache[8]

        inject_t = max(0, min(self.pre_vols, n_t))
        pre_y = pre_stim_obs if pre_stim_obs is not None else y_obs
        pre_m = pre_stim_mask if pre_stim_mask is not None else obs_mask
        ssm_A = self.latent_ssm.transition_matrix()
        ssm_A_art = self.latent_ssm.artifact_transition_matrix() if self.latent_ssm.split_obs_latent else None
        z_dim = self.latent_ssm.z_dim
        if inject_t > 0:
            hist_k = inject_t if self.history_vols <= 0 else max(1, min(inject_t, int(self.history_vols)))
            z0_loc, _, ssm_dv0, ssm_dca0 = self.latent_ssm.encode_initial(
                pre_y[:, inject_t - hist_k:inject_t, :],
                None if pre_m is None else pre_m[:, inject_t - hist_k:inject_t, :],
                dt_obs=dt_obs_f,
            )
            ssm_z = z0_loc[:, :z_dim]
            ssm_art = z0_loc[:, z_dim:] if self.latent_ssm.split_obs_latent else None
        else:
            ssm_z = torch.zeros((batch_size, z_dim), device=stim_input.device, dtype=stim_input.dtype)
            ssm_art = torch.zeros((batch_size, self.latent_ssm.artifact_dim), device=stim_input.device, dtype=stim_input.dtype) if self.latent_ssm.split_obs_latent else None
            ssm_dv0, ssm_dca0 = None, None

        stim_act = torch.zeros_like(v)
        decay_rise = math.exp(-dt_sim / max(self.stim_tau_rise, 1e-6))
        decay_decay = math.exp(-dt_sim / max(self.stim_tau_decay, 1e-6))
        preds = []
        for t in range(n_t):
            if t == inject_t and ssm_dv0 is not None:
                v = v + ssm_dv0
                ca = torch.clamp(ca + ssm_dca0, min=1e-4)
            ssm_cur = None
            if t >= inject_t:
                ssm_z = self.latent_ssm.step_latent(ssm_z, v, e_rest, A_mat=ssm_A, eps=None)
                ssm_cur = self.latent_ssm.latent_current(ssm_z)
                if ssm_art is not None:
                    ssm_art = self.latent_ssm.step_artifact(ssm_art, A_art=ssm_A_art, eps=None)
            ext_t = stim_input[:, t, :]
            for _ in range(self.substeps):
                if self.stim_actuator:
                    stim_act = _advance_stim_actuator(
                        stim_act, ext_t, dt_sim, self.stim_tau_rise, self.stim_tau_decay,
                        decay_rise=decay_rise, decay_decay=decay_decay,
                    )
                    ext_eff = stim_act
                else:
                    ext_eff = ext_t
                v = self.model(
                    v, ExternalInput=ext_eff, dt=dt_sim, sensory_states=sensory_states,
                    syn_cache=syn_cache, BackgroundCurrent=ssm_cur, syn_states=syn_states,
                    neuron_cache=neuron_cache, stim_gain_cached=stim_gain_cached,
                )
                ca, b = self.gcamp.step(v, ca, b, dt=dt_sim, v_th_ca=v_th_ca, cached=gcamp_cache)
            y_t = gcamp_gain * (b - b_baseline) + gcamp_offset
            if ssm_art is not None and t >= inject_t:
                y_t = y_t + self.latent_ssm.obs_residual(ssm_art)
            elif t >= inject_t and not self.latent_ssm.split_obs_latent:
                y_t = y_t + self.latent_ssm.obs_residual(ssm_z)
            preds.append(y_t)
            # Assimilate current frame y_obs[:, t, :] ONLY AFTER producing causal prediction y_t
            if t >= inject_t:
                ssm_z = self.latent_ssm.filter_step(ssm_z, y_obs[:, t, :], obs_mask[:, t, :], y_t)
        out = torch.stack(preds, dim=1)
        return out.squeeze(0) if squeezed else out


def check_closed_loop_stability(
    rec_model: RecurrentNematode,
    n_steps: int = 240,
    max_v_dev: float = 0.05,
    max_z_norm: float = 20.0,
) -> Dict[str, float]:
    """Verify long zero-stimulus rollout boundedness and closed-loop Jacobian spectral radius."""
    if rec_model.latent_ssm is None:
        raise ValueError("check_closed_loop_stability requires rec_model.latent_ssm != None")
    ssm = rec_model.latent_ssm
    dev = next(rec_model.parameters()).device
    dt_obs = float(rec_model.dt_obs)
    dt_sim = dt_obs / max(1, int(rec_model.substeps))
    n_neu = int(rec_model.model.neuronSize)
    z_dim = int(ssm.z_dim)

    with torch.no_grad():
        e_rest = rec_model.model.Neuron.conductance.E.unsqueeze(0).to(dev)
        neuron_cache = rec_model.model.Neuron.conductance.build_step_cache(dt=dt_sim)
        syn_cache = rec_model.model.synapse.build_step_cache(n_neu, dev, e_rest.dtype, dt=dt_sim)
        v_eq = e_rest.clone()
        for _ in range(max(8, rec_model.substeps * 4)):
            v_eq = rec_model.model(v_eq, dt=dt_sim, syn_cache=syn_cache, neuron_cache=neuron_cache)
        syn_states = rec_model.model.synapse.init_syn_states(v_eq, syn_cache=syn_cache)

        A = ssm.transition_matrix()
        a_rho = float(torch.linalg.eigvals(A).abs().max().item())

        v = v_eq.clone()
        z = torch.zeros((1, z_dim), device=dev, dtype=v_eq.dtype)
        max_dv = 0.0
        max_zn = 0.0
        mid_dv = 0.0
        end_dv = 0.0
        for step_i in range(int(n_steps)):
            z = ssm.step_latent(z, v, e_rest, A_mat=A, eps=None)
            cur = ssm.latent_current(z)
            for _ in range(rec_model.substeps):
                v = rec_model.model(
                    v, dt=dt_sim, syn_cache=syn_cache, BackgroundCurrent=cur,
                    syn_states=syn_states, neuron_cache=neuron_cache,
                )
            dv = float((v - v_eq).abs().max().item())
            zn = float(z.norm().item())
            max_dv = max(max_dv, dv)
            max_zn = max(max_zn, zn)
            if step_i == int(n_steps) // 2:
                mid_dv = dv
            end_dv = dv

        def _one_step(v_in: torch.Tensor, z_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            z_out = ssm.step_latent(z_in, v_in, e_rest, A_mat=A, eps=None)
            cur = ssm.latent_current(z_out)
            v_cur = v_in
            for _ in range(rec_model.substeps):
                v_cur = rec_model.model(
                    v_cur, dt=dt_sim, syn_cache=syn_cache, BackgroundCurrent=cur, neuron_cache=neuron_cache
                )
            return v_cur, z_out

    if n_neu <= 32:
        x_eq = torch.cat([v_eq.squeeze(0), torch.zeros(z_dim, device=dev, dtype=v_eq.dtype)], dim=0)

        def _jac_fn(x: torch.Tensor) -> torch.Tensor:
            v_n, z_n = _one_step(x[:n_neu].unsqueeze(0), x[n_neu:].unsqueeze(0))
            return torch.cat([v_n.squeeze(0), z_n.squeeze(0)], dim=0)

        J = torch.autograd.functional.jacobian(_jac_fn, x_eq)
        cl_rho = float(torch.linalg.eigvals(J).abs().max().item())
    else:
        with torch.no_grad():
            u_v = torch.randn_like(v_eq)
            u_z = torch.randn((1, z_dim), device=dev, dtype=v_eq.dtype)
            norm0 = math.sqrt(float((u_v * u_v).sum() + (u_z * u_z).sum()))
            u_v, u_z = u_v / norm0, u_z / norm0
            eps_fd = 1e-4
            cl_rho = 0.0
            for _ in range(25):
                vp, zp = _one_step(v_eq + eps_fd * u_v, eps_fd * u_z)
                vm, zm = _one_step(v_eq - eps_fd * u_v, -eps_fd * u_z)
                jv = (vp - vm) / (2.0 * eps_fd)
                jz = (zp - zm) / (2.0 * eps_fd)
                jn = math.sqrt(float((jv * jv).sum() + (jz * jz).sum()))
                cl_rho = jn
                if jn > 1e-12:
                    u_v, u_z = jv / jn, jz / jn
                else:
                    break

    finite = bool(math.isfinite(max_dv) and math.isfinite(max_zn) and math.isfinite(cl_rho))
    stable = bool(
        finite
        and max_dv < float(max_v_dev)
        and max_zn < float(max_z_norm)
        and a_rho < 0.99
        and cl_rho < 1.05
    )
    return {
        "stable": stable,
        "finite": finite,
        "max_abs_v_dev": float(max_dv),
        "max_z_norm": float(max_zn),
        "end_to_mid_v_ratio": float(end_dv / max(mid_dv, 1e-9)),
        "autonomous_A_spectral_radius": float(a_rho),
        "closed_loop_spectral_radius": float(cl_rho),
    }


SensoryList = {"ASHL": ASH, "ASHR": ASH}


def _normalize_neuron_name(name: str) -> str:
    s = str(name).strip()
    if len(s) == 4 and s[2] == "0":
        return s[0] + s[1] + s[3]
    return s


def _nt_to_reversal_prior(nt_str: str, wide_amp_priors: bool = False) -> Tuple[float, float]:
    """Map neurotransmitter annotation to chemical synapse reversal potential prior (Volts)."""
    nt = str(nt_str).strip()
    if "GABA" in nt:
        return -0.075, 0.015
    if "Acetylcholine" in nt:
        return (-0.010, 0.035) if wide_amp_priors else (0.0, 0.020)
    if "Glutamate" in nt:
        # C. elegans glutamate can be excitatory (NMDA/AMPA) or inhibitory (GluCl)
        return (-0.035, 0.040) if wide_amp_priors else (-0.035, 0.035)
    return -0.025, 0.035


def readConnectome(
    path: str = "./data/CElegansNeuronTables.xls",
    rnn: bool = False,
    use_sensory_models: bool = False,
    use_extrasynaptic: bool = False,
    use_bnn_residual: bool = False,
    bnn_hidden_dim: int = 16,
    per_edge_release: bool = False,
    per_neuron_stim_gain: bool = False,
    wide_amp_priors: bool = False,
    evoked_syn_priors: bool = False,
    dimensionless: bool = False,
    leak_gauge: str = "unit",
    split_extrasynaptic: bool = False,
    use_syn_filter: bool = False,
    tau_np_prior: float = 4.0,
    tau_ma_prior: float = 1.0,
    tau_mod_prior_std: float = 0.5,
    genotype_conditioned: bool = False,
    aconnectome_ids_path: str = "./pumpprobe/pumpprobe/aconnectome_ids.txt",
    edgelist_path: str = "./data/herm_full_edgelist.csv",
    np_path: str = "./data/edgelist_NP.csv",
    ma_path: str = "./data/edgelist_MA.csv",
    shuffle_seed: Optional[int] = None,
):
    """Build C. elegans connectome step model aligned to the 300 canonical neuron IDs.

    ``shuffle_seed`` applies a fixed random permutation to all edge endpoints
    (gap, chemical, and extrasynaptic) while leaving neuron identities,
    sensory masks, and stimulus targets fixed.  This is the connectome-shuffle
    control: graph degree is preserved, but the mapping from canonical neuron
    identity to local topology is destroyed.
    """
    sensory_df = pd.read_excel(path, sheet_name="Sensory")
    sensory_set = set(sensory_df["Neuron"].dropna().astype(str).map(_normalize_neuron_name))

    nt_df = pd.read_excel(path, sheet_name="Connectome")
    nt_lookup: Dict[Tuple[str, str], str] = {}
    for _, row in nt_df.iterrows():
        if str(row["Type"]).strip() == "Send":
            src_n = _normalize_neuron_name(row["Origin"])
            dst_n = _normalize_neuron_name(row["Target"])
            nt_lookup[(src_n, dst_n)] = str(row["Neurotransmitter"]).strip()

    connectome = pd.read_csv(edgelist_path)
    neuron_names = np.loadtxt(aconnectome_ids_path, dtype=str)[:, 1]
    name_to_idx = {str(n): idx for idx, n in enumerate(neuron_names)}

    neuron_list: Dict[int, PyroModule] = {}
    sensory_mask = np.zeros(len(neuron_names), dtype=np.float32)
    for idx, n_name in enumerate(neuron_names):
        if n_name in sensory_set:
            sensory_mask[idx] = 1.0
            if use_sensory_models:
                neuron_list[idx] = SensoryList[n_name]() if n_name in SensoryList else FallbackSensory()

    gap_pairs = {}
    wicks_src, wicks_dst, wicks_w, wicks_erev_loc, wicks_erev_scale = [], [], [], [], []
    generic_src, generic_dst, generic_w = [], [], []

    for _, row in connectome.iterrows():
        origin = _normalize_neuron_name(row["Source"])
        target = _normalize_neuron_name(row["Target"])
        if origin not in name_to_idx or target not in name_to_idx:
            continue
        src_idx = name_to_idx[origin]
        dst_idx = name_to_idx[target]
        edge_type = str(row["Type"]).strip()
        weight = float(row["Weight"])

        if edge_type == "electrical":
            # A self-loop contributes no electrical current (V_i - V_i == 0)
            # and would be doubled by the symmetric W + W.T assembly.
            if src_idx == dst_idx:
                continue
            pair = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            gap_pairs[pair] = max(gap_pairs.get(pair, 0.0), weight)
        elif edge_type == "chemical":
            wicks_src.append(src_idx)
            wicks_dst.append(dst_idx)
            wicks_w.append(weight)
            loc_v, scale_v = _nt_to_reversal_prior(
                nt_lookup.get((origin, target), "unknown"),
                wide_amp_priors=wide_amp_priors,
            )
            wicks_erev_loc.append(loc_v)
            wicks_erev_scale.append(scale_v)

    neuropeptide = pd.read_csv(np_path, header=None)[[0, 1]].drop_duplicates()
    monoamine = pd.read_csv(ma_path, header=None)[[0, 1]].drop_duplicates()

    def _df_to_edges(df_pairs):
        e_src, e_dst, e_w = [], [], []
        seen = set()
        for _, row in df_pairs.iterrows():
            origin = _normalize_neuron_name(row[0])
            target = _normalize_neuron_name(row[1])
            if origin in name_to_idx and target in name_to_idx:
                src_idx = name_to_idx[origin]
                dst_idx = name_to_idx[target]
                if src_idx == dst_idx or (src_idx, dst_idx) in seen:
                    continue
                seen.add((src_idx, dst_idx))
                e_src.append(src_idx)
                e_dst.append(dst_idx)
                e_w.append(1.0)
        return e_src, e_dst, e_w

    np_src, np_dst, np_w = _df_to_edges(neuropeptide)
    ma_src, ma_dst, ma_w = _df_to_edges(monoamine)
    extrasynaptic = pd.concat([neuropeptide, monoamine], ignore_index=True).drop_duplicates()
    generic_src, generic_dst, generic_w = _df_to_edges(extrasynaptic)

    if shuffle_seed is not None:
        rng = np.random.default_rng(int(shuffle_seed))
        perm = rng.permutation(len(neuron_names))
        gap_src = [int(perm[pair[0]]) for pair in gap_pairs]
        gap_dst = [int(perm[pair[1]]) for pair in gap_pairs]
        wicks_src = [int(perm[i]) for i in wicks_src]
        wicks_dst = [int(perm[i]) for i in wicks_dst]
        generic_src = [int(perm[i]) for i in generic_src]
        generic_dst = [int(perm[i]) for i in generic_dst]
        np_src = [int(perm[i]) for i in np_src]
        np_dst = [int(perm[i]) for i in np_dst]
        ma_src = [int(perm[i]) for i in ma_src]
        ma_dst = [int(perm[i]) for i in ma_dst]
        # A permutation is injective, so non-self edges stay non-self.
    else:
        gap_src = [pair[0] for pair in gap_pairs]
        gap_dst = [pair[1] for pair in gap_pairs]
    gap_w = [gap_pairs[pair] for pair in gap_pairs]
    print("Gap Junction: ", len(gap_src))
    print("Chemical Synapse: ", len(wicks_src))
    print("Extrasynaptic Connection: ", len(generic_src), f"(NP={len(np_src)}, MA={len(ma_src)})")
    synapse_list = {
        "General": (gap_src, gap_dst, gap_w),
        "Wicks": (wicks_src, wicks_dst, wicks_w, wicks_erev_loc, wicks_erev_scale),
        "Generic": (generic_src, generic_dst, generic_w),
        "NP": (np_src, np_dst, np_w),
        "MA": (ma_src, ma_dst, ma_w),
    }
    if rnn:
        model = RNNematode(len(neuron_names), neuron_list, synapse_list)
    else:
        model = NematodeForStep(
            len(neuron_names),
            neuron_list,
            synapse_list,
            use_extrasynaptic=use_extrasynaptic,
            use_bnn_residual=use_bnn_residual,
            bnn_hidden_dim=bnn_hidden_dim,
            per_edge_release=per_edge_release,
            per_neuron_stim_gain=per_neuron_stim_gain,
            wide_amp_priors=wide_amp_priors,
            evoked_syn_priors=evoked_syn_priors,
            dimensionless=dimensionless,
            leak_gauge=leak_gauge,
            split_extrasynaptic=split_extrasynaptic,
            use_syn_filter=use_syn_filter,
            tau_np_prior=tau_np_prior,
            tau_ma_prior=tau_ma_prior,
            tau_mod_prior_std=tau_mod_prior_std,
            genotype_conditioned=genotype_conditioned,
        )
    model.register_buffer("sensory_mask", torch.from_numpy(sensory_mask))
    return model
