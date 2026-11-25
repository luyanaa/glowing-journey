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

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.E = _sample_normal(self, -0.065, 0.005, shape=(input_size,), to_event=1)
        self.G = _sample_lognormal(self, math.log(0.5), 0.4, shape=(input_size,), to_event=1)
        self.C = _sample_lognormal(self, math.log(0.05), 0.4, shape=(input_size,), to_event=1)

    def forward(
        self,
        Prev: torch.Tensor,
        inputSignal: torch.Tensor,
        dt: float = timeStep,
        extraConductance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_leak = torch.clamp(self.G, min=1e-4)
        c = torch.clamp(self.C, min=1e-4)
        alpha = dt / c
        g_total = g_leak if extraConductance is None else (g_leak + torch.clamp(extraConductance, min=0.0))
        return (Prev + alpha * (inputSignal + g_leak * self.E)) / (1.0 + alpha * g_total)


class neuronLayer(PyroModule):
    """Neuron population layer combining conductance dynamics and optional sensory models."""

    def __init__(self, neuronSize: int, neuronList: Optional[Dict[int, PyroModule]] = None):
        super().__init__()
        self.neuronSize = neuronSize
        self.conductance = conductanceLayer(input_size=neuronSize)
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
    ) -> torch.Tensor:
        if externalInput is None:
            total_current = inputSignal
        else:
            total_current = inputSignal + externalInput

        output = self.conductance(Prev, total_current, dt=dt, extraConductance=extraConductance)

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

    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_syn=None, symmetric: bool = True):
        super().__init__()
        self.symmetric = bool(symmetric)
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
                    self, math.log(0.015), 0.85, shape=(self.inputSize,), to_event=1
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
            g_max_med = 0.015 if self.evoked_syn_priors else 0.01
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
        else:
            self.register_buffer("g_max", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_rest", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_slope", torch.empty(0, dtype=torch.float32))
            self.register_buffer("E_rev", torch.empty(0, dtype=torch.float32))
            self.register_buffer("v_rest_edge_delta", torch.empty(0, dtype=torch.float32))
            self.register_buffer("v_slope_edge_logdelta", torch.empty(0, dtype=torch.float32))

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
        cached: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
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
            w_drive, w_cond, v_rest, v_slope = mats
            act = torch.sigmoid((inputSignal - v_rest) / v_slope)
            return act @ w_drive, act @ w_cond

        drive = torch.zeros_like(inputSignal)
        cond = torch.zeros_like(inputSignal)
        dim = 0 if inputSignal.dim() == 1 else 1
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
        g = torch.clamp(self.g_max, min=0.0)
        e_rev = torch.clamp(self.E_rev, min=-0.095, max=0.030)
        act = torch.sigmoid((pre_v - v_rest) / slope)
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
    """Aggregates chemical (Wicks) and electrical (gap junction) synaptic currents."""

    def __init__(
        self,
        synapseList: dict,
        use_extrasynaptic: bool = False,
        neuron_size: Optional[int] = None,
        per_edge_release: bool = False,
        wide_amp_priors: bool = False,
        evoked_syn_priors: bool = False,
    ):
        super().__init__()
        self.per_edge_release = bool(per_edge_release)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        wicks_tuple = synapseList["Wicks"]
        self.Wicks_SRC, self.Wicks_DST, self.Wicks_Weight = wicks_tuple[:3]
        e_rev_loc = wicks_tuple[3] if len(wicks_tuple) > 3 else None
        e_rev_scale = wicks_tuple[4] if len(wicks_tuple) > 4 else None

        self.Gap_Junction_SRC, self.Gap_Junction_DST, self.Gap_Junction_Weight = synapseList["General"]
        self.Generic_SRC, self.Generic_DST, self.Generic_Weight = synapseList["Generic"]
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
        )
        self.general = GeneralSynapse(
            self.Gap_Junction_SRC,
            self.Gap_Junction_DST,
            self.Gap_Junction_Weight,
        )
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.use_extrasynaptic = use_extrasynaptic
        if use_extrasynaptic and len(self.Generic_SRC) > 0:
            self.extrasyn_gain = _sample_normal(
                self, 0.0, 0.05, shape=(len(self.Generic_SRC),), to_event=1
            )
            self.register_buffer("generic_src_t", torch.as_tensor(self.Generic_SRC, dtype=torch.long))
            self.register_buffer("generic_dst_t", torch.as_tensor(self.Generic_DST, dtype=torch.long))

    def build_step_cache(
        self, n_neurons: int, device: torch.device, dtype: torch.dtype
    ) -> dict:
        """Build per-rollout dense (N, N) synaptic matrices once instead of per ODE substep."""
        w_extra = None
        if self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            w_extra = torch.zeros((n_neurons, n_neurons), device=device, dtype=dtype).index_put(
                (self.generic_src_t, self.generic_dst_t),
                self.extrasyn_gain.to(dtype=dtype),
                accumulate=True,
            )
        return {
            "wicks": self.wicks.build_weight_matrices(n_neurons, device, dtype),
            "general": self.general.build_weight_matrix(n_neurons, device, dtype),
            "extra": w_extra,
        }

    def forward_terms(
        self, inputSignal: torch.Tensor, syn_cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wicks_cache = None if syn_cache is None else syn_cache.get("wicks")
        gen_cache = None if syn_cache is None else syn_cache.get("general")
        drive_w, cond_w = self.wicks.forward_terms(inputSignal, cached=wicks_cache)
        drive_g, cond_g = self.general.forward_terms(inputSignal, cached=gen_cache)
        drive = drive_w + drive_g
        cond = cond_w + cond_g
        if self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            w_extra = None if syn_cache is None else syn_cache.get("extra")
            if w_extra is not None:
                drive = drive + torch.sigmoid((inputSignal + 0.055) / 0.02) @ w_extra
            else:
                dim = 0 if inputSignal.dim() == 1 else 1
                pre_v = torch.index_select(inputSignal, dim, self.generic_src_t)
                mod_cur = self.extrasyn_gain * torch.sigmoid((pre_v + 0.055) / 0.02)
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

    def step(
        self,
        voltage: torch.Tensor,
        ca_prev: torch.Tensor,
        b_prev: torch.Tensor,
        dt: float,
        v_th_ca: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_th = self.v_th_ca if v_th_ca is None else v_th_ca
        influx = self.alpha_ca * torch.sigmoid((voltage - v_th) / self.v_slope_ca)
        ca_next = (ca_prev + dt * (self.ca_rest / self.tau_ca + influx)) / (1.0 + dt / self.tau_ca)
        ca_n = torch.pow(torch.clamp(ca_next, min=1e-6), self.hill_n)
        rate_on = self.k_on * ca_n
        b_next = (b_prev + dt * rate_on) / (1.0 + dt * (rate_on + self.k_off))
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
      1. Shared per-neuron local encoder (4 -> 16 -> 5) over masked (y_last, y_recent, y_slope, obs_any)
         producing initial (delta_v0, delta_ca0, i_bg_local) and bi-exponential fluorescence relaxation
         (drift_fast0, drift_slow0).
      2. Zero-initialized low-rank whole-brain bottleneck (3*N -> 32 -> rank -> 2*N) capturing
         cross-neuron spontaneous behavioral modes (tonic current + slow drift).
      3. Learnable positive decay timescales (tau_bg, tau_fast, tau_slow).
    """

    def __init__(self, neuron_size: int, rank: int = 16, local_hidden: int = 16):
        super().__init__()
        self.neuron_size = int(neuron_size)
        self.rank = int(rank)

        self.local_fc1 = PyroModule[nn.Linear](4, local_hidden)
        self.local_fc2 = PyroModule[nn.Linear](local_hidden, 5)

        self.global_fc1 = PyroModule[nn.Linear](self.neuron_size * 3, 32)
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

    @staticmethod
    def extract_global_features(
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly causal artifact-safe summary features extracted solely from ``y[:, :pre_vols]``."""
        if y_pre.dim() == 2:
            y_pre = y_pre.unsqueeze(0)
            if mask_pre is not None and mask_pre.dim() == 2:
                mask_pre = mask_pre.unsqueeze(0)
        _, t_pre, _ = y_pre.shape
        dtype = y_pre.dtype
        device = y_pre.device

        m = torch.ones_like(y_pre, dtype=dtype) if mask_pre is None else mask_pre.to(device=device, dtype=dtype)

        # Reject pre-stimulus windows with extreme bimodal spikes (max |y_pre| > 2.0) and smoothly
        # saturate remaining observations via 0.8 * tanh(y / 0.8).
        y_raw_masked = torch.where(m > 0.5, torch.nan_to_num(y_pre, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(y_pre))
        pre_abs_max = y_raw_masked.abs().max(dim=1).values  # (B, N)
        clean_pre = (pre_abs_max <= 2.0).to(dtype=dtype)
        y_safe = torch.where(m > 0.5, 0.8 * torch.tanh(y_raw_masked / 0.8), torch.zeros_like(y_pre))
        obs_count = m.sum(dim=1)  # (B, N)
        obs_any = (obs_count > 0.5).to(dtype=dtype) * clean_pre

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

        global_in = torch.cat([y_last, y_slope, obs_any], dim=-1)  # (B, 3*N)
        return global_in, y_last, y_recent, y_slope, obs_any

    def forward(
        self,
        y_pre: torch.Tensor,
        mask_pre: Optional[torch.Tensor] = None,
        z_bg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (delta_v0, delta_ca0, i_bg0, drift_fast0, drift_slow0, tau_bg, tau_fast, tau_slow)."""
        global_in, y_last, y_recent, y_slope, obs_any = self.extract_global_features(y_pre, mask_pre)
        n_neu = self.neuron_size

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
        return (
            delta_v0,
            delta_ca0,
            i_bg0,
            drift_fast0,
            drift_slow0,
            self.tau_bg,
            self.tau_fast,
            self.tau_slow,
        )


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
    ):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuronSize = neuronSize
        self.per_neuron_stim_gain = bool(per_neuron_stim_gain)
        self.wide_amp_priors = bool(wide_amp_priors)
        self.evoked_syn_priors = bool(evoked_syn_priors)
        self.Neuron = neuronLayer(neuronSize, NeuronList)
        self.synapse = synapseLayer(
            synapseList,
            use_extrasynaptic=use_extrasynaptic,
            neuron_size=neuronSize,
            per_edge_release=per_edge_release,
            wide_amp_priors=wide_amp_priors,
            evoked_syn_priors=evoked_syn_priors,
        )
        # Optogenetic Chrimson cation-channel conductance (nS) with physiological reversal E_opt = 0 V
        # (and inhibitory reversal E_inh = -85 mV if negative stimulation is supplied).
        self.stim_e_rev = 0.0
        self.stim_e_inh = -0.085
        stim_loc = math.log(2.0) if self.wide_amp_priors else math.log(1.2)
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
    ) -> torch.Tensor:
        drive_cur, syn_cond = self.synapse.forward_terms(Prev, syn_cache=syn_cache)
        if self.bnn_residual is not None:
            drive_cur = drive_cur + self.bnn_residual(Prev)
        if ExternalInput is None:
            ext_current = torch.zeros_like(Prev)
            total_extra_cond = syn_cond
        else:
            gain = self.stim_gain_vec if self.per_neuron_stim_gain else self.stim_gain
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
) -> torch.Tensor:
    """Exact first-order rise/decay update for the optogenetic actuator.

    The input is held constant over one simulation substep.  Comparing the
    current state with the target avoids overshoot even when ``dt_sim``
    exceeds the rise time, and keeps negative (inhibitory) commands instead of
    clamping them to zero.  For a positive step from zero this is
    ``1 - exp(-dt_sim / tau_rise)``; for a decay toward zero it is
    ``stim_act * exp(-dt_sim / tau_decay)``.
    """
    tau_rise_t = torch.as_tensor(max(float(tau_rise), 1e-6), dtype=stim_act.dtype, device=stim_act.device)
    tau_decay_t = torch.as_tensor(max(float(tau_decay), 1e-6), dtype=stim_act.dtype, device=stim_act.device)
    tau = torch.where(ext_t > stim_act, tau_rise_t, tau_decay_t)
    return ext_t + (stim_act - ext_t) * torch.exp(-float(dt_sim) / tau)


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
        per_neuron_kinetics: bool = False,
        per_edge_release: bool = False,
        stim_actuator: bool = False,
        stim_tau_rise: float = 0.08,
        stim_tau_decay: float = 0.4,
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

        self.register_buffer("_dev_ref", torch.tensor(0.0))
        neuron_size = getattr(model, "neuronSize", 300)
        self.per_neuron_kinetics = bool(per_neuron_kinetics)
        self.gcamp = CalciumGCaMPLayer(neuron_size, per_neuron_kinetics=self.per_neuron_kinetics)
        self.prestim_encoder = (
            CausalPreStimEncoder(neuron_size, rank=self.prestim_rank) if self.use_prestim_encoder else None
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_t, n_neurons = stim_input.shape
        dt_sim = dt_obs / max(1, self.substeps)

        if hasattr(self.model, "Neuron") and hasattr(self.model.Neuron, "conductance"):
            v_rest = self.model.Neuron.conductance.E.unsqueeze(0).expand(batch_size, n_neurons)
        else:
            v_rest = torch.full((batch_size, n_neurons), -0.065, device=stim_input.device, dtype=stim_input.dtype)

        v = v_rest
        sensory_states: Dict[str, object] = {}
        syn_cache = (
            self.model.synapse.build_step_cache(n_neurons, v.device, v.dtype)
            if (hasattr(self.model, "synapse") and hasattr(self.model.synapse, "build_step_cache"))
            else None
        )
        # Settle coupled network resting equilibrium before anchoring F0 baseline
        settle_steps = max(8, self.substeps * 4)
        for _ in range(settle_steps):
            v = self.model(
                v,
                ExternalInput=None,
                VoltageClamp=None,
                dt=dt_sim,
                sensory_states=sensory_states,
                syn_cache=syn_cache,
            )
        v_th_ca = self.gcamp.v_th_ca
        ca, b, b_baseline = self.gcamp.resting_state(v, v_th_ca=v_th_ca)

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
                    # Per-component gates: (5, B, N) for delta_v0, delta_ca0,
                    # i_bg0, drift_fast0, drift_slow0.
                    for i in range(5):
                        values[i] = values[i] * g[i]
                else:
                    if g.dim() == 1:
                        g = g.unsqueeze(0)
                    if g.shape[0] == 1 and batch_size > 1:
                        g = g.expand(batch_size, -1)
                    # Suppress the pre-stimulus state/drift on selected neurons (e.g. direct
                    # stimulus targets and their 1-hop downstream) while keeping it on
                    # unconnected neurons. Time constants are scalar and are not gated.
                    for i in range(5):
                        values[i] = values[i] * g
                prestim_state = tuple(values)

        y_preds = []
        v_traj = []
        ca_traj = []
        stim_act = torch.zeros_like(v)

        for t in range(n_t):
            if prestim_state is not None and t == inject_t:
                delta_v0, delta_ca0, i_bg0, drift_fast0, drift_slow0, tau_bg, tau_fast, tau_slow = prestim_state
                v = v + delta_v0
                ca = torch.clamp(ca + delta_ca0, min=1e-4)

            ext_t = stim_input[:, t, :]
            clamp_t = None if voltage_clamp is None else voltage_clamp[:, t, :]
            for sub in range(self.substeps):
                bg_cur = None
                if prestim_state is not None and t >= inject_t:
                    t_rel_sim = (t - inject_t) * dt_obs + (sub + 0.5) * dt_sim
                    bg_cur = prestim_state[2] * torch.exp(-t_rel_sim / prestim_state[5])
                if self.stim_actuator:
                    # First-order rise/decay actuator; the command is not applied
                    # instantaneously to the membrane.  Use the exact exponential
                    # update so dt_sim > tau_rise cannot overshoot, and preserve
                    # negative commands.
                    stim_act = _advance_stim_actuator(
                        stim_act, ext_t, dt_sim, self.stim_tau_rise, self.stim_tau_decay
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
                )
                ca, b = self.gcamp.step(v, ca, b, dt=dt_sim, v_th_ca=v_th_ca)
            y_t = self.gcamp.readout(b, b_baseline)
            if prestim_state is not None and t >= inject_t:
                t_rel_obs = (t - inject_t) * dt_obs
                y_t = (
                    y_t
                    + prestim_state[3] * torch.exp(-t_rel_obs / prestim_state[6])
                    + prestim_state[4] * torch.exp(-t_rel_obs / prestim_state[7])
                )
            y_preds.append(y_t)
            v_traj.append(v)
            ca_traj.append(ca)

        return (
            torch.stack(y_preds, dim=1),
            torch.stack(v_traj, dim=1),
            torch.stack(ca_traj, dim=1),
        )

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
    ) -> torch.Tensor:
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

        if self.prestim_encoder is not None and pre_obs is None and y_obs is not None:
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
        if self.prestim_encoder is not None and self.variational_prestim:
            with batch_plate:
                z_zeros = torch.zeros(batch_size, self.prestim_rank, device=ext.device, dtype=ext.dtype)
                prestim_z = pyro.sample(
                    "prestim_z",
                    dist.Normal(z_zeros, torch.ones_like(z_zeros)).to_event(1),
                )

        y_pred, _, _ = self._rollout(
            ext,
            v_clamp,
            self.dt_obs if dt_obs is None else float(dt_obs),
            pre_stim_obs=pre_obs,
            pre_stim_mask=pre_msk,
            drift_gate=d_gate,
            prestim_z=prestim_z,
        )
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

    neuropeptide = pd.read_csv(np_path, header=None)[[0, 1]]
    monoamine = pd.read_csv(ma_path, header=None)[[0, 1]]
    extrasynaptic = pd.concat([neuropeptide, monoamine], ignore_index=True).drop_duplicates()

    for _, row in extrasynaptic.iterrows():
        origin = _normalize_neuron_name(row[0])
        target = _normalize_neuron_name(row[1])
        if origin in name_to_idx and target in name_to_idx:
            generic_src.append(name_to_idx[origin])
            generic_dst.append(name_to_idx[target])
            generic_w.append(1.0)

    if shuffle_seed is not None:
        rng = np.random.default_rng(int(shuffle_seed))
        perm = rng.permutation(len(neuron_names))
        gap_src = [int(perm[pair[0]]) for pair in gap_pairs]
        gap_dst = [int(perm[pair[1]]) for pair in gap_pairs]
        wicks_src = [int(perm[i]) for i in wicks_src]
        wicks_dst = [int(perm[i]) for i in wicks_dst]
        generic_src = [int(perm[i]) for i in generic_src]
        generic_dst = [int(perm[i]) for i in generic_dst]
        # A permutation is injective, so non-self edges stay non-self.
    else:
        gap_src = [pair[0] for pair in gap_pairs]
        gap_dst = [pair[1] for pair in gap_pairs]
    gap_w = [gap_pairs[pair] for pair in gap_pairs]
    print("Gap Junction: ", len(gap_src))
    print("Chemical Synapse: ", len(wicks_src))
    print("Extrasynaptic Connection: ", len(generic_src))
    synapse_list = {
        "General": (gap_src, gap_dst, gap_w),
        "Wicks": (wicks_src, wicks_dst, wicks_w, wicks_erev_loc, wicks_erev_scale),
        "Generic": (generic_src, generic_dst, generic_w),
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
        )
    model.register_buffer("sensory_mask", torch.from_numpy(sensory_mask))
    return model
