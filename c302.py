import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
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
    ) -> torch.Tensor:
        if externalInput is None:
            total_current = inputSignal
        else:
            total_current = inputSignal + externalInput

        output = self.conductance(Prev, total_current, dt=dt, extraConductance=extraConductance)

        if len(self._neuron_List) > 0:
            is_1d = output.dim() == 1
            ext = torch.zeros_like(total_current) if externalInput is None else externalInput
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

    def __init__(self, synapseInput, synapseOutput, synapseWeight, g_syn=None):
        super().__init__()
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
                self.g_syn = _sample_lognormal(
                    self, math.log(0.01), 0.4, shape=(self.inputSize,), to_event=1
                )
            else:
                self.g_syn = g_syn
        else:
            self.register_buffer("g_syn", torch.empty(0, dtype=torch.float32))

    def forward_terms(self, inputSignal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (synaptic_drive_current, postsynaptic_conductance_sum) for semi-implicit integration."""
        drive = torch.zeros_like(inputSignal)
        cond = torch.zeros_like(inputSignal)
        if self.inputSize == 0:
            return drive, cond
        dim = 0 if inputSignal.dim() == 1 else 1
        pre_v = torch.index_select(inputSignal, dim, self.synapseInput)
        g_eff = torch.clamp(self.g_syn, min=0.0) * self.synapseWeight
        if dim == 0:
            drive.index_add_(0, self.synapseOutput, g_eff * pre_v)
            cond.index_add_(0, self.synapseOutput, g_eff)
        else:
            drive.index_add_(1, self.synapseOutput, g_eff.unsqueeze(0) * pre_v)
            cond_1d = torch.zeros(inputSignal.shape[1], device=inputSignal.device, dtype=inputSignal.dtype)
            cond_1d.index_add_(0, self.synapseOutput, g_eff)
            cond = cond_1d.unsqueeze(0).expand_as(inputSignal)
        return drive, cond

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        drive, cond = self.forward_terms(inputSignal)
        return drive - cond * inputSignal


# Refer to Wicks et al. (1996) / HAL-03705452:
# Vslope = +15 mV (0.015 V), gsyn = 0.6 nS, Vrest = -76 mV (-0.076 V)
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
    ):
        super().__init__()
        src = np.asarray(synapseInput, dtype=np.int64).reshape(-1)
        dst = np.asarray(synapseOutput, dtype=np.int64).reshape(-1)
        w = np.asarray(synapseWeight, dtype=np.float32).reshape(-1)
        self.inputSize = int(len(src))
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

            self.g_max = (
                _sample_lognormal(self, math.log(0.01), 0.4, shape=(self.inputSize,), to_event=1)
                if g_max is None
                else g_max
            )
            self.V_rest = (
                _sample_normal(self, -0.042, 0.008, shape=(self.inputSize,), to_event=1)
                if V_rest is None
                else V_rest
            )
            self.V_slope = (
                _sample_lognormal(self, math.log(0.015), 0.3, shape=(self.inputSize,), to_event=1)
                if V_slope is None
                else V_slope
            )
            self.E_rev = (
                PyroSample(lambda mod: dist.Normal(mod.e_rev_loc_buf, mod.e_rev_scale_buf).to_event(1))
                if E_rev is None
                else E_rev
            )
        else:
            self.register_buffer("g_max", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_rest", torch.empty(0, dtype=torch.float32))
            self.register_buffer("V_slope", torch.empty(0, dtype=torch.float32))
            self.register_buffer("E_rev", torch.empty(0, dtype=torch.float32))

    def forward_terms(self, inputSignal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (synaptic_drive_current, postsynaptic_conductance_sum) for semi-implicit integration."""
        drive = torch.zeros_like(inputSignal)
        cond = torch.zeros_like(inputSignal)
        if self.inputSize == 0:
            return drive, cond
        dim = 0 if inputSignal.dim() == 1 else 1
        pre_v = torch.index_select(inputSignal, dim, self.synapseInput)
        slope = torch.clamp(self.V_slope, min=1e-3)
        g = torch.clamp(self.g_max, min=0.0)
        act = torch.sigmoid((pre_v - self.V_rest) / slope)
        g_eff = g * self.synapseWeight * act
        drive.index_add_(dim, self.synapseOutput, g_eff * self.E_rev)
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

    def __init__(self, synapseList: dict, use_extrasynaptic: bool = False):
        super().__init__()
        wicks_tuple = synapseList["Wicks"]
        self.Wicks_SRC, self.Wicks_DST, self.Wicks_Weight = wicks_tuple[:3]
        e_rev_loc = wicks_tuple[3] if len(wicks_tuple) > 3 else None
        e_rev_scale = wicks_tuple[4] if len(wicks_tuple) > 4 else None

        self.Gap_Junction_SRC, self.Gap_Junction_DST, self.Gap_Junction_Weight = synapseList["General"]
        self.Generic_SRC, self.Generic_DST, self.Generic_Weight = synapseList["Generic"]

        self.wicks = WicksSynapse(
            self.Wicks_SRC,
            self.Wicks_DST,
            self.Wicks_Weight,
            e_rev_loc=e_rev_loc,
            e_rev_scale=e_rev_scale,
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

    def forward_terms(self, inputSignal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        drive_w, cond_w = self.wicks.forward_terms(inputSignal)
        drive_g, cond_g = self.general.forward_terms(inputSignal)
        drive = drive_w + drive_g
        cond = cond_w + cond_g
        if self.use_extrasynaptic and len(self.Generic_SRC) > 0:
            dim = 0 if inputSignal.dim() == 1 else 1
            pre_v = torch.index_select(inputSignal, dim, self.generic_src_t)
            mod_cur = self.extrasyn_gain * torch.sigmoid((pre_v + 0.055) / 0.02)
            drive.index_add_(dim, self.generic_dst_t, mod_cur)
        return drive, cond

    def forward(self, inputSignal: torch.Tensor) -> torch.Tensor:
        drive, cond = self.forward_terms(inputSignal)
        return drive - cond * inputSignal


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

    def __init__(self, neuron_size: int, ca_rest: float = 0.05, hill_n: float = 2.0):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuron_size = neuron_size
        self.ca_rest = float(ca_rest)
        self.hill_n = float(hill_n)

        # Shared biophysical indicator kinetics + neuron-specific gain/offset
        self.v_th_ca = _sample_normal(self, -0.048, 0.01)
        self.v_slope_ca = _sample_lognormal(self, math.log(0.012), 0.3)
        self.alpha_ca = _sample_lognormal(self, math.log(2.0), 0.4)
        self.tau_ca = _sample_lognormal(self, math.log(0.6), 0.4)
        self.k_on = _sample_lognormal(self, math.log(4.0), 0.35)
        self.k_off = _sample_lognormal(self, math.log(1.2), 0.35)

        self.gain = _sample_lognormal(self, math.log(1.5), 0.5, shape=(neuron_size,), to_event=1)
        self.offset = _sample_normal(self, 0.0, 0.05, shape=(neuron_size,), to_event=1)

    def resting_state(self, v_rest: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute analytical steady-state ([Ca]_ss, b_ss, b_ss) at resting voltage."""
        influx_ss = self.alpha_ca * torch.sigmoid((v_rest - self.v_th_ca) / self.v_slope_ca)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        influx = self.alpha_ca * torch.sigmoid((voltage - self.v_th_ca) / self.v_slope_ca)
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
        ca, b, b_base = self.resting_state(v_rest)
        outs = []
        for t in range(v_seq.shape[1]):
            ca, b = self.step(v_seq[:, t, :], ca, b, dt=dt)
            outs.append(self.readout(b, b_base))
        res = torch.stack(outs, dim=1)
        return res.squeeze(0) if squeeze else res


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
    ):
        super().__init__()
        self.register_buffer("_dev_ref", torch.tensor(0.0))
        self.neuronSize = neuronSize
        self.Neuron = neuronLayer(neuronSize, NeuronList)
        self.synapse = synapseLayer(synapseList, use_extrasynaptic=use_extrasynaptic)
        self.stim_gain = _sample_lognormal(self, math.log(0.04), 0.5)
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
    ) -> torch.Tensor:
        drive_cur, syn_cond = self.synapse.forward_terms(Prev)
        if self.bnn_residual is not None:
            drive_cur = drive_cur + self.bnn_residual(Prev)
        if ExternalInput is None:
            ext_current = torch.zeros_like(Prev)
        else:
            ext_current = self.stim_gain * ExternalInput
        ConnectomeOutput = self.Neuron(
            Prev,
            drive_cur,
            ext_current,
            dt=dt,
            sensory_states=sensory_states,
            extraConductance=syn_cond,
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
    ):
        super().__init__()
        self.model = model
        # Keep backward compatibility with callers passing `scale=` while never mutating globals
        self.substeps = int(scale if scale is not None else substeps)
        self.dt_obs = float(dt_obs)
        self.likelihood = likelihood
        self.student_t_df = float(student_t_df)

        self.register_buffer("_dev_ref", torch.tensor(0.0))
        neuron_size = getattr(model, "neuronSize", 300)
        self.gcamp = CalciumGCaMPLayer(neuron_size)
        # Dimensionless observation noise multiplier (when noise_scale is provided) or base std
        self.obs_sigma = _sample_lognormal(self, math.log(0.15), 0.5)

    def _rollout(
        self,
        stim_input: torch.Tensor,
        voltage_clamp: Optional[torch.Tensor],
        dt_obs: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_t, n_neurons = stim_input.shape
        dt_sim = dt_obs / max(1, self.substeps)

        if hasattr(self.model, "Neuron") and hasattr(self.model.Neuron, "conductance"):
            v_rest = self.model.Neuron.conductance.E.unsqueeze(0).expand(batch_size, n_neurons)
        else:
            v_rest = torch.full((batch_size, n_neurons), -0.065, device=stim_input.device, dtype=stim_input.dtype)

        v = v_rest
        sensory_states: Dict[str, object] = {}
        # Settle coupled network resting equilibrium before anchoring F0 baseline
        settle_steps = max(8, self.substeps * 4)
        for _ in range(settle_steps):
            v = self.model(
                v,
                ExternalInput=None,
                VoltageClamp=None,
                dt=dt_sim,
                sensory_states=sensory_states,
            )
        ca, b, b_baseline = self.gcamp.resting_state(v)

        y_preds = []
        v_traj = []
        ca_traj = []

        for t in range(n_t):
            ext_t = stim_input[:, t, :]
            clamp_t = None if voltage_clamp is None else voltage_clamp[:, t, :]
            for _ in range(self.substeps):
                v = self.model(
                    v,
                    ExternalInput=ext_t,
                    VoltageClamp=clamp_t,
                    dt=dt_sim,
                    sensory_states=sensory_states,
                )
                ca, b = self.gcamp.step(v, ca, b, dt=dt_sim)
            y_t = self.gcamp.readout(b, b_baseline)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward rollout and return (y_gcamp_pred, V_traj, Ca_traj)."""
        squeezed = stim_input.dim() == 2
        if squeezed:
            stim_input = stim_input.unsqueeze(0)
            if voltage_clamp is not None:
                voltage_clamp = voltage_clamp.unsqueeze(0)
        y_pred, v_traj, ca_traj = self._rollout(
            stim_input, voltage_clamp, self.dt_obs if dt_obs is None else float(dt_obs)
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
        else:
            ext = ExternalInput
            v_clamp = VoltageClamp
            y_obs = y
            obs_mask = mask
            n_scale = noise_scale

        if ext is None:
            ext = torch.zeros_like(v_clamp)

        y_pred, _, _ = self._rollout(
            ext, v_clamp, self.dt_obs if dt_obs is None else float(dt_obs)
        )
        batch_size, n_t, _ = y_pred.shape
        obs_sigma = torch.clamp(self.obs_sigma, min=1e-3)

        if n_scale is not None:
            # When empirical per-neuron noise_scale (B, N) or (B, T, N) is passed,
            # obs_sigma acts as a global calibration floor.
            scale_t = torch.sqrt((obs_sigma * 0.5) ** 2 + n_scale ** 2)
        else:
            scale_t = obs_sigma

        with pyro.plate("batch", batch_size):
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

        self.model.synapse.wicks = WicksSynapse(
            self.model.synapse.Wicks_SRC,
            self.model.synapse.Wicks_DST,
            self.model.synapse.Wicks_Weight,
            g_max=PyroSample(dist.LogNormal(wicks_loc[0], torch.clamp(wicks_scale[0], min=1e-4)).to_event(1)),
            V_rest=PyroSample(dist.Normal(wicks_loc[1], torch.clamp(wicks_scale[1], min=1e-4)).to_event(1)),
            V_slope=PyroSample(dist.LogNormal(wicks_loc[2], torch.clamp(wicks_scale[2], min=1e-4)).to_event(1)),
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


def _nt_to_reversal_prior(nt_str: str) -> Tuple[float, float]:
    """Map neurotransmitter annotation to chemical synapse reversal potential prior (Volts)."""
    nt = str(nt_str).strip()
    if "GABA" in nt:
        return -0.075, 0.015
    if "Acetylcholine" in nt:
        return 0.0, 0.020
    if "Glutamate" in nt:
        # C. elegans glutamate can be excitatory (NMDA/AMPA) or inhibitory (GluCl)
        return -0.035, 0.035
    return -0.025, 0.035


def readConnectome(
    path: str = "./data/CElegansNeuronTables.xls",
    rnn: bool = False,
    use_sensory_models: bool = False,
    use_extrasynaptic: bool = False,
    use_bnn_residual: bool = False,
    bnn_hidden_dim: int = 16,
    aconnectome_ids_path: str = "./pumpprobe/pumpprobe/aconnectome_ids.txt",
    edgelist_path: str = "./data/herm_full_edgelist.csv",
    np_path: str = "./data/edgelist_NP.csv",
    ma_path: str = "./data/edgelist_MA.csv",
):
    """Build C. elegans connectome step model aligned to the 300 canonical neuron IDs."""
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

    gap_src, gap_dst, gap_w = [], [], []
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
            gap_src.append(src_idx)
            gap_dst.append(dst_idx)
            gap_w.append(weight)
        elif edge_type == "chemical":
            wicks_src.append(src_idx)
            wicks_dst.append(dst_idx)
            wicks_w.append(weight)
            loc_v, scale_v = _nt_to_reversal_prior(nt_lookup.get((origin, target), "unknown"))
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
        )
    model.register_buffer("sensory_mask", torch.from_numpy(sensory_mask))
    return model
