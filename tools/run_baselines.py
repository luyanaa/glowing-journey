#!/usr/bin/env python3
"""Step 4.2: Unified 3-Task Competitive Baselines on Fold-0 Causal Randi WT.

Strictly separates:
  Task 1 - Open-loop forecasting (pre-stim 0..9 observed -> rollout post-stim 10..59)
  Task 2 - Online causal filtering (0.5s / 1-step and 2.5s / 5-step ahead using y_{<=t-k})
  Task 3 - Intervention prediction (counterfactual stim vs 0 + pairwise K_ij(t) kernels)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import c302
import vae
from DMM import Combiner, Emitter, GatedTransition
from pyro_svi_c302 import compute_stratified_metrics, evaluate_predictions
from utils import RandiTrialDataset

RUNS_BASE = "/Users/yanlu/Documents/glowing-journey_runs/phase25_decoupled_extrasyn_fold0_2026-11-25"


def _safe_r(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if af.size < 2 or bf.size < 2:
        return 0.0
    da = af - af.mean()
    db = bf - bf.mean()
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    return float(np.sum(da * db) / denom) if denom > 1e-12 else 0.0


def _score_forecast_slice(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    if not np.any(valid):
        return {}
    yt, yp = y_true[valid], y_pred[valid]
    out = {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r": _safe_r(yt, yp),
        "n_obs": int(yt.size),
    }
    inlier = valid & (np.abs(y_true) <= 3.0)
    if np.any(inlier):
        yti, ypi = y_true[inlier], y_pred[inlier]
        out["inlier_r2"] = float(r2_score(yti, ypi))
        out["inlier_r"] = _safe_r(yti, ypi)
        out["inlier_mae"] = float(mean_absolute_error(yti, ypi))
    return out


def _summarize_metrics(m: Dict[str, float]) -> Dict[str, dict]:
    open_keys = [
        "r2", "inlier_r2", "post_r2", "post_inlier_r2", "post_inlier_r", "post_mae", "post_rmse",
    ]
    inter_keys = [
        "evoked_post_r2", "evoked_post_r",
        "evoked_stim_r2", "evoked_stim_r",
        "evoked_hop1_r2", "evoked_hop1_r", "evoked_hop1_amp_ratio",
        "evoked_hop2_r2", "evoked_hop2_r",
        "pair_evoked_hop1_count", "pair_evoked_hop1_r", "pair_evoked_hop1_r2", "pair_evoked_hop1_sign_acc",
        "kernel_hop1_n_pairs", "kernel_hop1_r", "kernel_hop1_r2", "kernel_hop1_pair_curve_r_mean",
        "kernel_hop1_auc_r", "kernel_hop1_peak_amp_r", "kernel_hop1_peak_amp_mae",
        "kernel_hop1_peak_latency_mae_s", "kernel_hop1_peak_latency_r", "kernel_hop1_fft_r", "kernel_hop1_sign_acc",
        "evoked_np_only_r2", "evoked_np_only_r",
        "evoked_np_no_hop12_r2", "evoked_np_no_hop12_r",
        "evoked_np_no_wired_path_r2", "evoked_np_no_wired_path_r",
    ]
    return {
        "open_loop": {k: m[k] for k in open_keys if k in m},
        "intervention": {k: m[k] for k in inter_keys if k in m},
    }


def _pack_model_result(
    y_te: np.ndarray,
    m_te: np.ndarray,
    st_te: np.ndarray,
    pred_open: np.ndarray,
    pred_zero_open: Optional[np.ndarray],
    pred_1step: Optional[np.ndarray],
    pred_5step: Optional[np.ndarray],
    adj: np.ndarray,
    np_adj: np.ndarray,
    train_wall_s: float,
    pre_vols: int = 10,
) -> Dict:
    strat = compute_stratified_metrics(
        y_te,
        pred_open,
        m_te,
        stim=st_te,
        pre_vols=pre_vols,
        adjacency=adj,
        evoked_baseline="mean" if pred_zero_open is not None else "linear",
        y_pred_counterfactual=pred_zero_open,
        np_adjacency=np_adj,
        dt_obs=0.5,
    )
    res = _summarize_metrics(strat)
    if pred_1step is not None or pred_5step is not None:
        online = {}
        if pred_1step is not None:
            v1 = np.zeros_like(m_te)
            v1[:, pre_vols:, :] = m_te[:, pre_vols:, :]
            online["step1_0.5s"] = _score_forecast_slice(y_te, pred_1step, v1)
        if pred_5step is not None:
            v5 = np.zeros_like(m_te)
            v5[:, pre_vols + 4 :, :] = m_te[:, pre_vols + 4 :, :]
            online["step5_2.5s"] = _score_forecast_slice(y_te, pred_5step, v5)
        res["online_filtering"] = online
    else:
        res["online_filtering"] = None
    res["train_wall_s"] = float(train_wall_s)
    return res


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target)[mask]
    return (diff * diff).mean()


# ---------------------------------------------------------------------------
# Baseline 1 & 2: Persistence & Causal Exponential Decay
# ---------------------------------------------------------------------------

def run_persistence_and_expdecay(
    y_tr: torch.Tensor, m_tr: torch.Tensor, y_te: torch.Tensor, m_te: torch.Tensor, st_te: torch.Tensor,
    adj: np.ndarray, np_adj: np.ndarray, pre_vols: int = 10
) -> Dict[str, dict]:
    t0 = time.time()
    y_np, m_np = y_te.numpy(), m_te.numpy().astype(bool)
    st_np = st_te.numpy()

    # Persistence: pre-stim observed mean carried open-loop; causal last-observed for 1-step & 5-step
    pre_cnt = np.maximum(m_np[:, :pre_vols, :].sum(axis=1, keepdims=True), 1)
    pre_mean = (y_np[:, :pre_vols, :] * m_np[:, :pre_vols, :]).sum(axis=1, keepdims=True) / pre_cnt
    pers_open = np.repeat(pre_mean, y_np.shape[1], axis=1)

    last_obs = np.zeros_like(y_np)
    cur = np.zeros((y_np.shape[0], y_np.shape[2]), dtype=np.float32)
    for t in range(y_np.shape[1]):
        last_obs[:, t, :] = cur
        mt = m_np[:, t, :]
        cur = np.where(mt, y_np[:, t, :], cur)
    pers_1step = last_obs
    pers_5step = np.zeros_like(y_np)
    for t in range(4, y_np.shape[1]):
        pers_5step[:, t, :] = last_obs[:, t - 4, :]

    out = {
        "persistence_premean": _pack_model_result(
            y_np, m_np, st_np, pers_open, pers_open, pers_1step, pers_5step, adj, np_adj, time.time() - t0, pre_vols
        )
    }

    # Causal Exp-Decay: y_{t+k} = alpha^k * x_t; fit alpha in (0, 1) on training pre->post
    t_ed = time.time()
    ytr_np, mtr_np = y_tr.numpy(), m_tr.numpy().astype(bool)
    cnt_tr = np.maximum(mtr_np[:, :pre_vols, :].sum(axis=1, keepdims=True), 1)
    pm_tr = (ytr_np[:, :pre_vols, :] * mtr_np[:, :pre_vols, :]).sum(axis=1, keepdims=True) / cnt_tr
    post_len = ytr_np.shape[1] - pre_vols
    steps_grid = np.arange(1, post_len + 1, dtype=np.float64)[None, :, None]
    best_alpha, best_loss = 0.9, float("inf")
    yt_p = ytr_np[:, pre_vols:, :]
    mt_p = mtr_np[:, pre_vols:, :] & (np.abs(yt_p) <= 3.0)
    for alpha in np.linspace(0.70, 0.995, 60):
        mse = float(np.mean(((pm_tr * (alpha ** steps_grid) - yt_p)[mt_p]) ** 2))
        if mse < best_loss:
            best_loss = mse
            best_alpha = float(alpha)

    ed_open = np.repeat(pre_mean, y_np.shape[1], axis=1)
    ed_open[:, pre_vols:, :] = pre_mean * (best_alpha ** steps_grid)
    ed_1step = best_alpha * pers_1step
    ed_5step = (best_alpha ** 5) * pers_5step
    ed_res = _pack_model_result(
        y_np, m_np, st_np, ed_open, ed_open, ed_1step, ed_5step, adj, np_adj, time.time() - t_ed, pre_vols
    )
    ed_res["learned_alpha"] = best_alpha
    out["exp_decay"] = ed_res
    return out


# ---------------------------------------------------------------------------
# Baseline 3 & 4: Graph VAR & Connectome-Constrained Latent LDS (Creamer et al. 2025 style)
# ---------------------------------------------------------------------------

class ConnectomeLinearSSM(nn.Module):
    """300-D connectome-constrained linear state-space / VAR model with Kalman update.

    If ``direct_obs=True``, acts as Graph VAR (C = I, d = 0).
    If ``direct_obs=False``, acts as Creamer-style Connectome-Constrained LDS with
    learned C, d, initial state encoder gain, and diagonal Kalman filter gain.
    """

    def __init__(self, adj_mask: torch.Tensor, direct_obs: bool = False):
        super().__init__()
        self.n = int(adj_mask.shape[0])
        self.register_buffer("adj_mask", adj_mask.float())
        self.diag_logit = nn.Parameter(torch.full((self.n,), 2.2))  # sigmoid ~ 0.90
        self.W = nn.Parameter(torch.zeros(self.n, self.n))
        self.B_stim = nn.Parameter(torch.full((self.n,), 0.25))
        self.kalman_logit = nn.Parameter(torch.full((self.n,), 1.0))
        self.direct_obs = bool(direct_obs)
        if not self.direct_obs:
            self.C_log = nn.Parameter(torch.zeros(self.n))
            self.d_bias = nn.Parameter(torch.zeros(self.n))

    def _emit(self, x: torch.Tensor) -> torch.Tensor:
        if self.direct_obs:
            return x
        return x * torch.exp(self.C_log.clamp(-2.0, 2.0)) + self.d_bias

    def forward_filter_and_rollout(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        stim: torch.Tensor,
        filter_until: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Kalman filter for t < filter_until, then open-loop rollout for t >= filter_until.

        Returns (pred_y_seq (B, T, N), filtered_x_hist (B, T, N)).
        """
        B, T, N = stim.shape
        a = torch.sigmoid(self.diag_logit)
        W_eff = self.W * self.adj_mask
        k_gain = torch.sigmoid(self.kalman_logit)

        x = torch.zeros((B, N), device=stim.device, dtype=stim.dtype)
        preds = []
        filt_states = []
        for t in range(T):
            y_hat = self._emit(x)
            preds.append(y_hat)
            if t < filter_until:
                mt = mask[:, t, :].to(dtype=stim.dtype)
                if self.direct_obs:
                    innov = y[:, t, :] - x
                    x = x + mt * k_gain * innov
                else:
                    c_eff = torch.exp(self.C_log.clamp(-2.0, 2.0))
                    innov = (y[:, t, :] - y_hat) / c_eff.clamp(min=0.1)
                    x = x + mt * k_gain * innov
            filt_states.append(x)
            x = x * a + (x @ W_eff) + stim[:, t, :] * self.B_stim
        return torch.stack(preds, dim=1), torch.stack(filt_states, dim=1)

    @torch.no_grad()
    def predict_all_tasks(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, pre_vols: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        B, T, N = stim.shape
        pred_open, _ = self.forward_filter_and_rollout(y, mask, stim, filter_until=pre_vols)
        pred_zero, _ = self.forward_filter_and_rollout(y, mask, torch.zeros_like(stim), filter_until=pre_vols)

        # Online causal filtering across whole T
        pred_1step, x_filt = self.forward_filter_and_rollout(y, mask, stim, filter_until=T)
        # Reconstruct true 5-step open-loop predictions: from filtered state after step t-5, step 5 times using u_{t-4..t}
        a = torch.sigmoid(self.diag_logit)
        W_eff = self.W * self.adj_mask
        pred_5step = torch.zeros_like(pred_1step)
        for t_end in range(pre_vols + 4, T):
            # Start from filtered state after observing t_end - 5, drive by stim[t_end - 5 .. t_end - 1]
            t_obs_last = t_end - 5
            x = x_filt[:, t_obs_last, :]
            for k in range(5):
                x = x * a + (x @ W_eff) + stim[:, t_obs_last + k, :] * self.B_stim
                y_h = self._emit(x)
            pred_5step[:, t_end, :] = y_h
        return (
            pred_open.cpu().numpy(),
            pred_zero.cpu().numpy(),
            pred_1step.cpu().numpy(),
            pred_5step.cpu().numpy(),
        )


def _train_pytorch_ssm(
    model: nn.Module,
    step_fn,
    tr_tensors: Tuple[torch.Tensor, ...],
    val_tensors: Tuple[torch.Tensor, ...],
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> float:
    t0 = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_tr = tr_tensors[0].shape[0]
    rng = torch.Generator().manual_seed(42)
    patience, wait = 8, 0

    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=rng)
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            batch = [t[idx] for t in tr_tensors]
            opt.zero_grad()
            loss = step_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(step_fn(model, list(val_tensors)).item())
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return float(time.time() - t0)


# ---------------------------------------------------------------------------
# Baseline 5: Dense Low-Rank Latent LDS (z_dim=32)
# ---------------------------------------------------------------------------

class DenseLowRankLDS(nn.Module):
    def __init__(self, n_neurons: int = 300, z_dim: int = 32):
        super().__init__()
        self.n = int(n_neurons)
        self.z_dim = int(z_dim)
        self.A = nn.Parameter(0.92 * torch.eye(z_dim))
        self.B_stim = nn.Linear(n_neurons, z_dim, bias=False)
        nn.init.normal_(self.B_stim.weight, std=0.05)
        self.C = nn.Linear(z_dim, n_neurons)
        nn.init.normal_(self.C.weight, std=0.05)
        nn.init.zeros_(self.C.bias)
        self.K_enc = nn.Linear(n_neurons, z_dim, bias=False)
        nn.init.normal_(self.K_enc.weight, std=0.02)

    def forward_filter_and_rollout(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, filter_until: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = stim.shape
        z = torch.zeros((B, self.z_dim), device=stim.device, dtype=stim.dtype)
        preds, filt_z = [], []
        for t in range(T):
            y_hat = self.C(z)
            preds.append(y_hat)
            if t < filter_until:
                mt = mask[:, t, :].to(dtype=stim.dtype)
                innov = (y[:, t, :] - y_hat) * mt
                z = z + torch.tanh(self.K_enc(innov)) * 0.5
            filt_z.append(z)
            z = z @ self.A.T + self.B_stim(stim[:, t, :])
        return torch.stack(preds, dim=1), torch.stack(filt_z, dim=1)

    @torch.no_grad()
    def predict_all_tasks(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, pre_vols: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        B, T, _ = stim.shape
        pred_open, _ = self.forward_filter_and_rollout(y, mask, stim, filter_until=pre_vols)
        pred_zero, _ = self.forward_filter_and_rollout(y, mask, torch.zeros_like(stim), filter_until=pre_vols)
        pred_1step, z_filt = self.forward_filter_and_rollout(y, mask, stim, filter_until=T)
        pred_5step = torch.zeros_like(pred_1step)
        for t_end in range(pre_vols + 4, T):
            t0 = t_end - 5
            z = z_filt[:, t0, :]
            for k in range(5):
                z = z @ self.A.T + self.B_stim(stim[:, t0 + k, :])
                yh = self.C(z)
            pred_5step[:, t_end, :] = yh
        return (
            pred_open.cpu().numpy(),
            pred_zero.cpu().numpy(),
            pred_1step.cpu().numpy(),
            pred_5step.cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# Baseline 6: Strictly Causal DMM (reuses Emitter, GatedTransition, Combiner from DMM.py)
# ---------------------------------------------------------------------------

class CausalDMM(nn.Module):
    """Deep Markov Model reusing DMM.py transition/emitter/combiner with causal forward GRU."""

    def __init__(self, input_dim: int = 300, z_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emitter = Emitter(input_dim, z_dim, hidden_dim)
        self.trans = GatedTransition(z_dim, hidden_dim, stim_dim=input_dim)
        self.combiner = Combiner(z_dim, hidden_dim)
        # Strictly causal forward GRU taking [y*m, m, stim]
        self.causal_rnn = nn.GRU(input_dim * 3, hidden_dim, batch_first=True)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))

    def model(self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, annealing_factor: float = 1.0):
        B, T, _ = y.shape
        pyro.module("causal_dmm", self)
        z_prev = self.z_0.expand(B, self.z_dim)
        time_mask = mask.any(dim=-1, keepdim=True)
        with pyro.plate("dmm_batch", B):
            for t in pyro.markov(range(T)):
                z_loc, z_scale = self.trans(z_prev, stim[:, t, :])
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1).mask(time_mask[:, t, 0]))
                obs_loc, obs_scale = self.emitter(z_t)
                pyro.sample(
                    f"obs_{t}",
                    dist.Normal(obs_loc, obs_scale).mask(mask[:, t, :]).to_event(1),
                    obs=y[:, t, :],
                )
                z_prev = z_t

    def guide(self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, annealing_factor: float = 1.0):
        B, T, _ = y.shape
        pyro.module("causal_dmm", self)
        m_f = mask.to(dtype=y.dtype)
        rnn_in = torch.cat([y * m_f, m_f, stim], dim=-1)
        h_seq, _ = self.causal_rnn(rnn_in)
        z_prev = self.z_q_0.expand(B, self.z_dim)
        time_mask = mask.any(dim=-1)
        with pyro.plate("dmm_batch", B):
            for t in pyro.markov(range(T)):
                z_loc, z_scale = self.combiner(z_prev, h_seq[:, t, :])
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", dist.Normal(z_loc, z_scale).to_event(1).mask(time_mask[:, t]))
                z_prev = z_t

    @torch.no_grad()
    def predict_all_tasks(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, pre_vols: int = 10, n_samples: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        B, T, _ = y.shape
        m_f = mask.to(dtype=y.dtype)
        rnn_in = torch.cat([y * m_f, m_f, stim], dim=-1)
        h_seq, _ = self.causal_rnn(rnn_in)

        # Posterior filtering trajectory q(z_t | y_{<=t}, stim_{<=t})
        z_post_means = []
        z_post_stds = []
        z_prev = self.z_q_0.expand(B, self.z_dim)
        for t in range(T):
            zl, zs = self.combiner(z_prev, h_seq[:, t, :])
            z_post_means.append(zl)
            z_post_stds.append(zs)
            z_prev = zl
        z_post_means = torch.stack(z_post_means, dim=1)  # (B, T, Z)
        z_post_stds = torch.stack(z_post_stds, dim=1)

        # Task 1 & Task 3: sample z at t = pre_vols - 1, then roll out trans(z, stim) and trans(z, 0) sharing noise!
        torch.manual_seed(42)
        open_s, zero_s = [], []
        z_pre_mean = z_post_means[:, pre_vols - 1, :]
        z_pre_std = z_post_stds[:, pre_vols - 1, :]
        for _ in range(n_samples):
            z0_s = z_pre_mean + z_pre_std * torch.randn_like(z_pre_mean)
            zs_st, zs_zr = z0_s, z0_s
            po = [self.emitter(z_post_means[:, t, :])[0] for t in range(pre_vols)]
            pz = list(po)
            for t in range(pre_vols, T):
                eps = torch.randn_like(zs_st)
                loc_s, sc_s = self.trans(zs_st, stim[:, t, :])
                loc_z, sc_z = self.trans(zs_zr, torch.zeros_like(stim[:, t, :]))
                zs_st = loc_s + sc_s * eps
                zs_zr = loc_z + sc_z * eps
                po.append(self.emitter(zs_st)[0])
                pz.append(self.emitter(zs_zr)[0])
            open_s.append(torch.stack(po, dim=1).cpu().numpy())
            zero_s.append(torch.stack(pz, dim=1).cpu().numpy())

        # Task 2 (Online 1-step & 5-step): posterior filtered mean at t-1 -> 1 trans step; t-5 -> 5 trans steps
        pred_1step = torch.zeros_like(y)
        for t in range(1, T):
            loc_1, _ = self.trans(z_post_means[:, t - 1, :], stim[:, t, :])
            pred_1step[:, t, :] = self.emitter(loc_1)[0]
        pred_5step = torch.zeros_like(y)
        for t_end in range(pre_vols + 4, T):
            z_k = z_post_means[:, t_end - 5, :]
            for k in range(1, 6):
                z_k, _ = self.trans(z_k, stim[:, t_end - 5 + k, :])
            pred_5step[:, t_end, :] = self.emitter(z_k)[0]

        return (
            np.mean(open_s, axis=0),
            np.mean(zero_s, axis=0),
            pred_1step.cpu().numpy(),
            pred_5step.cpu().numpy(),
        )


def train_causal_dmm(
    y_tr: torch.Tensor, m_tr: torch.Tensor, st_tr: torch.Tensor,
    y_va: torch.Tensor, m_va: torch.Tensor, st_va: torch.Tensor,
    epochs: int = 12, batch_size: int = 128, lr: float = 3e-3,
) -> Tuple[CausalDMM, float]:
    t0 = time.time()
    pyro.clear_param_store()
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    dmm = CausalDMM(input_dim=y_tr.shape[2], z_dim=32, hidden_dim=128)
    svi = SVI(dmm.model, dmm.guide, ClippedAdam({"lr": lr, "clip_norm": 5.0}), Trace_ELBO())
    n_tr = y_tr.shape[0]
    rng = torch.Generator().manual_seed(42)
    for ep in range(epochs):
        af = min(1.0, (ep + 1) / max(1, epochs // 2))
        perm = torch.randperm(n_tr, generator=rng)
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            svi.step(y_tr[idx], m_tr[idx], st_tr[idx], annealing_factor=af)
    return dmm, float(time.time() - t0)


# ---------------------------------------------------------------------------
# Baseline 7: Causal GRU (hidden=128)
# ---------------------------------------------------------------------------

class CausalGRU(nn.Module):
    def __init__(self, n_neurons: int = 300, hidden_dim: int = 128):
        super().__init__()
        self.n = int(n_neurons)
        self.hidden_dim = int(hidden_dim)
        self.cell = nn.GRUCell(n_neurons * 3, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_neurons)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward_filter_and_rollout(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, filter_until: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = stim.shape
        h = torch.zeros((B, self.hidden_dim), device=stim.device, dtype=stim.dtype)
        y_prev = torch.zeros((B, self.n), device=stim.device, dtype=stim.dtype)
        preds, hs, y_filts = [], [], []
        for t in range(T):
            y_hat = y_prev + self.head(h)
            preds.append(y_hat)
            if t < filter_until:
                mt = mask[:, t, :].to(dtype=stim.dtype)
                y_cur = torch.where(mask[:, t, :], y[:, t, :].clamp(-3.0, 3.0), y_hat.clamp(-3.0, 3.0))
            else:
                mt = torch.zeros((B, self.n), device=stim.device, dtype=stim.dtype)
                y_cur = y_hat.clamp(-3.0, 3.0)
            x_in = torch.cat([y_cur, mt, stim[:, t, :]], dim=-1)
            h = self.cell(x_in, h)
            y_prev = y_cur
            hs.append(h)
            y_filts.append(y_prev)
        return torch.stack(preds, dim=1), torch.stack(hs, dim=1), torch.stack(y_filts, dim=1)

    @torch.no_grad()
    def predict_all_tasks(
        self, y: torch.Tensor, mask: torch.Tensor, stim: torch.Tensor, pre_vols: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        B, T, _ = stim.shape
        pred_open, _, _ = self.forward_filter_and_rollout(y, mask, stim, filter_until=pre_vols)
        pred_zero, _, _ = self.forward_filter_and_rollout(y, mask, torch.zeros_like(stim), filter_until=pre_vols)
        pred_1step, h_hist, y_filt = self.forward_filter_and_rollout(y, mask, stim, filter_until=T)
        pred_5step = torch.zeros_like(pred_1step)
        zeros_m = torch.zeros((B, self.n), device=stim.device, dtype=stim.dtype)
        for t_end in range(pre_vols + 4, T):
            t0 = t_end - 5
            h = h_hist[:, t0, :]
            y_prev = y_filt[:, t0, :]
            y_h = y_prev + self.head(h)
            for k in range(1, 5):
                x_in = torch.cat([y_h.clamp(-3.0, 3.0), zeros_m, stim[:, t0 + k, :]], dim=-1)
                h = self.cell(x_in, h)
                y_h = y_h + self.head(h)
            pred_5step[:, t_end, :] = y_h
        return (
            pred_open.cpu().numpy(),
            pred_zero.cpu().numpy(),
            pred_1step.cpu().numpy(),
            pred_5step.cpu().numpy(),
        )


def _eval_saved_physics_ckpt(
    checkpoint_path: str, dataset: RandiTrialDataset, test_idx: List[int],
    adj: np.ndarray, np_adj: np.ndarray, use_extrasyn: bool, shuffle_seed: Optional[int], num_samples: int = 8
) -> Dict:
    t0 = time.time()
    pyro.clear_param_store()
    snap = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ms = set(snap["MODEL_STATE"].keys())
    per_neuron_stim_gain = any("stim_gain_vec" in k for k in ms)
    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=use_extrasyn,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
        shuffle_seed=shuffle_seed,
    )
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2, use_prestim_encoder=True, pre_vols=10, prestim_rank=16,
        variational_prestim=True, stim_actuator=True, stim_tau_rise=0.08, stim_tau_decay=0.4,
    )
    model = vae.VAElegans(
        rec, rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC), len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    )
    model.load_state_dict(snap["MODEL_STATE"])
    pyro.get_param_store().set_state(snap["PARAM_STORE"])
    model.eval()

    loader = torch.utils.data.DataLoader(Subset(dataset, test_idx), batch_size=len(test_idx), shuffle=False)
    np.random.seed(42); torch.manual_seed(42); pyro.set_rng_seed(42)
    m = evaluate_predictions(
        model, loader, torch.device("cpu"), guide=model.guide,
        num_samples=num_samples, pre_vols=10, adjacency=adj, np_adjacency=np_adj,
    )
    res = _summarize_metrics(m)
    res["online_filtering"] = None
    res["checkpoint"] = checkpoint_path
    res["eval_wall_s"] = float(time.time() - t0)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/randi_causal.npz")
    p.add_argument("--split-file", default=f"{RUNS_BASE}/p15_5f_s1/fold_0.json")
    p.add_argument("--out", default=os.path.join(REPO, "report/data/phase4_baselines_fold0.json"))
    p.add_argument("--epochs", default=35, type=int)
    p.add_argument("--dmm-epochs", default=10, type=int)
    p.add_argument("--num-samples", default=8, type=int)
    args = p.parse_args()

    dataset = RandiTrialDataset.load(args.cache)
    with open(args.split_file) as fh:
        split = json.load(fh)
    tr_idx = [int(i) for i in split["train"]]
    va_idx = [int(i) for i in split["val"]]
    te_idx = [int(i) for i in split["test"]]

    y_tr, m_tr, st_tr = dataset.y[tr_idx], dataset.obs_mask[tr_idx], dataset.stim_inputs[tr_idx]
    y_va, m_va, st_va = dataset.y[va_idx], dataset.obs_mask[va_idx], dataset.stim_inputs[va_idx]
    y_te, m_te, st_te = dataset.y[te_idx], dataset.obs_mask[te_idx], dataset.stim_inputs[te_idx]
    y_te_np, m_te_np, st_te_np = y_te.numpy(), m_te.numpy().astype(bool), st_te.numpy()

    intact_step = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adj = intact_step.synapse.adjacency_matrix(intact_step.neuronSize)
    np_adj = intact_step.synapse.extrasynaptic_adjacency_matrix(intact_step.neuronSize)
    shuf7_step = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"), shuffle_seed=7)
    adj_shuf7 = shuf7_step.synapse.adjacency_matrix(shuf7_step.neuronSize)

    results: Dict[str, dict] = {
        "cache": args.cache,
        "split_file": args.split_file,
        "train_trials": len(tr_idx),
        "val_trials": len(va_idx),
        "test_trials": len(te_idx),
        "models": {},
    }

    # 1 & 2: Persistence & Exp-Decay
    print("\n=== Running Persistence & Exp-Decay ===", flush=True)
    results["models"].update(run_persistence_and_expdecay(y_tr, m_tr, y_te, m_te, st_te, adj, np_adj))

    # Helper loss for PyTorch SSM / GRU (joint open-loop + 1-step)
    def _ssm_loss(mod, batch):
        yb, mb, stb = batch
        p_open = mod.forward_filter_and_rollout(yb, mb, stb, filter_until=10)[0]
        p_1 = mod.forward_filter_and_rollout(yb, mb, stb, filter_until=yb.shape[1])[0]
        return _masked_mse(p_open[:, 10:, :], yb[:, 10:, :], mb[:, 10:, :]) + 0.5 * _masked_mse(
            p_1[:, 1:, :], yb[:, 1:, :], mb[:, 1:, :]
        )

    # 3 & 4: Graph VAR + Connectome LDS (intact & shuffle7)
    for label, adj_use, direct_obs in [
        ("graph_var_intact", adj, True),
        ("graph_var_shuffle7", adj_shuf7, True),
        ("connectome_lds_intact", adj, False),
        ("connectome_lds_shuffle7", adj_shuf7, False),
    ]:
        print(f"\n=== Training {label} ===", flush=True)
        torch.manual_seed(42)
        mod = ConnectomeLinearSSM(torch.from_numpy(adj_use), direct_obs=direct_obs)
        wt = _train_pytorch_ssm(mod, _ssm_loss, (y_tr, m_tr, st_tr), (y_va, m_va, st_va), epochs=args.epochs, lr=1e-2)
        po, pz, p1, p5 = mod.predict_all_tasks(y_te, m_te, st_te, pre_vols=10)
        res = _pack_model_result(y_te_np, m_te_np, st_te_np, po, pz, p1, p5, adj, np_adj, wt)
        results["models"][label] = res
        print(f"  {label}: open post_inlier_r2={res['open_loop'].get('post_inlier_r2'):.4f} "
              f"1step_inlier_r2={res['online_filtering']['step1_0.5s']['inlier_r2']:.4f} "
              f"evoked_hop1_r2={res['intervention'].get('evoked_hop1_r2', 0):.5f} ({wt:.1f}s)", flush=True)

    # 5: Dense Low-Rank LDS
    print("\n=== Training dense_lowrank_lds (z_dim=32) ===", flush=True)
    torch.manual_seed(42)
    d_lds = DenseLowRankLDS(n_neurons=300, z_dim=32)
    wt = _train_pytorch_ssm(d_lds, _ssm_loss, (y_tr, m_tr, st_tr), (y_va, m_va, st_va), epochs=args.epochs, lr=3e-3)
    po, pz, p1, p5 = d_lds.predict_all_tasks(y_te, m_te, st_te, pre_vols=10)
    res = _pack_model_result(y_te_np, m_te_np, st_te_np, po, pz, p1, p5, adj, np_adj, wt)
    results["models"]["dense_lowrank_lds"] = res
    print(f"  dense_lowrank_lds: open post_inlier_r2={res['open_loop'].get('post_inlier_r2'):.4f} "
          f"1step_inlier_r2={res['online_filtering']['step1_0.5s']['inlier_r2']:.4f} ({wt:.1f}s)", flush=True)

    # 6: Strictly Causal DMM
    print("\n=== Training dmm_svi (z_dim=32, causal combiner) ===", flush=True)
    dmm, wt = train_causal_dmm(y_tr, m_tr, st_tr, y_va, m_va, st_va, epochs=args.dmm_epochs)
    po, pz, p1, p5 = dmm.predict_all_tasks(y_te, m_te, st_te, pre_vols=10, n_samples=args.num_samples)
    res = _pack_model_result(y_te_np, m_te_np, st_te_np, po, pz, p1, p5, adj, np_adj, wt)
    results["models"]["dmm_svi"] = res
    print(f"  dmm_svi: open post_inlier_r2={res['open_loop'].get('post_inlier_r2'):.4f} "
          f"1step_inlier_r2={res['online_filtering']['step1_0.5s']['inlier_r2']:.4f} ({wt:.1f}s)", flush=True)

    # 7: Causal GRU
    print("\n=== Training causal_gru (hidden=128) ===", flush=True)
    torch.manual_seed(42)
    gru = CausalGRU(n_neurons=300, hidden_dim=128)
    wt = _train_pytorch_ssm(gru, _ssm_loss, (y_tr, m_tr, st_tr), (y_va, m_va, st_va), epochs=args.epochs, lr=3e-3)
    po, pz, p1, p5 = gru.predict_all_tasks(y_te, m_te, st_te, pre_vols=10)
    res = _pack_model_result(y_te_np, m_te_np, st_te_np, po, pz, p1, p5, adj, np_adj, wt)
    results["models"]["causal_gru"] = res
    print(f"  causal_gru: open post_inlier_r2={res['open_loop'].get('post_inlier_r2'):.4f} "
          f"1step_inlier_r2={res['online_filtering']['step1_0.5s']['inlier_r2']:.4f} ({wt:.1f}s)", flush=True)

    # 8: Saved Fold-0 Phase 1.5 S2 & Phase 2.5 S2 (intact + shuffle7)
    for label, ckpt, ex, sh in [
        ("phase15_f0_s2_intact", f"{RUNS_BASE}/p15_5f_s2/fold0/intact_seed42/vae_epoch_10.pt", False, None),
        ("phase15_f0_s2_shuffle7", f"{RUNS_BASE}/p15_5f_s2/fold0/shuffle7_seed42/vae_epoch_10.pt", False, 7),
        ("phase25_f0_s2_intact", f"{RUNS_BASE}/p25_f0_s2/fold0/intact_seed42/vae_epoch_10.pt", True, None),
        ("phase25_f0_s2_shuffle7", f"{RUNS_BASE}/p25_f0_s2/fold0/shuffle7_seed42/vae_epoch_10.pt", True, 7),
    ]:
        print(f"\n=== Re-evaluating {label} ===", flush=True)
        res = _eval_saved_physics_ckpt(ckpt, dataset, te_idx, adj, np_adj, ex, sh, num_samples=args.num_samples)
        results["models"][label] = res
        print(f"  {label}: open post_inlier_r2={res['open_loop'].get('post_inlier_r2'):.4f} "
              f"evoked_hop1_r2={res['intervention'].get('evoked_hop1_r2', 0):.5f} "
              f"kernel_hop1_r={res['intervention'].get('kernel_hop1_r', 0):.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
