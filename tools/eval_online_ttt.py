#!/usr/bin/env python3
"""Causal prequential evaluation for Online Bayesian Test-Time Training (TTT).

Evaluates a pretrained biophysical connectome VAE checkpoint on unseen worms by
adapting a tiered subset of individual worm parameters strictly from past,
non-overlapping trials of the same worm while anchoring the adapted posterior
q_worm(theta) to the pretrained population posterior q_pop(theta) via an exact
diagonal KL penalty:

    L_TTT = NLL_post(past_trials; theta_worm) + kl_weight * KL(q_worm || q_pop)

Guarantees:
  1. Strictly chronological per-worm ordering by `stim_vol` (no future leakage).
  2. Non-overlap guard: a past trial `tau` is only eligible if
     `stim_vol[tau] + post_vols <= stim_vol[target]` (strictly causal at the
     prediction decision point `t = target_vol`; pass `strict_disjoint_pre=True`
     for `stim_vol[tau] + post_vols <= stim_vol[target] - pre_vols`).
  3. Full bitwise reset to `q_pop` between every worm (zero cross-worm leakage).
  4. Optional paired intact vs. shuffled connectome TTT control.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_c302 import (
    compute_drift_gate,
    compute_stratified_metrics,
    extract_and_expand_posterior_samples,
)
from pyro_svi_vae import reweight_hop1_noise
from utils import RandiTrialDataset


def group_and_order_trials_by_worm(
    meta: Sequence[dict],
    indices: Optional[Sequence[int]] = None,
) -> Dict[int, List[int]]:
    """Group trial indices by `rec_id` and sort chronologically within each worm."""
    idx_list = list(range(len(meta))) if indices is None else [int(i) for i in indices]
    by_worm: Dict[int, List[int]] = {}
    for idx in idx_list:
        rec_id = int(meta[idx].get("rec_id", 0))
        by_worm.setdefault(rec_id, []).append(idx)

    for rec_id, worm_indices in by_worm.items():
        worm_indices.sort(
            key=lambda i: (
                int(meta[i].get("stim_vol", meta[i].get("event_idx", i))),
                int(meta[i].get("event_idx", i)),
                int(i),
            )
        )
    return dict(sorted(by_worm.items(), key=lambda kv: kv[0]))


def select_causal_history_indices(
    ordered_indices: Sequence[int],
    target_pos: int,
    meta: Sequence[dict],
    post_vols: int,
    max_history: Optional[int] = None,
    pre_vols: int = 0,
    strict_disjoint_pre: bool = False,
) -> List[int]:
    """Return strictly past, non-overlapping trial indices for `ordered_indices[target_pos]`."""
    if target_pos <= 0:
        return []
    target_idx = int(ordered_indices[target_pos])
    target_meta = meta[target_idx]
    has_target_vol = "stim_vol" in target_meta and target_meta["stim_vol"] is not None
    target_vol = int(target_meta["stim_vol"]) if has_target_vol else None
    cutoff_vol = (
        (target_vol - int(pre_vols) if strict_disjoint_pre else target_vol)
        if target_vol is not None
        else None
    )

    eligible: List[int] = []
    for pos in range(target_pos):
        h_idx = int(ordered_indices[pos])
        h_meta = meta[h_idx]
        if cutoff_vol is not None and ("stim_vol" in h_meta and h_meta["stim_vol"] is not None):
            h_vol = int(h_meta["stim_vol"])
            if h_vol + int(post_vols) > cutoff_vol:
                continue
        eligible.append(h_idx)

    if max_history is not None and int(max_history) > 0 and len(eligible) > int(max_history):
        eligible = eligible[-int(max_history) :]
    return eligible


def _resolve_raw_param(guide: torch.nn.Module, attr_name: str) -> Optional[torch.nn.Parameter]:
    """Resolve the underlying trainable torch.nn.Parameter for a PyroParam attribute."""
    raw_name = f"{attr_name}_unconstrained"
    if hasattr(guide, raw_name):
        p = getattr(guide, raw_name)
        if isinstance(p, torch.nn.Parameter):
            return p
    if hasattr(guide, attr_name):
        p = getattr(guide, attr_name)
        if isinstance(p, torch.nn.Parameter):
            return p
    return None


def _get_tier_anchor_specs(
    model: vae.VAElegans,
    tier: str,
    adapt_obs_sigma: bool = False,
    ttt_mode: str = "auto",
) -> List[Tuple[str, torch.nn.Parameter, torch.Tensor]]:
    """Return list of (name, trainable_param, pop_sigma) for the requested TTT mode/tier."""
    if tier not in ("tier1", "tier2", "tier3"):
        raise ValueError(f"Unsupported TTT tier={tier!r}; expected 'tier1', 'tier2', or 'tier3'.")
    if ttt_mode not in ("auto", "ttt_a", "ttt_b", "ttt_c", "ttt_d", "ttt_e", "ttt_f"):
        raise ValueError(f"Unsupported ttt_mode={ttt_mode!r}.")

    guide = model.guide
    specs: List[Tuple[str, torch.nn.Parameter, torch.Tensor]] = []

    def _add(loc_attr: str, scale_tensor: Optional[torch.Tensor], coord_mult: float = 1.0) -> None:
        raw_p = _resolve_raw_param(guide, loc_attr)
        if raw_p is None:
            return
        if scale_tensor is None:
            sig = torch.full_like(raw_p.detach(), 0.25 / max(coord_mult, 1e-6))
        else:
            s = scale_tensor.detach().to(device=raw_p.device, dtype=raw_p.dtype)
            if s.shape != raw_p.shape:
                s = torch.full_like(raw_p.detach(), float(s.mean().item()))
            sig = torch.clamp(s / max(coord_mult, 1e-6), min=1e-4)
        specs.append((loc_attr, raw_p, sig.clone()))

    eff = tier if ttt_mode == "auto" else ttt_mode
    include_gcamp = eff in ("tier1", "tier2", "tier3", "ttt_a", "ttt_c", "ttt_d", "ttt_e", "ttt_f")
    include_stim = eff in ("tier1", "tier2", "tier3", "ttt_b", "ttt_c", "ttt_d", "ttt_e", "ttt_f")
    include_global_syn = eff in ("tier2", "ttt_e", "ttt_f")
    include_excitability = eff in ("tier2", "tier3", "ttt_f")

    if include_gcamp:
        _add("gcamp_gain_loc", getattr(guide, "gcamp_gain_scale", None))
        _add("gcamp_offset_loc", getattr(guide, "gcamp_offset_scale", None))
    if include_stim:
        if getattr(guide, "per_neuron_stim_gain", False):
            _add("stim_gain_vec_loc", getattr(guide, "stim_gain_vec_scale", None))
        else:
            _add("stim_gain_loc", getattr(guide, "stim_gain_scale", None))
    if adapt_obs_sigma:
        _add("obs_sigma_loc", getattr(guide, "obs_sigma_scale", None))

    if getattr(guide, "direct_global_guide", False):
        if include_global_syn:
            if getattr(guide, "wicks_dim", 0) > 0:
                _add("wicks_g_max_global_z", getattr(guide, "wicks_g_max_scale", None), coord_mult=1.5)
            if getattr(guide, "gap_dim", 0) > 0:
                _add("gap_g_syn_global_z", getattr(guide, "gap_g_syn_scale", None), coord_mult=1.5)
        elif eff == "tier3" and getattr(guide, "gap_dim", 0) > 0:
            _add("gap_g_syn_global_z", getattr(guide, "gap_g_syn_scale", None), coord_mult=1.5)

        if include_excitability:
            _add("neuron_E_z", getattr(guide, "neuron_E_scale", None), coord_mult=0.01)
            _add("wicks_V_rest_z", getattr(guide, "wicks_V_rest_scale", None), coord_mult=0.01)
            _add("wicks_V_slope_loc", getattr(guide, "wicks_V_slope_scale", None), coord_mult=1.0)
            _add("neuron_C_loc", getattr(guide, "neuron_C_scale", None), coord_mult=1.0)

        if eff == "tier3" and getattr(guide, "wicks_dim", 0) > 0:
            _add("wicks_g_max_z", getattr(guide, "wicks_g_max_scale", None), coord_mult=3.0)
            _add("wicks_E_rev_z", getattr(guide, "wicks_E_rev_scale", None), coord_mult=0.04)

    return specs


def _build_worm_latent_edge_basis(
    model: vae.VAElegans, worm_latent_dim: int, device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Construct and cache low-rank graph-spectral basis U_wicks, U_gap for worm identity latent b_w."""
    cache = getattr(model, "_worm_latent_basis_cache", None)
    if cache is None:
        cache = {}
        model._worm_latent_basis_cache = cache
    key = (int(worm_latent_dim), str(device), str(dtype))
    if key in cache:
        return cache[key]

    syn = model.decoder.model.synapse
    n_neu = int(getattr(model.decoder.model, "neuronSize", 300))
    adj_np = np.asarray(syn.adjacency_matrix(n_neu), dtype=np.float64)
    adj_sym = 0.5 * (adj_np + adj_np.T)
    try:
        u_node, _, _ = np.linalg.svd(adj_sym, full_matrices=False)
    except np.linalg.LinAlgError:
        u_node = np.eye(n_neu, dtype=np.float64)
    d_w = int(worm_latent_dim)
    k_eff = min(d_w, u_node.shape[1])
    u_k = np.zeros((n_neu, d_w), dtype=np.float64)
    u_k[:, :k_eff] = u_node[:, :k_eff]
    u_t = torch.as_tensor(u_k, device=device, dtype=dtype)

    out: Dict[str, torch.Tensor] = {}
    if getattr(model.guide, "wicks_dim", 0) > 0:
        w_src = torch.as_tensor(syn.Wicks_SRC, device=device, dtype=torch.long)
        w_dst = torch.as_tensor(syn.Wicks_DST, device=device, dtype=torch.long)
        uw = (u_t[w_src] + u_t[w_dst]) / math.sqrt(2.0)
        rms = torch.sqrt(torch.mean(uw * uw, dim=0, keepdim=True)).clamp(min=1e-6)
        out["U_wicks"] = uw / rms
    if getattr(model.guide, "gap_dim", 0) > 0:
        g_src = torch.as_tensor(syn.Gap_Junction_SRC, device=device, dtype=torch.long)
        g_dst = torch.as_tensor(syn.Gap_Junction_DST, device=device, dtype=torch.long)
        ug = (u_t[g_src] + u_t[g_dst]) / math.sqrt(2.0)
        rms = torch.sqrt(torch.mean(ug * ug, dim=0, keepdim=True)).clamp(min=1e-6)
        out["U_gap"] = ug / rms
    cache[key] = out
    return out


def snapshot_population_state(model: vae.VAElegans) -> dict:
    """Deep-copy model state_dict for exact per-worm resets."""
    return {
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def restore_population_state(model: vae.VAElegans, pop_snapshot: dict) -> None:
    """Restore model bitwise in-place and clear ParamStore so Pyro re-binds live nn.Parameters."""
    pyro.clear_param_store()
    snap_sd = pop_snapshot["model_state"]
    copied = set()
    with torch.no_grad():
        for name, param in model.named_parameters():
            src = snap_sd.get(name)
            if src is not None:
                param.copy_(src)
                copied.add(name)
        for name, buf in model.named_buffers():
            src = snap_sd.get(name)
            if src is not None:
                buf.copy_(src)
                copied.add(name)
    if len(copied) != len(snap_sd):
        model.load_state_dict({k: v.detach().clone() for k, v in snap_sd.items()})


def _posterior_median_from_guide(guide, guide_kwargs: dict) -> Dict[str, torch.Tensor]:
    """Extract differentiable posterior medians from guide without Monte Carlo noise."""
    gtr = pyro.poutine.trace(guide).get_trace(**guide_kwargs)
    ps: Dict[str, torch.Tensor] = {}
    for name, site in gtr.nodes.items():
        if site["type"] != "sample" or name == "batch":
            continue
        fn = site["fn"]
        while isinstance(fn, dist.Independent):
            fn = fn.base_dist
        if isinstance(fn, (dist.Normal, torch.distributions.Normal)):
            ps[name] = fn.loc
        elif isinstance(fn, (dist.LogNormal, torch.distributions.LogNormal)):
            ps[name] = torch.exp(fn.loc)
        else:
            ps[name] = site["value"]
    return ps


def _student_t_or_normal_nll(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    obs_mask: torch.Tensor,
    scale: torch.Tensor,
    likelihood: str = "student_t",
    student_t_df: float = 4.0,
) -> torch.Tensor:
    """Compute masked negative log-likelihood summed over valid observations."""
    m = obs_mask.to(dtype=y_pred.dtype)
    err = (y_true - y_pred) / torch.clamp(scale, min=1e-4)
    if likelihood == "student_t":
        df = float(student_t_df)
        nll = (
            0.5 * (df + 1.0) * torch.log1p((err * err) / df)
            + torch.log(torch.clamp(scale, min=1e-4))
        )
    else:
        nll = 0.5 * (err * err) + torch.log(torch.clamp(scale, min=1e-4))
    return (nll * m).sum()


def adapt_worm_posterior(
    model: vae.VAElegans,
    stim_hist: torch.Tensor,
    y_hist: torch.Tensor,
    mask_hist: torch.Tensor,
    noise_hist: torch.Tensor,
    tier: str = "tier1",
    ttt_mode: str = "auto",
    worm_latent_dim: int = 8,
    temporal_val_frac: float = 0.0,
    ttt_steps: int = 8,
    ttt_lr: float = 5e-2,
    kl_weight: float = 1.0,
    nll_temp: float = 1.0,
    adapt_obs_sigma: bool = False,
    pre_vols: int = 10,
    stim_loss_weight: float = 1.0,
    hop1_loss_weight: float = 1.0,
    adjacency_t: Optional[torch.Tensor] = None,
    drift_gate: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Adapt worm-specific parameters in-place in whitened q_pop coordinates (u = (mu - mu0)/sigma0)."""
    eff_mode = tier if ttt_mode == "auto" else ttt_mode
    if ttt_steps <= 0 or stim_hist.shape[0] == 0:
        return {
            "n_hist_trials": int(stim_hist.shape[0]),
            "ttt_steps": 0,
            "selected_step": 0,
            "loss_start": float("nan"),
            "loss_end": float("nan"),
            "kl_final": 0.0,
            "max_param_shift": 0.0,
            "max_whitened_shift": 0.0,
        }

    anchor_specs = _get_tier_anchor_specs(
        model, tier=tier, adapt_obs_sigma=bool(adapt_obs_sigma), ttt_mode=ttt_mode
    )
    use_worm_latent = (eff_mode == "ttt_d") and bool(getattr(model.guide, "direct_global_guide", False))
    if not anchor_specs and not use_worm_latent:
        return {
            "n_hist_trials": int(stim_hist.shape[0]),
            "ttt_steps": 0,
            "selected_step": 0,
            "loss_start": float("nan"),
            "loss_end": float("nan"),
            "kl_final": 0.0,
            "max_param_shift": 0.0,
            "max_whitened_shift": 0.0,
        }

    # Chronological temporal validation split on past trials (no future leakage)
    k_tot = int(stim_hist.shape[0])
    n_val = max(1, int(round(k_tot * float(temporal_val_frac)))) if (float(temporal_val_frac) > 0.0 and k_tot >= 4) else 0
    if n_val >= k_tot:
        n_val = 0
    n_tr = k_tot - n_val

    # Freeze all parameters, then unfreeze only the selected mode parameters.
    prev_req_grad = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    anchors: List[Tuple[str, torch.nn.Parameter, torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
    whitened_params: List[torch.nn.Parameter] = []
    managed_params: List[torch.nn.Parameter] = []
    for name, param, sig in anchor_specs:
        param.requires_grad_(True)
        u_param = torch.nn.Parameter(torch.zeros_like(param.detach()))
        anchors.append((name, param, u_param, param.detach().clone(), sig))
        whitened_params.append(u_param)
        managed_params.append(param)

    worm_state: Optional[dict] = None
    gen_worm_bw: Optional[torch.nn.Parameter] = None
    gen_worm_dim = int(getattr(model.decoder, "worm_latent_dim", 0))
    if use_worm_latent and gen_worm_dim > 0:
        gen_worm_bw = torch.nn.Parameter(
            torch.zeros((1, gen_worm_dim), device=stim_hist.device, dtype=stim_hist.dtype)
        )
        whitened_params.append(gen_worm_bw)
        managed_params.append(gen_worm_bw)
    elif use_worm_latent:
        basis = _build_worm_latent_edge_basis(
            model, worm_latent_dim=int(worm_latent_dim), device=stim_hist.device, dtype=stim_hist.dtype
        )
        b_w = torch.nn.Parameter(torch.zeros(int(worm_latent_dim), device=stim_hist.device, dtype=stim_hist.dtype))
        edge_bindings = []
        for attr, bkey, sc_attr, coord_mult in [
            ("wicks_g_max_z", "U_wicks", "wicks_g_max_scale", 3.0),
            ("gap_g_syn_z", "U_gap", "gap_g_syn_scale", 1.5),
        ]:
            raw_e = _resolve_raw_param(model.guide, attr)
            if raw_e is not None and bkey in basis:
                raw_e.requires_grad_(True)
                sc_t = getattr(model.guide, sc_attr, None)
                sig_e = torch.clamp(sc_t.detach().to(device=raw_e.device, dtype=raw_e.dtype) / coord_mult, min=1e-4)
                edge_bindings.append((raw_e, raw_e.detach().clone(), sig_e, basis[bkey]))
                managed_params.append(raw_e)
        if edge_bindings:
            whitened_params.append(b_w)
            worm_state = {"b_w": b_w, "edge_bindings": edge_bindings, "mod_scale": 0.25}

    active_wicks_mask: Optional[torch.Tensor] = None
    if eff_mode == "tier3" and hasattr(model.decoder.model.synapse, "Wicks_SRC"):
        wicks_src = torch.as_tensor(
            model.decoder.model.synapse.Wicks_SRC, device=stim_hist.device, dtype=torch.long
        )
        stim_sources = (stim_hist[:n_tr].abs().sum(dim=(0, 1)) > 0)
        active_wicks_mask = stim_sources[wicks_src].to(dtype=stim_hist.dtype)

    dec = model.decoder
    has_prestim = bool(getattr(dec, "use_prestim_encoder", False))
    ssm_mod = getattr(dec, "latent_ssm", None)
    has_ssm = ssm_mod is not None
    has_history = has_prestim or has_ssm
    total_ssm_dim = int(getattr(ssm_mod, "total_z_dim", getattr(ssm_mod, "z_dim", 16))) if has_ssm else 0
    likelihood = str(getattr(dec, "likelihood", "student_t"))
    student_t_df = float(getattr(dec, "student_t_df", 4.0))

    t_cut = min(int(pre_vols), y_hist.shape[1])

    eff_noise_all = noise_hist
    if (stim_loss_weight != 1.0 or hop1_loss_weight != 1.0) and adjacency_t is not None:
        eff_noise_all = reweight_hop1_noise(
            noise_hist, stim_hist, adjacency_t,
            weight=float(hop1_loss_weight), stim_weight=float(stim_loss_weight),
        )

    post_mask_all = torch.zeros_like(mask_hist, dtype=torch.bool)
    if t_cut < y_hist.shape[1]:
        post_mask_all[:, t_cut:, :] = mask_hist[:, t_cut:, :].bool()
    else:
        post_mask_all = mask_hist.bool()

    def _build_kwargs(sl: slice, noise_sl: torch.Tensor) -> Tuple[dict, dict]:
        s_sl = stim_hist[sl]
        y_sl = y_hist[sl]
        m_sl = mask_hist[sl]
        g_kw = dict(ExternalInput=s_sl, mask=None, y=None, noise_scale=noise_sl)
        if has_prestim:
            g_kw["pre_stim_obs"] = y_sl[:, :t_cut, :]
            g_kw["pre_stim_mask"] = m_sl[:, :t_cut, :]
            if drift_gate is not None:
                g_kw["drift_gate"] = drift_gate[sl] if drift_gate.shape[0] == k_tot else drift_gate
        m_kw = dict(g_kw)
        if has_ssm and not has_prestim:
            m_kw["pre_stim_obs"] = y_sl[:, :t_cut, :]
            m_kw["pre_stim_mask"] = m_sl[:, :t_cut, :]
        if has_ssm:
            m_kw["ssm_eps"] = torch.zeros(
                (s_sl.shape[0], s_sl.shape[1] + 1, total_ssm_dim),
                device=s_sl.device,
                dtype=s_sl.dtype,
            )
        if gen_worm_bw is not None:
            m_kw["worm_b_override"] = gen_worm_bw
        m_kw["sample_obs"] = False
        return g_kw, m_kw

    guide_kw_tr, model_kw_tr = _build_kwargs(slice(0, n_tr), eff_noise_all[:n_tr])
    if n_val > 0:
        guide_kw_va, model_kw_va = _build_kwargs(slice(n_tr, k_tot), noise_hist[n_tr:k_tot])

    def _eval_val_nll() -> float:
        with torch.no_grad():
            ps_v = _posterior_median_from_guide(model.guide, guide_kw_va)
            yp_v = pyro.condition(model, data=ps_v)(**model_kw_va)
            obs_s = torch.clamp(ps_v["decoder.obs_sigma"], min=1e-3)
            sc_v = torch.sqrt((obs_s * 0.5) ** 2 + noise_hist[n_tr:k_tot].unsqueeze(1) ** 2)
            vn = _student_t_or_normal_nll(
                y_hist[n_tr:k_tot], yp_v, post_mask_all[n_tr:k_tot], sc_v,
                likelihood=likelihood, student_t_df=student_t_df,
            )
            return float(vn.item())

    opt = torch.optim.Adam(whitened_params, lr=float(ttt_lr))
    loss_start = float("nan")
    loss_end = float("nan")
    kl_final = 0.0
    selected_step = int(ttt_steps)
    val_nll_start = float("nan")
    best_val_nll = float("inf")
    best_state: Optional[List[torch.Tensor]] = None

    gen_state_cpu = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    pyro.set_rng_seed(int(seed))

    try:
        if n_val > 0:
            val_nll_start = _eval_val_nll()
            best_val_nll = val_nll_start
            selected_step = 0
            best_state = [p.detach().clone() for p in managed_params]

        for step_idx in range(int(ttt_steps)):
            opt.zero_grad(set_to_none=True)
            for p in managed_params:
                p.grad = None

            ps = _posterior_median_from_guide(model.guide, guide_kw_tr)
            y_pred = pyro.condition(model, data=ps)(**model_kw_tr)

            obs_sigma = torch.clamp(ps["decoder.obs_sigma"], min=1e-3)
            scale = torch.sqrt((obs_sigma * 0.5) ** 2 + eff_noise_all[:n_tr].unsqueeze(1) ** 2)
            nll = _student_t_or_normal_nll(
                y_hist[:n_tr], y_pred, post_mask_all[:n_tr], scale,
                likelihood=likelihood, student_t_df=student_t_df,
            )

            kl = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
            for _, param, _, pop_val, pop_sig in anchors:
                diff = (param - pop_val) / pop_sig
                kl = kl + 0.5 * (diff * diff).sum()
            if gen_worm_bw is not None:
                kl = kl + 0.5 * (gen_worm_bw * gen_worm_bw).sum()
            if worm_state is not None:
                bw = worm_state["b_w"]
                kl = kl + 0.5 * (bw * bw).sum()

            loss = float(nll_temp) * nll + float(kl_weight) * kl
            loss.backward()

            for name, param, u_param, _, pop_sig in anchors:
                if param.grad is None:
                    continue
                g_u = param.grad.detach() * pop_sig
                if active_wicks_mask is not None and name in ("wicks_g_max_z", "wicks_E_rev_z"):
                    g_u = g_u * active_wicks_mask
                u_param.grad = g_u

            if worm_state is not None:
                bw = worm_state["b_w"]
                ms = float(worm_state["mod_scale"])
                g_bw = torch.zeros_like(bw.detach())
                for raw_e, _, sig_e, basis_e in worm_state["edge_bindings"]:
                    if raw_e.grad is not None:
                        g_bw = g_bw + basis_e.T @ (ms * raw_e.grad.detach() * sig_e)
                bw.grad = g_bw

            torch.nn.utils.clip_grad_norm_(whitened_params, max_norm=5.0)
            opt.step()

            with torch.no_grad():
                for _, param, u_param, pop_val, pop_sig in anchors:
                    param.copy_(pop_val + pop_sig * u_param)
                if worm_state is not None:
                    bw = worm_state["b_w"]
                    ms = float(worm_state["mod_scale"])
                    for raw_e, base_e, sig_e, basis_e in worm_state["edge_bindings"]:
                        raw_e.copy_(base_e + sig_e * (ms * (basis_e @ bw)))

            loss_val = float(loss.detach().item())
            if step_idx == 0:
                loss_start = loss_val
            loss_end = loss_val
            kl_final = float(kl.detach().item())

            if n_val > 0:
                vn = _eval_val_nll()
                if vn < best_val_nll - 1e-6:
                    best_val_nll = vn
                    selected_step = step_idx + 1
                    best_state = [p.detach().clone() for p in managed_params]

        if n_val > 0 and best_state is not None:
            with torch.no_grad():
                for p, sv in zip(managed_params, best_state):
                    p.copy_(sv)
    finally:
        torch.random.set_rng_state(gen_state_cpu)
        for p in model.parameters():
            p.requires_grad_(prev_req_grad.get(id(p), True))

    shifts = [float((param.detach() - pop_val).abs().max().item()) for _, param, _, pop_val, _ in anchors]
    wshifts = [float(u_param.detach().abs().max().item()) for _, _, u_param, _, _ in anchors]
    if gen_worm_bw is not None:
        gbw = float(gen_worm_bw.detach().abs().max().item())
        wshifts.append(gbw)
        shifts.append(gbw)
    if worm_state is not None:
        bw_max = float(worm_state["b_w"].detach().abs().max().item())
        wshifts.append(bw_max)
        for raw_e, base_e, _, _ in worm_state["edge_bindings"]:
            shifts.append(float((raw_e.detach() - base_e).abs().max().item()))
    out_log = {
        "n_hist_trials": k_tot,
        "n_train_hist": int(n_tr),
        "n_val_hist": int(n_val),
        "ttt_steps": int(ttt_steps),
        "selected_step": int(selected_step),
        "loss_start": loss_start,
        "loss_end": loss_end,
        "val_nll_start": val_nll_start,
        "val_nll_best": float(best_val_nll) if n_val > 0 else float("nan"),
        "kl_final": kl_final,
        "max_param_shift": max(shifts) if shifts else 0.0,
        "max_whitened_shift": max(wshifts) if wshifts else 0.0,
    }
    if gen_worm_bw is not None:
        out_log["_worm_b_tensor"] = gen_worm_bw.detach().clone()
    return out_log


def _predict_single_trial_counterfactual(
    model: vae.VAElegans,
    stim: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
    num_samples: int,
    pre_vols: int,
    eval_seed: int,
    drift_gate: Optional[torch.Tensor] = None,
    worm_b_override: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict (y_pred, y_pred_cf) for a single trial batch (1, T, N) with paired counterfactual."""
    device = stim.device
    dec = model.decoder
    has_prestim = bool(getattr(dec, "use_prestim_encoder", False))
    ssm_mod = getattr(dec, "latent_ssm", None)
    has_ssm = ssm_mod is not None
    has_history = has_prestim or has_ssm

    t_cut = min(int(pre_vols), y.shape[1])
    pre_obs = y[:, :t_cut, :] if has_history else None
    pre_mask = mask[:, :t_cut, :] if has_history else None

    guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_prestim:
        guide_kw["pre_stim_obs"] = pre_obs
        guide_kw["pre_stim_mask"] = pre_mask
        if drift_gate is not None:
            guide_kw["drift_gate"] = drift_gate

    stacked_kw = dict(
        ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
        mask=None,
        y=None,
        noise_scale=torch.cat([noise, noise], dim=0),
        sample_obs=False,
    )
    if has_history:
        stacked_kw["pre_stim_obs"] = torch.cat([pre_obs, pre_obs], dim=0)
        stacked_kw["pre_stim_mask"] = torch.cat([pre_mask, pre_mask], dim=0)
    if has_prestim and drift_gate is not None:
        stack_dim = 1 if drift_gate.dim() == 3 else 0
        stacked_kw["drift_gate"] = torch.cat([drift_gate, drift_gate], dim=stack_dim)
    if worm_b_override is not None:
        bw_t = worm_b_override.to(device=stim.device, dtype=stim.dtype)
        if bw_t.dim() == 1:
            bw_t = bw_t.unsqueeze(0)
        stacked_kw["worm_b_override"] = torch.cat([bw_t, bw_t], dim=0)

    bsz = stim.shape[0]
    torch.manual_seed(int(eval_seed))
    pyro.set_rng_seed(int(eval_seed))
    ps_acc = None
    pz_acc = None
    with torch.no_grad():
        for _ in range(max(1, int(num_samples))):
            gtr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
            ps = extract_and_expand_posterior_samples(gtr, repeat_factor=2)
            if has_ssm:
                total_z_dim = int(getattr(ssm_mod, "total_z_dim", ssm_mod.z_dim))
                eps_draw = torch.randn(
                    (bsz, stim.shape[1] + 1, total_z_dim),
                    device=device,
                    dtype=stim.dtype,
                )
                both = pyro.condition(model, data=ps)(
                    **stacked_kw, ssm_eps=torch.cat([eps_draw, eps_draw], dim=0)
                )
            else:
                both = pyro.condition(model, data=ps)(**stacked_kw)
            s_out, z_out = both.chunk(2, dim=0)
            ps_acc = s_out if ps_acc is None else (ps_acc + s_out)
            pz_acc = z_out if pz_acc is None else (pz_acc + z_out)

    pred_mean = (ps_acc / max(1, int(num_samples))).cpu().numpy()
    pred_cf = (pz_acc / max(1, int(num_samples))).cpu().numpy()
    return pred_mean, pred_cf


def _summarize_metrics(m: Dict[str, float]) -> Dict[str, float]:
    keep = (
        "r2", "inlier_r2", "post_r2", "post_inlier_r2", "post_inlier_r",
        "evoked_post_r2", "evoked_post_r",
        "evoked_stim_r2", "evoked_stim_r", "stim_r",
        "evoked_hop1_r2", "evoked_hop1_r", "evoked_hop1_amp_ratio",
        "evoked_hop2_r2", "evoked_hop2_r",
        "pair_evoked_hop1_count", "pair_evoked_hop1_r", "pair_evoked_hop1_r2", "pair_evoked_hop1_sign_acc",
        "kernel_hop1_n_pairs", "kernel_hop1_r", "kernel_hop1_r2", "kernel_hop1_auc_r",
        "kernel_hop1_peak_latency_mae_s", "kernel_hop1_fft_r",
        "evoked_np_only_r2", "evoked_np_only_r",
        "evoked_np_no_hop12_r2", "evoked_np_no_hop12_r",
    )
    return {k: m[k] for k in keep if k in m}


def evaluate_online_ttt(
    model: vae.VAElegans,
    dataset: RandiTrialDataset,
    test_indices: Sequence[int],
    device: torch.device,
    adjacency: np.ndarray,
    np_adjacency: Optional[np.ndarray] = None,
    ma_adjacency: Optional[np.ndarray] = None,
    tier: str = "tier1",
    ttt_mode: str = "auto",
    worm_latent_dim: int = 8,
    temporal_val_frac: float = 0.0,
    ttt_steps: int = 8,
    ttt_lr: float = 5e-2,
    kl_weight: float = 1.0,
    nll_temp: float = 1.0,
    adapt_obs_sigma: bool = False,
    min_history: int = 0,
    k_tier_schedule: str = "none",
    pre_vols: int = 10,
    post_vols: Optional[int] = None,
    max_history: Optional[int] = None,
    strict_disjoint_pre: bool = False,
    stim_loss_weight: float = 1.0,
    hop1_loss_weight: float = 1.0,
    drift_gate_mode: str = "none",
    num_samples: int = 8,
    eval_seed: int = 42,
    history_bins: Sequence[Tuple[int, int]] = ((0, 0), (1, 3), (4, 8), (9, 999)),
    zero_shot_batch_size: int = 64,
) -> Dict:
    """Run strictly causal prequential Online Bayesian TTT across unseen test worms."""
    if k_tier_schedule not in ("none", "t1_then_t2", "ttt_c_then_e", "ttt_c_then_d"):
        raise ValueError(
            f"Unsupported k_tier_schedule={k_tier_schedule!r}; expected 'none', 't1_then_t2', 'ttt_c_then_e', or 'ttt_c_then_d'."
        )
    t0 = time.time()
    model.eval()
    pop_snapshot = snapshot_population_state(model)
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    has_prestim = bool(getattr(model.decoder, "use_prestim_encoder", False))

    inferred_post = int(dataset.y.shape[1]) - int(pre_vols)
    eff_post_vols = inferred_post if post_vols is None else int(post_vols)

    by_worm = group_and_order_trials_by_worm(dataset.meta, test_indices)

    y_trues: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []
    y_cfs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    stims: List[np.ndarray] = []
    hist_counts: List[int] = []
    rec_ids: List[int] = []
    adapt_logs: List[Dict[str, float]] = []

    try:
        if int(ttt_steps) <= 0:
            flat_trials: List[Tuple[int, int, int]] = []
            for rec_id, worm_indices in by_worm.items():
                for target_pos, target_idx in enumerate(worm_indices):
                    hist_idx = select_causal_history_indices(
                        worm_indices,
                        target_pos=target_pos,
                        meta=dataset.meta,
                        post_vols=eff_post_vols,
                        max_history=max_history,
                        pre_vols=pre_vols,
                        strict_disjoint_pre=bool(strict_disjoint_pre),
                    )
                    flat_trials.append((int(rec_id), int(target_idx), len(hist_idx)))
            bsz_zs = max(1, int(zero_shot_batch_size))
            for b_start in range(0, len(flat_trials), bsz_zs):
                chunk = flat_trials[b_start : b_start + bsz_zs]
                idx_batch = [t[1] for t in chunk]
                stim_b = dataset.stim_inputs[idx_batch].to(device)
                y_b = dataset.y[idx_batch].to(device)
                mask_b = dataset.obs_mask[idx_batch].to(device)
                noise_b = dataset.noise_scale[idx_batch].to(device)
                gate_b = (
                    compute_drift_gate(stim_b, adj_t, drift_gate_mode)
                    if (has_prestim and drift_gate_mode != "none")
                    else None
                )
                pred_mean, pred_cf = _predict_single_trial_counterfactual(
                    model,
                    stim=stim_b,
                    y=y_b,
                    mask=mask_b,
                    noise=noise_b,
                    num_samples=num_samples,
                    pre_vols=pre_vols,
                    eval_seed=int(eval_seed) + b_start,
                    drift_gate=gate_b,
                )
                y_trues.append(y_b.cpu().numpy())
                y_preds.append(pred_mean)
                y_cfs.append(pred_cf)
                masks.append(mask_b.bool().cpu().numpy())
                stims.append(stim_b.cpu().numpy())
                for rec_i, _, k_h in chunk:
                    hist_counts.append(k_h)
                    rec_ids.append(rec_i)
        else:
            for worm_pos, (rec_id, worm_indices) in enumerate(by_worm.items()):
                for target_pos, target_idx in enumerate(worm_indices):
                    restore_population_state(model, pop_snapshot)
                    hist_idx = select_causal_history_indices(
                        worm_indices,
                        target_pos=target_pos,
                        meta=dataset.meta,
                        post_vols=eff_post_vols,
                        max_history=max_history,
                        pre_vols=pre_vols,
                        strict_disjoint_pre=bool(strict_disjoint_pre),
                    )

                    k_hist = len(hist_idx)
                    eff_tier: Optional[str] = None
                    eff_mode: Optional[str] = None
                    if k_hist > 0 and ttt_steps > 0:
                        if k_hist < int(min_history):
                            if k_tier_schedule == "t1_then_t2":
                                eff_tier = "tier1"
                                eff_mode = "ttt_c" if ttt_mode.startswith("ttt_") else "auto"
                            elif k_tier_schedule in ("ttt_c_then_e", "ttt_c_then_d"):
                                eff_tier = tier
                                eff_mode = "ttt_c"
                        else:
                            eff_tier = tier
                            if k_tier_schedule == "ttt_c_then_e" and ttt_mode == "auto":
                                eff_mode = "ttt_e"
                            elif k_tier_schedule == "ttt_c_then_d" and ttt_mode == "auto":
                                eff_mode = "ttt_d"
                            else:
                                eff_mode = ttt_mode

                    active_worm_b = None
                    if eff_tier is not None and eff_mode is not None:
                        stim_h = dataset.stim_inputs[hist_idx].to(device)
                        y_h = dataset.y[hist_idx].to(device)
                        mask_h = dataset.obs_mask[hist_idx].to(device)
                        noise_h = dataset.noise_scale[hist_idx].to(device)
                        gate_h = (
                            compute_drift_gate(stim_h, adj_t, drift_gate_mode)
                            if (has_prestim and drift_gate_mode != "none")
                            else None
                        )
                        log = adapt_worm_posterior(
                            model,
                            stim_hist=stim_h,
                            y_hist=y_h,
                            mask_hist=mask_h,
                            noise_hist=noise_h,
                            tier=eff_tier,
                            ttt_mode=eff_mode,
                            worm_latent_dim=worm_latent_dim,
                            temporal_val_frac=temporal_val_frac,
                            ttt_steps=ttt_steps,
                            ttt_lr=ttt_lr,
                            kl_weight=kl_weight,
                            nll_temp=nll_temp,
                            adapt_obs_sigma=adapt_obs_sigma,
                            pre_vols=pre_vols,
                            stim_loss_weight=stim_loss_weight,
                            hop1_loss_weight=hop1_loss_weight,
                            adjacency_t=adj_t,
                            drift_gate=gate_h,
                            seed=int(eval_seed) + 1000 * worm_pos + target_pos,
                        )
                        active_worm_b = log.pop("_worm_b_tensor", None)
                        adapt_logs.append(log)

                    stim_t = dataset.stim_inputs[target_idx : target_idx + 1].to(device)
                    y_t = dataset.y[target_idx : target_idx + 1].to(device)
                    mask_t = dataset.obs_mask[target_idx : target_idx + 1].to(device)
                    noise_t = dataset.noise_scale[target_idx : target_idx + 1].to(device)
                    gate_t = (
                        compute_drift_gate(stim_t, adj_t, drift_gate_mode)
                        if (has_prestim and drift_gate_mode != "none")
                        else None
                    )

                    pred_mean, pred_cf = _predict_single_trial_counterfactual(
                        model,
                        stim=stim_t,
                        y=y_t,
                        mask=mask_t,
                        noise=noise_t,
                        num_samples=num_samples,
                        pre_vols=pre_vols,
                        eval_seed=int(eval_seed) + 10000 * worm_pos + target_pos,
                        drift_gate=gate_t,
                        worm_b_override=active_worm_b,
                    )

                    y_trues.append(y_t.cpu().numpy())
                    y_preds.append(pred_mean)
                    y_cfs.append(pred_cf)
                    masks.append(mask_t.bool().cpu().numpy())
                    stims.append(stim_t.cpu().numpy())
                    hist_counts.append(len(hist_idx))
                    rec_ids.append(int(rec_id))
    finally:
        restore_population_state(model, pop_snapshot)

    y_true_arr = np.concatenate(y_trues, axis=0)
    y_pred_arr = np.concatenate(y_preds, axis=0)
    y_cf_arr = np.concatenate(y_cfs, axis=0)
    mask_arr = np.concatenate(masks, axis=0)
    stim_arr = np.concatenate(stims, axis=0)
    hist_arr = np.asarray(hist_counts, dtype=np.int64)
    dt_obs = float(getattr(model.decoder, "dt_obs", 0.5))

    overall_raw = compute_stratified_metrics(
        y_true_arr,
        y_pred_arr,
        mask_arr,
        stim=stim_arr,
        pre_vols=pre_vols,
        adjacency=adjacency,
        evoked_baseline="mean",
        y_pred_counterfactual=y_cf_arr,
        np_adjacency=np_adjacency,
        ma_adjacency=ma_adjacency,
        dt_obs=dt_obs,
    )

    learning_curve: Dict[str, Dict] = {}
    for lo, hi in history_bins:
        sel = (hist_arr >= int(lo)) & (hist_arr <= int(hi))
        label = f"k={lo}" if lo == hi else (f"k>={lo}" if hi >= 999 else f"k={lo}..{hi}")
        if not np.any(sel):
            learning_curve[label] = {"n_trials": 0, "metrics": {}}
            continue
        bin_raw = compute_stratified_metrics(
            y_true_arr[sel],
            y_pred_arr[sel],
            mask_arr[sel],
            stim=stim_arr[sel],
            pre_vols=pre_vols,
            adjacency=adjacency,
            evoked_baseline="mean",
            y_pred_counterfactual=y_cf_arr[sel],
            np_adjacency=np_adjacency,
            ma_adjacency=ma_adjacency,
            dt_obs=dt_obs,
        )
        learning_curve[label] = {
            "n_trials": int(sel.sum()),
            "metrics": _summarize_metrics(bin_raw),
        }

    return {
        "tier": tier,
        "ttt_mode": ttt_mode,
        "worm_latent_dim": int(worm_latent_dim),
        "temporal_val_frac": float(temporal_val_frac),
        "ttt_steps": int(ttt_steps),
        "ttt_lr": float(ttt_lr),
        "kl_weight": float(kl_weight),
        "nll_temp": float(nll_temp),
        "adapt_obs_sigma": bool(adapt_obs_sigma),
        "min_history": int(min_history),
        "k_tier_schedule": k_tier_schedule,
        "drift_gate_mode": drift_gate_mode,
        "strict_disjoint_pre": bool(strict_disjoint_pre),
        "max_history": None if max_history is None else int(max_history),
        "n_worms": int(len(by_worm)),
        "n_trials": int(y_true_arr.shape[0]),
        "n_adapted_trials": int(len(adapt_logs)),
        "mean_selected_step": float(np.mean([d.get("selected_step", ttt_steps) for d in adapt_logs])) if adapt_logs else 0.0,
        "mean_kl_final": float(np.mean([d["kl_final"] for d in adapt_logs])) if adapt_logs else 0.0,
        "max_param_shift": float(np.max([d["max_param_shift"] for d in adapt_logs])) if adapt_logs else 0.0,
        "max_whitened_shift": float(np.max([d["max_whitened_shift"] for d in adapt_logs])) if adapt_logs else 0.0,
        "overall_metrics": _summarize_metrics(overall_raw),
        "learning_curve": learning_curve,
        "eval_wall_s": float(time.time() - t0),
    }


def _compute_ttt_promotion_gate(zero_shot: Dict, online_ttt: Dict) -> Dict[str, object]:
    """Evaluate Section 5.2 promotion gate: evoked improves & post_inlier_r2 >= 95% zero-shot."""
    zm = zero_shot.get("overall_metrics", {}) or {}
    om = online_ttt.get("overall_metrics", {}) or {}
    zs_post = float(zm.get("post_inlier_r2", float("nan")))
    on_post = float(om.get("post_inlier_r2", float("nan")))
    zs_ev_s = float(zm.get("evoked_stim_r2", float("nan")))
    on_ev_s = float(om.get("evoked_stim_r2", float("nan")))
    zs_ev_h = float(zm.get("evoked_hop1_r2", float("nan")))
    on_ev_h = float(om.get("evoked_hop1_r2", float("nan")))
    zs_k = float(zm.get("kernel_hop1_r2", float("nan")))
    on_k = float(om.get("kernel_hop1_r2", float("nan")))

    ratio = float(on_post / zs_post) if (np.isfinite(zs_post) and abs(zs_post) > 1e-12) else float("nan")
    post_ok = bool(np.isfinite(on_post) and np.isfinite(zs_post) and (on_post >= (0.95 * zs_post if zs_post >= 0 else zs_post)))
    stim_delta = float(on_ev_s - zs_ev_s)
    hop1_delta = float(on_ev_h - zs_ev_h)
    kernel_delta = float(on_k - zs_k) if (np.isfinite(on_k) and np.isfinite(zs_k)) else float("nan")
    evoked_ok = bool(stim_delta > 0.0 and (hop1_delta >= 0.0 or (np.isfinite(kernel_delta) and kernel_delta > 0.0)))
    return {
        "post_inlier_r2_zero_shot": zs_post,
        "post_inlier_r2_online": on_post,
        "post_inlier_r2_ratio": ratio,
        "post_inlier_r2_pass": post_ok,
        "evoked_stim_r2_delta": stim_delta,
        "evoked_hop1_r2_delta": hop1_delta,
        "kernel_hop1_r2_delta": kernel_delta,
        "evoked_improved": evoked_ok,
        "pass": bool(post_ok and evoked_ok),
    }


def _load_checkpoint_model(
    ckpt_path: str,
    args: argparse.Namespace,
    shuffle_seed: Optional[int],
    device: torch.device,
) -> vae.VAElegans:
    pyro.clear_param_store()
    snap = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = snap["MODEL_STATE"]
    ms_keys = set(ms.keys())
    per_neuron_stim_gain = args.per_neuron_stim_gain or any("guide.stim_gain_vec" in k for k in ms_keys)
    tau_ca_t = ms.get("guide.tau_ca_loc_unconstrained")
    per_neuron_kinetics = args.per_neuron_kinetics or (tau_ca_t is not None and tau_ca_t.ndim >= 1 and tau_ca_t.numel() > 1)
    per_edge_release = args.per_edge_release or ("guide.v_rest_edge_loc_unconstrained" in ms_keys)
    split_extrasyn = args.split_extrasynaptic or any("guide.extrasyn_np_gain_z" in k or "guide.extrasyn_ma_gain_z" in k for k in ms_keys)
    use_syn_filter = args.syn_filter or any("guide.tau_syn_loc" in k for k in ms_keys)
    use_extrasyn = bool(args.use_extrasynaptic or split_extrasyn or any("guide.extrasyn_gain_z" in k for k in ms_keys))
    use_prestim = (not args.no_prestim_encoder) and any("decoder.prestim_encoder." in k for k in ms_keys)
    prestim_oscillatory = args.prestim_oscillatory or any("decoder.prestim_encoder.osc_out" in k for k in ms_keys)
    use_ssm = args.latent_ssm or any("decoder.latent_ssm." in k for k in ms_keys)
    direct_guide = args.direct_global_guide or any("guide.wicks_g_max_z" in k for k in ms_keys)
    dimensionless = args.dimensionless_physiology or (
        direct_guide and ("guide.neuron_G_loc_unconstrained" not in ms_keys)
    )
    leak_gauge = (
        "zero_mean_log"
        if any(
            k in ms_keys
            for k in (
                "guide.neuron_log_G_raw_loc",
                "guide.neuron_log_G_raw_loc_unconstrained",
            )
        )
        else "unit"
    )
    ssm_encoder = "gru" if any("decoder.latent_ssm.pre_gru." in k for k in ms_keys) else "pooling"
    ssm_split = any("decoder.latent_ssm.art_to_obs." in k for k in ms_keys)
    art_key = next(
        (
            k
            for k in (
                "decoder.latent_ssm.A_art_decay_logit",
                "decoder.latent_ssm.A_art_decay_logit_unconstrained",
            )
            if k in ms_keys
        ),
        None,
    )
    ssm_art_dim = int(ms[art_key].numel()) if art_key is not None else 8
    ssm_vtraj = any("decoder.latent_ssm.q_traj_gru." in k for k in ms_keys)
    ssm_smooth = any("decoder.latent_ssm.q_smooth_gru." in k for k in ms_keys)
    decay_key = next(
        (
            k
            for k in (
                "decoder.latent_ssm.A_decay_logit",
                "decoder.latent_ssm.A_decay_logit_unconstrained",
                "decoder.latent_ssm.A_diag_raw",
                "decoder.latent_ssm.A_diag_raw_unconstrained",
            )
            if k in ms_keys
        ),
        None,
    )
    if decay_key is not None:
        eff_ssm_dim = int(ms[decay_key].numel())
    else:
        eff_ssm_dim = int(args.latent_ssm_dim)
    ssm_state_observer = bool(
        getattr(args, "ssm_state_observer", False) or any("decoder.latent_ssm.obs_local_fc1." in k for k in ms_keys)
    )
    ssm_min_decay = float(getattr(args, "ssm_min_decay", 0.0))
    u_chem_t = next((ms[k] for k in ("decoder.U_chem_unconstrained", "decoder.U_chem") if k in ms_keys), None)
    worm_b_t = next(
        (ms[k] for k in ("guide.worm_b_loc_unconstrained", "guide.worm_b_loc", "decoder.worm_b_unconstrained", "decoder.worm_b") if k in ms_keys),
        None,
    )
    worm_dim_ckpt = int(u_chem_t.shape[1]) if u_chem_t is not None else 0
    num_worms_ckpt = int(worm_b_t.shape[0]) if worm_b_t is not None else 0

    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=use_extrasyn,
        per_edge_release=per_edge_release,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
        evoked_syn_priors=args.evoked_syn_priors,
        dimensionless=dimensionless,
        leak_gauge=leak_gauge,
        split_extrasynaptic=split_extrasyn,
        use_syn_filter=use_syn_filter,
        shuffle_seed=shuffle_seed,
        shuffle_disjoint=bool(getattr(args, "shuffle_disjoint", False)),
    )
    rec = c302.RecurrentNematode(
        step,
        dt_obs=0.5,
        substeps=int(args.substeps),
        use_prestim_encoder=bool(use_prestim),
        pre_vols=int(args.pre_vols),
        prestim_rank=16,
        variational_prestim=bool(direct_guide and use_prestim),
        prestim_oscillatory=bool(prestim_oscillatory),
        per_neuron_kinetics=bool(per_neuron_kinetics),
        per_edge_release=bool(per_edge_release),
        stim_actuator=bool(args.stim_actuator),
        stim_tau_rise=float(args.stim_tau_rise),
        stim_tau_decay=float(args.stim_tau_decay),
        use_latent_ssm=bool(use_ssm),
        latent_ssm_dim=eff_ssm_dim,
        ssm_encoder=ssm_encoder,
        ssm_split_latent=ssm_split,
        ssm_artifact_dim=ssm_art_dim,
        ssm_variational_traj=ssm_vtraj,
        ssm_smoother=ssm_smooth,
        ssm_state_observer=ssm_state_observer,
        ssm_min_decay=ssm_min_decay,
        worm_latent_dim=worm_dim_ckpt,
        num_worms=num_worms_ckpt,
    )
    model = vae.VAElegans(
        rec,
        rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC),
        len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=bool(direct_guide),
    ).to(device)
    pyro.clear_param_store()
    model.load_state_dict(snap["MODEL_STATE"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Causal prequential Online Bayesian TTT evaluation")
    p.add_argument("--cache", required=True, help="Path to causal RandiTrialDataset .npz cache")
    p.add_argument("--split-file", default=None, help="Optional split JSON with 'test' indices (defaults to all trials)")
    p.add_argument("--checkpoint", required=True, help="Intact pretrained checkpoint .pt path")
    p.add_argument("--shuffle-checkpoint", default=None, help="Optional shuffled control checkpoint .pt path")
    p.add_argument("--shuffle-seed", default=7, type=int, help="Shuffle seed for --shuffle-checkpoint")
    p.add_argument("--out", required=True, help="Output JSON summary path")
    p.add_argument("--tier", choices=["tier1", "tier2", "tier3"], default="tier1")
    p.add_argument("--ttt-mode", choices=["auto", "ttt_a", "ttt_b", "ttt_c", "ttt_d", "ttt_e", "ttt_f"], default="auto")
    p.add_argument(
        "--ttt-modes", default=None, type=str,
        help="Optional comma-separated modes (e.g. ttt_c,ttt_d,ttt_c_then_d) sharing a single zero-shot evaluation",
    )
    p.add_argument("--worm-latent-dim", default=8, type=int)
    p.add_argument("--temporal-val-frac", default=0.0, type=float)
    p.add_argument("--ttt-steps", default=8, type=int)
    p.add_argument("--ttt-lr", default=5e-2, type=float)
    p.add_argument("--kl-weight", default=1.0, type=float)
    p.add_argument("--nll-temp", default=1.0, type=float, help="Autocorrelation likelihood tempering factor")
    p.add_argument("--adapt-obs-sigma", action="store_true", help="Allow obs_sigma_loc adaptation during TTT")
    p.add_argument("--min-history", default=0, type=int, help="Keep zero-shot (or Tier 1 if scheduled) below this k")
    p.add_argument("--k-tier-schedule", choices=["none", "t1_then_t2", "ttt_c_then_e", "ttt_c_then_d"], default="none")
    p.add_argument("--max-history", default=None, type=int)
    p.add_argument("--strict-disjoint-pre", action="store_true", help="Also exclude overlap with target pre-stim window")
    p.add_argument("--stim-loss-weight", default=1.0, type=float)
    p.add_argument("--hop1-loss-weight", default=1.0, type=float)
    p.add_argument("--drift-gate", choices=["none", "direct", "direct_hop1"], default="none")
    p.add_argument("--num-samples", default=8, type=int)
    p.add_argument("--eval-seed", default=42, type=int)
    p.add_argument("--substeps", default=4, type=int)
    p.add_argument("--pre-vols", default=10, type=int)
    p.add_argument("--no-prestim-encoder", action="store_true")
    p.add_argument("--per-neuron-kinetics", action="store_true")
    p.add_argument("--per-edge-release", action="store_true")
    p.add_argument("--per-neuron-stim-gain", action="store_true")
    p.add_argument("--evoked-syn-priors", action="store_true")
    p.add_argument("--direct-global-guide", action="store_true")
    p.add_argument("--use-extrasynaptic", action="store_true")
    p.add_argument("--split-extrasynaptic", action="store_true")
    p.add_argument("--syn-filter", action="store_true")
    p.add_argument("--prestim-oscillatory", action="store_true")
    p.add_argument("--dimensionless-physiology", action="store_true")
    p.add_argument("--latent-ssm", action="store_true")
    p.add_argument("--latent-ssm-dim", default=16, type=int)
    p.add_argument("--ssm-state-observer", action="store_true")
    p.add_argument("--ssm-min-decay", default=0.0, type=float)
    p.add_argument("--shuffle-disjoint", action="store_true")
    p.add_argument("--stim-actuator", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stim-tau-rise", default=0.08, type=float)
    p.add_argument("--stim-tau-decay", default=0.4, type=float)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RandiTrialDataset.load(args.cache)
    if args.split_file is not None:
        with open(args.split_file) as fh:
            split = json.load(fh)
        test_indices = [int(i) for i in split["test"]]
    else:
        test_indices = list(range(len(dataset)))

    intact_conn = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=True,
        split_extrasynaptic=True,
    )
    adjacency = intact_conn.synapse.adjacency_matrix(intact_conn.neuronSize)
    np_adjacency = intact_conn.synapse.np_adjacency_matrix(intact_conn.neuronSize)
    ma_adjacency = intact_conn.synapse.ma_adjacency_matrix(intact_conn.neuronSize)

    # 1. Zero-shot baseline (ttt_steps=0) and Online TTT on intact checkpoint
    intact_model = _load_checkpoint_model(args.checkpoint, args, shuffle_seed=None, device=device)
    if args.ttt_modes:
        mode_names = [m.strip() for m in str(args.ttt_modes).split(",") if m.strip()]

        def _parse_mode(name: str) -> Tuple[str, str]:
            if name in ("ttt_c_then_d", "ttt_c_then_e"):
                return "auto", name
            return name, args.k_tier_schedule

        intact_zero = evaluate_online_ttt(
            intact_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
            tier=args.tier, ttt_mode="auto", worm_latent_dim=args.worm_latent_dim,
            temporal_val_frac=args.temporal_val_frac, ttt_steps=0, pre_vols=args.pre_vols,
            strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
            num_samples=args.num_samples, eval_seed=args.eval_seed,
        )
        modes_out: Dict[str, Dict] = {}
        for m_name in mode_names:
            eff_m, eff_sched = _parse_mode(m_name)
            m_on = evaluate_online_ttt(
                intact_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
                tier=args.tier, ttt_mode=eff_m, worm_latent_dim=args.worm_latent_dim,
                temporal_val_frac=args.temporal_val_frac,
                ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
                nll_temp=args.nll_temp, adapt_obs_sigma=args.adapt_obs_sigma,
                min_history=args.min_history, k_tier_schedule=eff_sched,
                pre_vols=args.pre_vols, max_history=args.max_history,
                strict_disjoint_pre=args.strict_disjoint_pre,
                stim_loss_weight=args.stim_loss_weight, hop1_loss_weight=args.hop1_loss_weight,
                drift_gate_mode=args.drift_gate,
                num_samples=args.num_samples, eval_seed=args.eval_seed,
            )
            modes_out[m_name] = {
                "promotion_gate": _compute_ttt_promotion_gate(intact_zero, m_on),
                "online_ttt": m_on,
            }
        result: Dict = {
            "cache": args.cache,
            "split_file": args.split_file,
            "checkpoint": args.checkpoint,
            "tier": args.tier,
            "ttt_modes": mode_names,
            "worm_latent_dim": int(args.worm_latent_dim),
            "temporal_val_frac": float(args.temporal_val_frac),
            "substeps": int(args.substeps),
            "intact_zero_shot": intact_zero,
            "modes": modes_out,
        }
        if args.shuffle_checkpoint is not None:
            shuf_model = _load_checkpoint_model(
                args.shuffle_checkpoint, args, shuffle_seed=args.shuffle_seed, device=device
            )
            shuf_zero = evaluate_online_ttt(
                shuf_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
                tier=args.tier, ttt_mode="auto", worm_latent_dim=args.worm_latent_dim,
                temporal_val_frac=args.temporal_val_frac, ttt_steps=0, pre_vols=args.pre_vols,
                strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
                num_samples=args.num_samples, eval_seed=args.eval_seed,
            )
            shuf_modes_out: Dict[str, Dict] = {}
            for m_name in mode_names:
                eff_m, eff_sched = _parse_mode(m_name)
                s_on = evaluate_online_ttt(
                    shuf_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
                    tier=args.tier, ttt_mode=eff_m, worm_latent_dim=args.worm_latent_dim,
                    temporal_val_frac=args.temporal_val_frac,
                    ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
                    nll_temp=args.nll_temp, adapt_obs_sigma=args.adapt_obs_sigma,
                    min_history=args.min_history, k_tier_schedule=eff_sched,
                    pre_vols=args.pre_vols, max_history=args.max_history,
                    strict_disjoint_pre=args.strict_disjoint_pre,
                    stim_loss_weight=args.stim_loss_weight, hop1_loss_weight=args.hop1_loss_weight,
                    drift_gate_mode=args.drift_gate,
                    num_samples=args.num_samples, eval_seed=args.eval_seed,
                )
                shuf_modes_out[m_name] = s_on
                modes_out[m_name]["delta_intact_minus_shuffle"] = {
                    "zero_shot_evoked_hop1_r2": float(
                        intact_zero["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                        - shuf_zero["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                    ),
                    "online_ttt_evoked_hop1_r2": float(
                        modes_out[m_name]["online_ttt"]["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                        - s_on["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                    ),
                }
            result["shuffle_checkpoint"] = args.shuffle_checkpoint
            result["shuffle_seed"] = int(args.shuffle_seed)
            result["shuffle_zero_shot"] = shuf_zero
            result["shuffle_modes"] = shuf_modes_out
    else:
        intact_zero = evaluate_online_ttt(
            intact_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
            tier=args.tier, ttt_mode=args.ttt_mode, worm_latent_dim=args.worm_latent_dim,
            temporal_val_frac=args.temporal_val_frac, ttt_steps=0, pre_vols=args.pre_vols,
            strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
            num_samples=args.num_samples, eval_seed=args.eval_seed,
        )
        intact_ttt = evaluate_online_ttt(
            intact_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
            tier=args.tier, ttt_mode=args.ttt_mode, worm_latent_dim=args.worm_latent_dim,
            temporal_val_frac=args.temporal_val_frac,
            ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
            nll_temp=args.nll_temp, adapt_obs_sigma=args.adapt_obs_sigma,
            min_history=args.min_history, k_tier_schedule=args.k_tier_schedule,
            pre_vols=args.pre_vols, max_history=args.max_history,
            strict_disjoint_pre=args.strict_disjoint_pre,
            stim_loss_weight=args.stim_loss_weight, hop1_loss_weight=args.hop1_loss_weight,
            drift_gate_mode=args.drift_gate,
            num_samples=args.num_samples, eval_seed=args.eval_seed,
        )

        result = {
            "cache": args.cache,
            "split_file": args.split_file,
            "checkpoint": args.checkpoint,
            "tier": args.tier,
            "ttt_mode": args.ttt_mode,
            "worm_latent_dim": int(args.worm_latent_dim),
            "temporal_val_frac": float(args.temporal_val_frac),
            "substeps": int(args.substeps),
            "promotion_gate": _compute_ttt_promotion_gate(intact_zero, intact_ttt),
            "intact_zero_shot": intact_zero,
            "intact_online_ttt": intact_ttt,
        }

        if args.shuffle_checkpoint is not None:
            shuf_model = _load_checkpoint_model(
                args.shuffle_checkpoint, args, shuffle_seed=args.shuffle_seed, device=device
            )
            shuf_zero = evaluate_online_ttt(
                shuf_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
                tier=args.tier, ttt_mode=args.ttt_mode, worm_latent_dim=args.worm_latent_dim,
                temporal_val_frac=args.temporal_val_frac, ttt_steps=0, pre_vols=args.pre_vols,
                strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
                num_samples=args.num_samples, eval_seed=args.eval_seed,
            )
            shuf_ttt = evaluate_online_ttt(
                shuf_model, dataset, test_indices, device, adjacency, np_adjacency, ma_adjacency,
                tier=args.tier, ttt_mode=args.ttt_mode, worm_latent_dim=args.worm_latent_dim,
                temporal_val_frac=args.temporal_val_frac,
                ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
                nll_temp=args.nll_temp, adapt_obs_sigma=args.adapt_obs_sigma,
                min_history=args.min_history, k_tier_schedule=args.k_tier_schedule,
                pre_vols=args.pre_vols, max_history=args.max_history,
                strict_disjoint_pre=args.strict_disjoint_pre,
                stim_loss_weight=args.stim_loss_weight, hop1_loss_weight=args.hop1_loss_weight,
                drift_gate_mode=args.drift_gate,
                num_samples=args.num_samples, eval_seed=args.eval_seed,
            )
            result["shuffle_checkpoint"] = args.shuffle_checkpoint
            result["shuffle_seed"] = int(args.shuffle_seed)
            result["shuffle_zero_shot"] = shuf_zero
            result["shuffle_online_ttt"] = shuf_ttt
            result["delta_intact_minus_shuffle"] = {
                "zero_shot_evoked_hop1_r2": float(
                    intact_zero["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                    - shuf_zero["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                ),
                "online_ttt_evoked_hop1_r2": float(
                    intact_ttt["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                    - shuf_ttt["overall_metrics"].get("evoked_hop1_r2", float("nan"))
                ),
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
