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
from pyro_svi_c302 import compute_drift_gate, compute_stratified_metrics
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
) -> List[Tuple[str, torch.nn.Parameter, torch.Tensor]]:
    """Return list of (name, trainable_param, pop_sigma) for the requested TTT tier.

    For vector/per-neuron parameters, `0.5 * sum(((mu - mu0) / s)^2)` matches the exact
    diagonal Normal/LogNormal KL. For Tier 2 scalar global shift parameters
    (`wicks_g_max_global_z`, `gap_g_syn_global_z`), `s` is set to the mean marginal
    posterior scale, yielding a compact hierarchical worm-level prior over global gain.
    Tier 3 freezes global chemical gain to avoid redundant coupling with `wicks_g_max_z`.
    """
    if tier not in ("tier1", "tier2", "tier3"):
        raise ValueError(f"Unsupported TTT tier={tier!r}; expected 'tier1', 'tier2', or 'tier3'.")

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

    # Tier 1: observation & optogenetic stimulation calibration (~900 dims, zero topology edits)
    _add("gcamp_gain_loc", getattr(guide, "gcamp_gain_scale", None))
    _add("gcamp_offset_loc", getattr(guide, "gcamp_offset_scale", None))
    if getattr(guide, "per_neuron_stim_gain", False):
        _add("stim_gain_vec_loc", getattr(guide, "stim_gain_vec_scale", None))
    else:
        _add("stim_gain_loc", getattr(guide, "stim_gain_scale", None))
    if adapt_obs_sigma:
        _add("obs_sigma_loc", getattr(guide, "obs_sigma_scale", None))

    if tier in ("tier2", "tier3"):
        # Tier 2: hierarchical worm-level global synaptic scale & per-neuron excitability
        if getattr(guide, "direct_global_guide", False):
            if tier == "tier2" and getattr(guide, "wicks_dim", 0) > 0:
                _add("wicks_g_max_global_z", getattr(guide, "wicks_g_max_scale", None), coord_mult=1.5)
            if getattr(guide, "gap_dim", 0) > 0:
                _add("gap_g_syn_global_z", getattr(guide, "gap_g_syn_scale", None), coord_mult=1.5)
            _add("neuron_E_z", getattr(guide, "neuron_E_scale", None), coord_mult=0.01)
            _add("wicks_V_rest_z", getattr(guide, "wicks_V_rest_scale", None), coord_mult=0.01)
            _add("wicks_V_slope_loc", getattr(guide, "wicks_V_slope_scale", None), coord_mult=1.0)
            _add("neuron_C_loc", getattr(guide, "neuron_C_scale", None), coord_mult=1.0)

    if tier == "tier3":
        # Tier 3: sparse local outgoing edges of previously stimulated source neurons
        if getattr(guide, "direct_global_guide", False) and getattr(guide, "wicks_dim", 0) > 0:
            _add("wicks_g_max_z", getattr(guide, "wicks_g_max_scale", None), coord_mult=3.0)
            _add("wicks_E_rev_z", getattr(guide, "wicks_E_rev_scale", None), coord_mult=0.04)

    return specs


def snapshot_population_state(model: vae.VAElegans) -> dict:
    """Deep-copy model state_dict for exact per-worm resets."""
    return {
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def restore_population_state(model: vae.VAElegans, pop_snapshot: dict) -> None:
    """Restore model bitwise and clear ParamStore so Pyro re-binds live nn.Parameters."""
    pyro.clear_param_store()
    model.load_state_dict({k: v.detach().clone() for k, v in pop_snapshot["model_state"].items()})


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
    if ttt_steps <= 0 or stim_hist.shape[0] == 0:
        return {
            "n_hist_trials": int(stim_hist.shape[0]),
            "ttt_steps": 0,
            "loss_start": float("nan"),
            "loss_end": float("nan"),
            "kl_final": 0.0,
            "max_param_shift": 0.0,
            "max_whitened_shift": 0.0,
        }

    anchor_specs = _get_tier_anchor_specs(model, tier=tier, adapt_obs_sigma=bool(adapt_obs_sigma))
    if not anchor_specs:
        return {
            "n_hist_trials": int(stim_hist.shape[0]),
            "ttt_steps": 0,
            "loss_start": float("nan"),
            "loss_end": float("nan"),
            "kl_final": 0.0,
            "max_param_shift": 0.0,
            "max_whitened_shift": 0.0,
        }

    # Freeze all parameters, then unfreeze only the selected tier parameters.
    prev_req_grad = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    # Optimize in whitened coordinates u = (param - pop_val) / pop_sig so Adam steps
    # are measured in units of population posterior std (sigma_pop), preventing narrow-scale
    # parameters like gcamp_offset_loc (sigma~0.02) from overshooting while wide-scale
    # log-gains (sigma~0.35) under-adapt.
    anchors: List[Tuple[str, torch.nn.Parameter, torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []
    whitened_params: List[torch.nn.Parameter] = []
    for name, param, sig in anchor_specs:
        param.requires_grad_(True)
        u_param = torch.nn.Parameter(torch.zeros_like(param.detach()))
        anchors.append((name, param, u_param, param.detach().clone(), sig))
        whitened_params.append(u_param)

    # For Tier 3, restrict edge-level gradients to outgoing edges of stimulated neurons in history.
    active_wicks_mask: Optional[torch.Tensor] = None
    if tier == "tier3" and hasattr(model.decoder.model.synapse, "Wicks_SRC"):
        wicks_src = torch.as_tensor(
            model.decoder.model.synapse.Wicks_SRC, device=stim_hist.device, dtype=torch.long
        )
        stim_sources = (stim_hist.abs().sum(dim=(0, 1)) > 0)  # (N,)
        active_wicks_mask = stim_sources[wicks_src].to(dtype=stim_hist.dtype)

    dec = model.decoder
    has_prestim = bool(getattr(dec, "use_prestim_encoder", False))
    ssm_mod = getattr(dec, "latent_ssm", None)
    has_ssm = ssm_mod is not None
    has_history = has_prestim or has_ssm
    likelihood = str(getattr(dec, "likelihood", "student_t"))
    student_t_df = float(getattr(dec, "student_t_df", 4.0))

    t_cut = min(int(pre_vols), y_hist.shape[1])
    pre_obs = y_hist[:, :t_cut, :] if has_history else None
    pre_mask = mask_hist[:, :t_cut, :] if has_history else None

    # Reweight observation noise on direct-stim / 1-hop targets if requested.
    eff_noise = noise_hist
    if (stim_loss_weight != 1.0 or hop1_loss_weight != 1.0) and adjacency_t is not None:
        eff_noise = reweight_hop1_noise(
            noise_hist,
            stim_hist,
            adjacency_t,
            weight=float(hop1_loss_weight),
            stim_weight=float(stim_loss_weight),
        )

    post_mask = torch.zeros_like(mask_hist, dtype=torch.bool)
    if t_cut < y_hist.shape[1]:
        post_mask[:, t_cut:, :] = mask_hist[:, t_cut:, :].bool()
    else:
        post_mask = mask_hist.bool()

    opt = torch.optim.Adam(whitened_params, lr=float(ttt_lr))
    loss_start = float("nan")
    loss_end = float("nan")
    kl_final = 0.0

    # Deterministic reparameterized guide trace per step so updates are reproducible.
    guide_kwargs = dict(
        ExternalInput=stim_hist,
        mask=None,
        y=None,
        noise_scale=eff_noise,
    )
    if has_prestim:
        guide_kwargs["pre_stim_obs"] = pre_obs
        guide_kwargs["pre_stim_mask"] = pre_mask
        if drift_gate is not None:
            guide_kwargs["drift_gate"] = drift_gate

    model_kwargs = dict(guide_kwargs)
    if has_ssm and not has_prestim:
        model_kwargs["pre_stim_obs"] = pre_obs
        model_kwargs["pre_stim_mask"] = pre_mask
    if has_ssm:
        model_kwargs["ssm_eps"] = torch.zeros(
            (stim_hist.shape[0], stim_hist.shape[1] + 1, int(ssm_mod.z_dim)),
            device=stim_hist.device,
            dtype=stim_hist.dtype,
        )
    model_kwargs["sample_obs"] = False

    # Fix RNG seed once per adaptation call for reproducible low-variance updates.
    gen_state_cpu = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    pyro.set_rng_seed(int(seed))

    try:
        for step_idx in range(int(ttt_steps)):
            opt.zero_grad(set_to_none=True)
            for _, param, _, _, _ in anchors:
                param.grad = None

            # Deterministic posterior-median rollout avoids Monte Carlo variance during
            # few-shot (5-10 step) TTT while keeping every loc/z PyroParam differentiable.
            ps = _posterior_median_from_guide(model.guide, guide_kwargs)
            y_pred = pyro.condition(model, data=ps)(**model_kwargs)

            obs_sigma = torch.clamp(ps["decoder.obs_sigma"], min=1e-3)
            scale = torch.sqrt((obs_sigma * 0.5) ** 2 + eff_noise.unsqueeze(1) ** 2)
            nll = _student_t_or_normal_nll(
                y_hist,
                y_pred,
                post_mask,
                scale,
                likelihood=likelihood,
                student_t_df=student_t_df,
            )

            kl = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
            for _, param, _, pop_val, pop_sig in anchors:
                diff = (param - pop_val) / pop_sig
                kl = kl + 0.5 * (diff * diff).sum()

            loss = float(nll_temp) * nll + float(kl_weight) * kl
            loss.backward()

            # Exact chain rule: dL/du = (dL/dparam) * pop_sig for param = pop_val + pop_sig * u
            for name, param, u_param, _, pop_sig in anchors:
                if param.grad is None:
                    continue
                g_u = param.grad.detach() * pop_sig
                if active_wicks_mask is not None and name in ("wicks_g_max_z", "wicks_E_rev_z"):
                    g_u = g_u * active_wicks_mask
                u_param.grad = g_u

            torch.nn.utils.clip_grad_norm_(whitened_params, max_norm=5.0)
            opt.step()

            with torch.no_grad():
                for _, param, u_param, pop_val, pop_sig in anchors:
                    param.copy_(pop_val + pop_sig * u_param)

            loss_val = float(loss.detach().item())
            if step_idx == 0:
                loss_start = loss_val
            loss_end = loss_val
            kl_final = float(kl.detach().item())
    finally:
        torch.random.set_rng_state(gen_state_cpu)
        for p in model.parameters():
            p.requires_grad_(prev_req_grad.get(id(p), True))

    max_shift = max(
        float((param.detach() - pop_val).abs().max().item())
        for _, param, _, pop_val, _ in anchors
    )
    max_whitened_shift = max(
        float(u_param.detach().abs().max().item())
        for _, _, u_param, _, _ in anchors
    )
    return {
        "n_hist_trials": int(stim_hist.shape[0]),
        "ttt_steps": int(ttt_steps),
        "loss_start": loss_start,
        "loss_end": loss_end,
        "kl_final": kl_final,
        "max_param_shift": max_shift,
        "max_whitened_shift": max_whitened_shift,
    }


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

    bsz = stim.shape[0]
    torch.manual_seed(int(eval_seed))
    pyro.set_rng_seed(int(eval_seed))
    ps_acc = None
    pz_acc = None
    with torch.no_grad():
        for _ in range(max(1, int(num_samples))):
            gtr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
            ps = {
                name: site["value"]
                for name, site in gtr.nodes.items()
                if site["type"] == "sample" and name != "batch"
            }
            pz = ps.get("prestim_z")
            if torch.is_tensor(pz) and pz.dim() >= 2 and pz.shape[0] == bsz:
                ps = dict(ps)
                ps["prestim_z"] = torch.cat([pz, pz], dim=0)
            if has_ssm:
                eps_draw = torch.randn(
                    (bsz, stim.shape[1] + 1, int(ssm_mod.z_dim)),
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
    tier: str = "tier1",
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
) -> Dict:
    """Run strictly causal prequential Online Bayesian TTT across unseen test worms."""
    if k_tier_schedule not in ("none", "t1_then_t2"):
        raise ValueError(f"Unsupported k_tier_schedule={k_tier_schedule!r}; expected 'none' or 't1_then_t2'.")
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
        for worm_pos, (rec_id, worm_indices) in enumerate(by_worm.items()):
            for target_pos, target_idx in enumerate(worm_indices):
                # Always start from clean population posterior q_pop before adapting on
                # strictly causal non-overlapping history for this trial.
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
                if k_hist > 0 and ttt_steps > 0:
                    if k_hist < int(min_history):
                        if k_tier_schedule == "t1_then_t2":
                            eff_tier = "tier1"
                    else:
                        eff_tier = tier

                if eff_tier is not None:
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
            dt_obs=dt_obs,
        )
        learning_curve[label] = {
            "n_trials": int(sel.sum()),
            "metrics": _summarize_metrics(bin_raw),
        }

    return {
        "tier": tier,
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
        "mean_kl_final": float(np.mean([d["kl_final"] for d in adapt_logs])) if adapt_logs else 0.0,
        "max_param_shift": float(np.max([d["max_param_shift"] for d in adapt_logs])) if adapt_logs else 0.0,
        "max_whitened_shift": float(np.max([d["max_whitened_shift"] for d in adapt_logs])) if adapt_logs else 0.0,
        "overall_metrics": _summarize_metrics(overall_raw),
        "learning_curve": learning_curve,
        "eval_wall_s": float(time.time() - t0),
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

    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=use_extrasyn,
        per_edge_release=per_edge_release,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
        evoked_syn_priors=args.evoked_syn_priors,
        dimensionless=dimensionless,
        split_extrasynaptic=split_extrasyn,
        use_syn_filter=use_syn_filter,
        shuffle_seed=shuffle_seed,
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
        latent_ssm_dim=int(args.latent_ssm_dim),
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
    p.add_argument("--ttt-steps", default=8, type=int)
    p.add_argument("--ttt-lr", default=5e-2, type=float)
    p.add_argument("--kl-weight", default=1.0, type=float)
    p.add_argument("--nll-temp", default=1.0, type=float, help="Autocorrelation likelihood tempering factor")
    p.add_argument("--adapt-obs-sigma", action="store_true", help="Allow obs_sigma_loc adaptation during TTT")
    p.add_argument("--min-history", default=0, type=int, help="Keep zero-shot (or Tier 1 if scheduled) below this k")
    p.add_argument("--k-tier-schedule", choices=["none", "t1_then_t2"], default="none")
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

    intact_conn = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adjacency = intact_conn.synapse.adjacency_matrix(intact_conn.neuronSize)
    np_adjacency = intact_conn.synapse.extrasynaptic_adjacency_matrix(intact_conn.neuronSize)

    # 1. Zero-shot baseline (ttt_steps=0) and Online TTT on intact checkpoint
    intact_model = _load_checkpoint_model(args.checkpoint, args, shuffle_seed=None, device=device)
    intact_zero = evaluate_online_ttt(
        intact_model, dataset, test_indices, device, adjacency, np_adjacency,
        tier=args.tier, ttt_steps=0, pre_vols=args.pre_vols,
        strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
        num_samples=args.num_samples, eval_seed=args.eval_seed,
    )
    intact_ttt = evaluate_online_ttt(
        intact_model, dataset, test_indices, device, adjacency, np_adjacency,
        tier=args.tier, ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
        nll_temp=args.nll_temp, adapt_obs_sigma=args.adapt_obs_sigma,
        min_history=args.min_history, k_tier_schedule=args.k_tier_schedule,
        pre_vols=args.pre_vols, max_history=args.max_history,
        strict_disjoint_pre=args.strict_disjoint_pre,
        stim_loss_weight=args.stim_loss_weight, hop1_loss_weight=args.hop1_loss_weight,
        drift_gate_mode=args.drift_gate,
        num_samples=args.num_samples, eval_seed=args.eval_seed,
    )

    result: Dict = {
        "cache": args.cache,
        "split_file": args.split_file,
        "checkpoint": args.checkpoint,
        "tier": args.tier,
        "substeps": int(args.substeps),
        "intact_zero_shot": intact_zero,
        "intact_online_ttt": intact_ttt,
    }

    if args.shuffle_checkpoint is not None:
        shuf_model = _load_checkpoint_model(
            args.shuffle_checkpoint, args, shuffle_seed=args.shuffle_seed, device=device
        )
        shuf_zero = evaluate_online_ttt(
            shuf_model, dataset, test_indices, device, adjacency, np_adjacency,
            tier=args.tier, ttt_steps=0, pre_vols=args.pre_vols,
            strict_disjoint_pre=args.strict_disjoint_pre, drift_gate_mode=args.drift_gate,
            num_samples=args.num_samples, eval_seed=args.eval_seed,
        )
        shuf_ttt = evaluate_online_ttt(
            shuf_model, dataset, test_indices, device, adjacency, np_adjacency,
            tier=args.tier, ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, kl_weight=args.kl_weight,
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
