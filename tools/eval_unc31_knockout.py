#!/usr/bin/env python3
"""Part 1: Zero-shot WT -> unc-31 evaluation and in-silico extrasynaptic knockout ablation."""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_c302 import (
    _prestim_baseline,
    compute_stratified_metrics,
    evaluate_predictions,
    extract_and_expand_posterior_samples,
)
from tools.kernel_metrics import compute_matched_kernel_difference
from tools.noise_ceiling import _compute_strata_masks, _split_half_ceiling
from utils import RandiTrialDataset

RUNS_BASE = "/Users/yanlu/Documents/glowing-journey_runs/phase25_decoupled_extrasyn_fold0_2026-11-25"


def _build_ckpt_model(
    ckpt_path: str,
    substeps: int,
    use_prestim: Optional[bool] = None,
    use_extrasyn: Optional[bool] = None,
    shuffle_seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    pre_vols: Optional[int] = None,
    evoked_syn_priors: bool = False,
    tau_np_prior: float = 4.0,
    tau_ma_prior: float = 1.0,
    tau_mod_prior_std: float = 0.5,
):
    pyro.clear_param_store()
    snap = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms_keys = set(snap["MODEL_STATE"].keys())
    per_neuron_stim_gain = any("stim_gain_vec" in k for k in ms_keys)
    per_neuron_kinetics = any("tau_ca_vec" in k or "k_off_vec" in k or (k == "guide.tau_ca_loc_unconstrained" and snap["MODEL_STATE"][k].numel() > 1) for k in ms_keys)
    per_edge_release = any("V_rest_edge" in k or "v_rest_edge_loc" in k for k in ms_keys)
    split_extrasyn = any("extrasyn_np_gain" in k or "extrasyn_ma_gain" in k for k in ms_keys)
    eff_extrasyn = bool(use_extrasyn) if use_extrasyn is not None else (split_extrasyn or any("extrasyn_gain" in k for k in ms_keys))
    eff_prestim = bool(use_prestim) if use_prestim is not None else any("decoder.prestim_encoder." in k for k in ms_keys)
    prestim_osc = any("decoder.prestim_encoder.osc_out" in k for k in ms_keys)
    use_syn_filter = any("tau_syn_loc" in k for k in ms_keys)
    use_ssm = any("decoder.latent_ssm." in k for k in ms_keys)
    direct_guide = True
    dimensionless = "guide.neuron_G_loc_unconstrained" not in ms_keys
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
    ssm_art_dim = (
        int(snap["MODEL_STATE"][art_key].numel())
        if art_key is not None
        else 8
    )
    ssm_vtraj = any("decoder.latent_ssm.q_traj_gru." in k for k in ms_keys)
    ssm_smooth = any("decoder.latent_ssm.q_smooth_gru." in k for k in ms_keys)
    eff_pre = int(pre_vols) if pre_vols is not None else 10
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
        ssm_dim = int(snap["MODEL_STATE"][decay_key].numel())
    else:
        ssm_dim = 16

    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=eff_extrasyn,
        per_edge_release=per_edge_release,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
        evoked_syn_priors=evoked_syn_priors,
        dimensionless=dimensionless,
        leak_gauge=leak_gauge,
        split_extrasynaptic=split_extrasyn,
        use_syn_filter=use_syn_filter,
        tau_np_prior=tau_np_prior,
        tau_ma_prior=tau_ma_prior,
        tau_mod_prior_std=tau_mod_prior_std,
        shuffle_seed=shuffle_seed,
    )
    rec = c302.RecurrentNematode(
        step,
        dt_obs=0.5,
        substeps=int(substeps),
        use_prestim_encoder=eff_prestim,
        pre_vols=eff_pre,
        prestim_rank=16,
        variational_prestim=eff_prestim,
        prestim_oscillatory=prestim_osc,
        per_neuron_kinetics=per_neuron_kinetics,
        per_edge_release=per_edge_release,
        stim_actuator=True,
        stim_tau_rise=0.08,
        stim_tau_decay=0.4,
        use_latent_ssm=use_ssm,
        latent_ssm_dim=ssm_dim,
        ssm_encoder=ssm_encoder,
        ssm_split_latent=ssm_split,
        ssm_artifact_dim=ssm_art_dim,
        ssm_variational_traj=ssm_vtraj,
        ssm_smoother=ssm_smooth,
    )
    model = vae.VAElegans(
        rec,
        rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC),
        len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=direct_guide,
    ).to(device)
    model.load_state_dict(snap["MODEL_STATE"])
    pyro.get_param_store().set_state(snap["PARAM_STORE"])
    model.eval()
    return model


def _eval_ckpt(
    ckpt_path: str,
    loader: DataLoader,
    adjacency: np.ndarray,
    np_adjacency: np.ndarray,
    substeps: int,
    use_prestim: bool,
    use_extrasyn: bool,
    shuffle_seed: Optional[int],
    num_samples: int = 8,
    eval_seed: int = 42,
    knockout_extrasyn: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    t0 = time.time()
    model = _build_ckpt_model(ckpt_path, substeps, use_prestim, use_extrasyn, shuffle_seed, device)
    if knockout_extrasyn:
        # Draw posterior samples from original guide with deterministic seed, zero extrasyn_gain inside posterior_samples dict
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        pyro.set_rng_seed(eval_seed)
        guide = model.guide
        y_trues, y_preds, y_pred_cfs, masks, stims = [], [], [], [], []
        extra_zeros = torch.zeros(len(model.decoder.model.synapse.Generic_SRC), device=device)
        with torch.no_grad():
            for batch in loader:
                stim, y, mask, noise = [b.to(device) for b in batch[:4]]
                B = stim.shape[0]
                guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
                if use_prestim:
                    guide_kw["pre_stim_obs"] = y[:, :10, :]
                    guide_kw["pre_stim_mask"] = mask[:, :10, :]
                stacked_kw = dict(
                    ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
                    mask=None, y=None,
                    noise_scale=torch.cat([noise, noise], dim=0),
                    sample_obs=False,
                )
                if use_prestim:
                    stacked_kw["pre_stim_obs"] = torch.cat([y[:, :10, :], y[:, :10, :]], dim=0)
                    stacked_kw["pre_stim_mask"] = torch.cat([mask[:, :10, :], mask[:, :10, :]], dim=0)
                ps_acc, pz_acc = None, None
                for _ in range(num_samples):
                    tr = pyro.poutine.trace(guide).get_trace(**guide_kw)
                    ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)
                    ps["decoder.model.synapse.extrasyn_gain"] = extra_zeros
                    out = pyro.condition(model, data=ps)(**stacked_kw)
                    ps_s = out[:B]
                    pz_s = out[B:]
                    ps_acc = ps_s if ps_acc is None else ps_acc + ps_s
                    pz_acc = pz_s if pz_acc is None else pz_acc + pz_s
                y_trues.append(y.cpu().numpy())
                y_preds.append((ps_acc / num_samples).cpu().numpy())
                y_pred_cfs.append((pz_acc / num_samples).cpu().numpy())
                masks.append(mask.bool().cpu().numpy())
                stims.append(stim.cpu().numpy())
        m = compute_stratified_metrics(
            np.concatenate(y_trues, axis=0),
            np.concatenate(y_preds, axis=0),
            np.concatenate(masks, axis=0),
            stim=np.concatenate(stims, axis=0),
            pre_vols=10,
            adjacency=adjacency,
            evoked_baseline="mean",
            y_pred_counterfactual=np.concatenate(y_pred_cfs, axis=0),
            np_adjacency=np_adjacency,
            dt_obs=0.5,
        )
    else:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        pyro.set_rng_seed(eval_seed)
        m = evaluate_predictions(
            model, loader, device, guide=model.guide,
            num_samples=num_samples, pre_vols=10, adjacency=adjacency, np_adjacency=np_adjacency,
        )

    keep = [
        "r2", "inlier_r2", "post_r2", "post_inlier_r2", "post_inlier_r",
        "evoked_post_r2", "evoked_post_r",
        "evoked_stim_r2", "evoked_stim_r", "stim_r",
        "evoked_hop1_r2", "evoked_hop1_r", "evoked_hop1_amp_ratio",
        "evoked_hop2_r2", "evoked_hop2_r",
        "pair_evoked_hop1_count", "pair_evoked_hop1_r", "pair_evoked_hop1_r2", "pair_evoked_hop1_sign_acc",
        "kernel_hop1_n_pairs", "kernel_hop1_r", "kernel_hop1_r2", "kernel_hop1_pair_curve_r_mean",
        "kernel_hop1_auc_r", "kernel_hop1_peak_amp_r", "kernel_hop1_peak_latency_mae_s", "kernel_hop1_fft_r",
        "evoked_np_only_count", "evoked_np_only_r2", "evoked_np_only_r",
        "evoked_np_no_hop12_count", "evoked_np_no_hop12_r2", "evoked_np_no_hop12_r",
        "evoked_np_no_wired_path_count", "evoked_np_no_wired_path_r2",
    ]
    out = {k: m[k] for k in keep if k in m}
    out["eval_wall_s"] = float(time.time() - t0)
    return out


def _gate_summary(intact_vals: List[float], shuf_vals: List[float]) -> Dict[str, float]:
    ia = np.asarray(intact_vals, dtype=float)
    sa = np.asarray(shuf_vals, dtype=float)
    i_mean = float(ia.mean())
    s_mean = float(sa.mean())
    i_std = float(ia.std(ddof=1)) if ia.size > 1 else 0.0
    s_std = float(sa.std(ddof=1)) if sa.size > 1 else 0.0
    sem = float(np.sqrt((i_std ** 2) / max(ia.size, 1) + (s_std ** 2) / max(sa.size, 1)))
    delta = i_mean - s_mean
    return {
        "intact_mean": i_mean,
        "intact_std": i_std,
        "shuffle_mean": s_mean,
        "shuffle_std": s_std,
        "delta_mean": delta,
        "pooled_sem": sem,
        "snr_x_sem": float(delta / sem) if sem > 1e-12 else 0.0,
        "pass": bool(i_mean > 0 and delta > 0 and delta > sem),
    }


def _eval_split_knockout_rollout(
    model: vae.VAElegans,
    loader: DataLoader,
    adjacency: np.ndarray,
    np_adj: np.ndarray,
    ma_adj: np.ndarray,
    pre_vols: int,
    num_samples: int = 8,
    eval_seed: int = 42,
    zero_np: bool = False,
    zero_ma: bool = False,
    device: torch.device = torch.device("cpu"),
    return_arrays: bool = False,
):
    """Run paired counterfactual rollout with optional NP and/or MA gain ablation."""
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    pyro.set_rng_seed(eval_seed)
    dec = model.decoder
    syn = dec.model.synapse
    has_pre = bool(getattr(dec, "use_prestim_encoder", False))
    ssm_mod = getattr(dec, "latent_ssm", None)
    has_ssm = ssm_mod is not None
    has_history = has_pre or has_ssm

    y_trues, y_preds, y_pred_cfs, masks, stims = [], [], [], [], []
    np_zeros = torch.zeros(len(getattr(syn, "NP_SRC", [])), device=device) if zero_np else None
    ma_zeros = torch.zeros(len(getattr(syn, "MA_SRC", [])), device=device) if zero_ma else None
    if has_ssm:
        total_z_dim = int(getattr(ssm_mod, "total_z_dim", ssm_mod.z_dim))
    with torch.no_grad():
        for batch in loader:
            stim, y, mask, noise = [b.to(device) for b in batch[:4]]
            B = stim.shape[0]
            t_cut = min(int(pre_vols), y.shape[1])
            guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
            if has_pre:
                guide_kw["pre_stim_obs"] = y[:, :t_cut, :]
                guide_kw["pre_stim_mask"] = mask[:, :t_cut, :]
            stacked_kw = dict(
                ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
                mask=None, y=None,
                noise_scale=torch.cat([noise, noise], dim=0),
                sample_obs=False,
            )
            if has_history:
                stacked_kw["pre_stim_obs"] = torch.cat([y[:, :t_cut, :], y[:, :t_cut, :]], dim=0)
                stacked_kw["pre_stim_mask"] = torch.cat([mask[:, :t_cut, :], mask[:, :t_cut, :]], dim=0)
            ps_acc, pz_acc = None, None
            for _ in range(max(1, int(num_samples))):
                tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
                ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)
                if np_zeros is not None and len(np_zeros) > 0:
                    ps["decoder.model.synapse.extrasyn_np_gain"] = np_zeros
                if ma_zeros is not None and len(ma_zeros) > 0:
                    ps["decoder.model.synapse.extrasyn_ma_gain"] = ma_zeros
                if has_ssm:
                    eps = torch.randn((B, stim.shape[1] + 1, total_z_dim), device=device, dtype=stim.dtype)
                    out = pyro.condition(model, data=ps)(**stacked_kw, ssm_eps=torch.cat([eps, eps], dim=0))
                else:
                    out = pyro.condition(model, data=ps)(**stacked_kw)
                ps_s, pz_s = out[:B], out[B:]
                ps_acc = ps_s if ps_acc is None else ps_acc + ps_s
                pz_acc = pz_s if pz_acc is None else pz_acc + pz_s
            y_trues.append(y.cpu().numpy())
            y_preds.append((ps_acc / max(1, int(num_samples))).cpu().numpy())
            y_pred_cfs.append((pz_acc / max(1, int(num_samples))).cpu().numpy())
            masks.append(mask.bool().cpu().numpy())
            stims.append(stim.cpu().numpy())

    yt = np.concatenate(y_trues, axis=0)
    yp = np.concatenate(y_preds, axis=0)
    ypc = np.concatenate(y_pred_cfs, axis=0)
    mk = np.concatenate(masks, axis=0)
    st = np.concatenate(stims, axis=0)
    dt_obs = float(getattr(dec, "dt_obs", 0.5))

    m_np = compute_stratified_metrics(
        yt, yp, mk, stim=st, pre_vols=pre_vols, adjacency=adjacency,
        evoked_baseline="mean", y_pred_counterfactual=ypc,
        np_adjacency=np_adj, ma_adjacency=ma_adj, dt_obs=dt_obs,
    )
    m_ma = compute_stratified_metrics(
        yt, yp, mk, stim=st, pre_vols=pre_vols, adjacency=adjacency,
        evoked_baseline="mean", y_pred_counterfactual=ypc, np_adjacency=ma_adj, dt_obs=dt_obs,
    )
    out = {
        "post_inlier_r2": float(m_np.get("post_inlier_r2", float("nan"))),
        "evoked_hop1_r2": float(m_np.get("evoked_hop1_r2", float("nan"))),
        "evoked_np_stratum_r2": float(m_np.get("evoked_np_only_r2", float("nan"))),
        "evoked_np_stratum_r": float(m_np.get("evoked_np_only_r", float("nan"))),
        "evoked_np_no_hop12_r2": float(m_np.get("evoked_np_no_hop12_r2", float("nan"))),
        "evoked_np_stratum_count": int(m_np.get("evoked_np_only_count", 0)),
        "evoked_ma_stratum_r2": float(m_ma.get("evoked_np_only_r2", float("nan"))),
        "evoked_ma_stratum_r": float(m_ma.get("evoked_np_only_r", float("nan"))),
        "evoked_ma_no_hop12_r2": float(m_ma.get("evoked_np_no_hop12_r2", float("nan"))),
        "evoked_ma_stratum_count": int(m_ma.get("evoked_np_only_count", 0)),
        "evoked_np_exclusive_r2": float(m_np.get("evoked_np_exclusive_r2", float("nan"))),
        "evoked_np_exclusive_r": float(m_np.get("evoked_np_exclusive_r", float("nan"))),
        "evoked_np_exclusive_count": int(m_np.get("evoked_np_exclusive_count", 0)),
        "evoked_np_exclusive_no_hop12_r2": float(m_np.get("evoked_np_exclusive_no_hop12_r2", float("nan"))),
        "evoked_ma_exclusive_r2": float(m_np.get("evoked_ma_exclusive_r2", float("nan"))),
        "evoked_ma_exclusive_r": float(m_np.get("evoked_ma_exclusive_r", float("nan"))),
        "evoked_ma_exclusive_count": int(m_np.get("evoked_ma_exclusive_count", 0)),
        "evoked_np_ma_overlap_r2": float(m_np.get("evoked_np_ma_overlap_r2", float("nan"))),
        "evoked_np_ma_overlap_r": float(m_np.get("evoked_np_ma_overlap_r", float("nan"))),
        "evoked_np_ma_overlap_count": int(m_np.get("evoked_np_ma_overlap_count", 0)),
        "kernel_hop1_r": float(m_np.get("kernel_hop1_r", float("nan"))),
        "kernel_hop1_peak_latency_mae_s": float(m_np.get("kernel_hop1_peak_latency_mae_s", float("nan"))),
    }
    if return_arrays:
        return out, (yt, yp - ypc, mk, st)
    return out


def evaluate_np_ma_double_dissociation(
    model: vae.VAElegans,
    loader_wt: DataLoader,
    loader_u31: Optional[DataLoader],
    adjacency: np.ndarray,
    np_adj: np.ndarray,
    ma_adj: np.ndarray,
    pre_vols: int = 10,
    num_samples: int = 8,
    eval_seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Quantify NP vs. MA double dissociation (full, np_zero, ma_zero, both_zero)."""
    guide = model.guide
    timescales = {}
    if hasattr(guide, "tau_np_loc"):
        timescales["tau_np_median_s"] = float(torch.exp(guide.tau_np_loc.detach()).cpu().item())
    if hasattr(guide, "tau_ma_loc"):
        timescales["tau_ma_median_s"] = float(torch.exp(guide.tau_ma_loc.detach()).cpu().item())
    if hasattr(guide, "tau_syn_loc"):
        t_syn = torch.exp(guide.tau_syn_loc.detach()).cpu().numpy()
        timescales["tau_syn_median_s"] = float(np.median(t_syn))

    def _run_cohort(ldr: DataLoader, capture_full: bool = False):
        full_ret = _eval_split_knockout_rollout(
            model, ldr, adjacency, np_adj, ma_adj, pre_vols, num_samples, eval_seed,
            zero_np=False, zero_ma=False, device=device, return_arrays=capture_full,
        )
        if capture_full:
            full, full_arrs = full_ret
        else:
            full, full_arrs = full_ret, None
        np_z = _eval_split_knockout_rollout(
            model, ldr, adjacency, np_adj, ma_adj, pre_vols, num_samples, eval_seed,
            zero_np=True, zero_ma=False, device=device,
        )
        ma_z = _eval_split_knockout_rollout(
            model, ldr, adjacency, np_adj, ma_adj, pre_vols, num_samples, eval_seed,
            zero_np=False, zero_ma=True, device=device,
        )
        both_z = _eval_split_knockout_rollout(
            model, ldr, adjacency, np_adj, ma_adj, pre_vols, num_samples, eval_seed,
            zero_np=True, zero_ma=True, device=device,
        )
        res_c = {
            "full": full,
            "np_zero": np_z,
            "ma_zero": ma_z,
            "both_zero": both_z,
            "delta_np_knockout": {
                "evoked_np_stratum_r2": float(full["evoked_np_stratum_r2"] - np_z["evoked_np_stratum_r2"]),
                "evoked_np_no_hop12_r2": float(full["evoked_np_no_hop12_r2"] - np_z["evoked_np_no_hop12_r2"]),
                "evoked_np_stratum_r": float(full["evoked_np_stratum_r"] - np_z["evoked_np_stratum_r"]),
                "evoked_np_exclusive_r2": float(full["evoked_np_exclusive_r2"] - np_z["evoked_np_exclusive_r2"]),
            },
            "delta_ma_knockout": {
                "evoked_ma_stratum_r2": float(full["evoked_ma_stratum_r2"] - ma_z["evoked_ma_stratum_r2"]),
                "evoked_ma_no_hop12_r2": float(full["evoked_ma_no_hop12_r2"] - ma_z["evoked_ma_no_hop12_r2"]),
                "evoked_ma_stratum_r": float(full["evoked_ma_stratum_r"] - ma_z["evoked_ma_stratum_r"]),
                "evoked_ma_exclusive_r2": float(full["evoked_ma_exclusive_r2"] - ma_z["evoked_ma_exclusive_r2"]),
            },
        }
        return (res_c, full_arrs) if capture_full else res_c

    wt_res, wt_arrs = _run_cohort(loader_wt, capture_full=(loader_u31 is not None))
    res = {"timescales": timescales, "wt": wt_res}
    if loader_u31 is not None:
        u31_res, u31_arrs = _run_cohort(loader_u31, capture_full=True)
        res["unc31"] = u31_res
        res["matched_delta_kernel"] = compute_matched_kernel_difference(
            wt_arrs[0], wt_arrs[1], wt_arrs[2], wt_arrs[3],
            u31_arrs[0], u31_arrs[1], u31_arrs[2], u31_arrs[3],
            adjacency=adjacency,
            np_adjacency=np_adj,
            ma_adjacency=ma_adj,
            pre_vols=pre_vols,
            dt_obs=float(getattr(model.decoder, "dt_obs", 0.5)),
        )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unc31-cache", default="/tmp/randi_unc31_causal.npz")
    p.add_argument("--wt-cache", default="/tmp/randi_causal.npz")
    p.add_argument("--split-file", default=f"{RUNS_BASE}/p15_5f_s1/fold_0.json")
    p.add_argument("--out", default=os.path.join(REPO, "report/data/phase4_unc31_knockout_eval.json"))
    p.add_argument("--split-checkpoint", default=None, help="Run NP/MA double dissociation on one split-extrasynaptic checkpoint")
    p.add_argument("--substeps", default=4, type=int)
    p.add_argument("--pre-vols", default=10, type=int)
    p.add_argument("--num-samples", default=8, type=int)
    p.add_argument("--evoked-syn-priors", action="store_true")
    p.add_argument("--tau-np-prior", default=4.0, type=float)
    p.add_argument("--tau-ma-prior", default=1.0, type=float)
    p.add_argument("--tau-mod-prior-std", default=0.5, type=float)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_u31 = RandiTrialDataset.load(args.unc31_cache)
    ds_wt = RandiTrialDataset.load(args.wt_cache)
    with open(args.split_file) as fh:
        wt_split = json.load(fh)
    wt_test_idx = [int(i) for i in wt_split["test"]]

    loader_u31 = DataLoader(ds_u31, batch_size=len(ds_u31), shuffle=False)
    loader_wt = DataLoader(Subset(ds_wt, wt_test_idx), batch_size=len(wt_test_idx), shuffle=False)

    intact = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adj = intact.synapse.adjacency_matrix(intact.neuronSize)
    np_adj = intact.synapse.np_adjacency_matrix(intact.neuronSize)
    ma_adj = intact.synapse.ma_adjacency_matrix(intact.neuronSize)
    extra_adj = intact.synapse.extrasynaptic_adjacency_matrix(intact.neuronSize)

    if args.split_checkpoint is not None:
        model = _build_ckpt_model(
            args.split_checkpoint,
            substeps=args.substeps,
            use_prestim=None,
            use_extrasyn=True,
            shuffle_seed=None,
            device=device,
            pre_vols=args.pre_vols,
            evoked_syn_priors=args.evoked_syn_priors,
            tau_np_prior=args.tau_np_prior,
            tau_ma_prior=args.tau_ma_prior,
            tau_mod_prior_std=args.tau_mod_prior_std,
        )
        dd = evaluate_np_ma_double_dissociation(
            model, loader_wt, loader_u31, adj, np_adj, ma_adj,
            pre_vols=args.pre_vols, num_samples=args.num_samples, device=device,
        )
        out = {
            "checkpoint": args.split_checkpoint,
            "wt_cache": args.wt_cache,
            "unc31_cache": args.unc31_cache,
            "split_file": args.split_file,
            "substeps": int(args.substeps),
            "pre_vols": int(args.pre_vols),
            "num_samples": int(args.num_samples),
            "double_dissociation": dd,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(json.dumps(out, indent=2))
        return

    # A. unc-31 Split-Half Noise Ceiling
    print("=== A. Computing unc-31 split-half noise ceiling ===", flush=True)
    y_u = ds_u31.y.numpy()
    m_u = ds_u31.obs_mask.numpy().astype(bool)
    st_u = ds_u31.stim_inputs.numpy()
    rec_u = np.array([int(m["rec_id"]) for m in ds_u31.meta], dtype=np.int64)
    stim_u = np.array([int(m.get("stim_neuron_idx", -1)) for m in ds_u31.meta], dtype=np.int64)
    yt_ev_u = y_u - _prestim_baseline(y_u, m_u, 10, mode="linear")
    strata_u = _compute_strata_masks(st_u, adj, extra_adj)
    nc_u31 = {
        "n_recordings": int(len(np.unique(rec_u))),
        "n_trials": int(len(ds_u31)),
        "n_unique_stim_neurons": int(len(np.unique(stim_u))),
        "within_recording": _split_half_ceiling(
            yt_ev_u, m_u, strata_u,
            group_keys=[(int(r), int(s)) for r, s in zip(rec_u, stim_u)],
            unit_ids=None, t0=10, t_ev_end=26,
        ),
        "across_recording": _split_half_ceiling(
            yt_ev_u, m_u, strata_u,
            group_keys=[(int(s),) for s in stim_u],
            unit_ids=rec_u, t0=10, t_ev_end=26,
        ),
    }

    # B. Zero-Shot WT -> unc-31 evaluation across Phase 1.5 & Phase 2.5 checkpoints
    print("=== B. Evaluating WT -> unc-31 zero-shot across substeps=[2, 4] ===", flush=True)
    u31_zeroshot: Dict[str, dict] = {}
    for sub in (2, 4):
        sub_res: Dict[str, dict] = {}
        # All 3 intact + 3 shuffle seeds on Phase 1.5 S1
        p15_intact, p15_shuf = [], []
        for s_seed in (42, 43, 44):
            ckpt = f"{RUNS_BASE}/p15_5f_s1/fold0/intact_seed{s_seed}/vae_epoch_45.pt"
            res = _eval_ckpt(ckpt, loader_u31, adj, extra_adj, sub, False, False, None, args.num_samples, device=device)
            p15_intact.append(res)
            print(f"  [sub={sub}] p15_s1 intact_s{s_seed} -> u31: hop1_r2={res.get('evoked_hop1_r2'):.5f} kernel_r={res.get('kernel_hop1_r',0):.4f}", flush=True)
        for sh_idx in (7, 8, 9):
            ep_sh = 40 if sh_idx == 9 else 45
            ckpt = f"{RUNS_BASE}/p15_5f_s1/fold0/shuffle{sh_idx}_seed42/vae_epoch_{ep_sh}.pt"
            res = _eval_ckpt(ckpt, loader_u31, adj, extra_adj, sub, False, False, sh_idx, args.num_samples, device=device)
            p15_shuf.append(res)
            print(f"  [sub={sub}] p15_s1 shuffle{sh_idx} -> u31: hop1_r2={res.get('evoked_hop1_r2'):.5f} kernel_r={res.get('kernel_hop1_r',0):.4f}", flush=True)
        sub_res["p15_f0_s1"] = {
            "intact_runs": p15_intact,
            "shuffle_runs": p15_shuf,
            "gate_evoked_hop1_r2": _gate_summary(
                [r["evoked_hop1_r2"] for r in p15_intact],
                [r["evoked_hop1_r2"] for r in p15_shuf],
            ),
        }

        for tag, rel_in, rel_sh, pre, ex in [
            ("p15_f0_s2", "p15_5f_s2/fold0/intact_seed42/vae_epoch_10.pt", "p15_5f_s2/fold0/shuffle7_seed42/vae_epoch_10.pt", True, False),
            ("p25_f0_s1b", "p25_f0_s1b/fold0/intact_seed42/vae_epoch_10.pt", "p25_f0_s1b/fold0/shuffle7_seed42/vae_epoch_15.pt", False, True),
            ("p25_f0_s2", "p25_f0_s2/fold0/intact_seed42/vae_epoch_10.pt", "p25_f0_s2/fold0/shuffle7_seed42/vae_epoch_10.pt", True, True),
        ]:
            r_in = _eval_ckpt(f"{RUNS_BASE}/{rel_in}", loader_u31, adj, extra_adj, sub, pre, ex, None, args.num_samples, device=device)
            r_sh = _eval_ckpt(f"{RUNS_BASE}/{rel_sh}", loader_u31, adj, extra_adj, sub, pre, ex, 7, args.num_samples, device=device)
            sub_res[tag] = {"intact_seed42": r_in, "shuffle7_seed42": r_sh}
            print(
                f"  [sub={sub}] {tag} -> u31: intact hop1_r2={r_in['evoked_hop1_r2']:.5f} np_only_r2={r_in.get('evoked_np_only_r2', 0):.5f} "
                f"post_inlier_r2={r_in.get('post_inlier_r2', 0):.4f} vs shuf hop1_r2={r_sh['evoked_hop1_r2']:.5f}",
                flush=True,
            )
        u31_zeroshot[f"substeps_{sub}"] = sub_res

    # C. In-silico unc-31 knockout ablation on WT vs unc-31
    print("\n=== C. In-silico unc-31 extrasynaptic knockout (extrasyn_gain -> 0) on WT vs unc-31 ===", flush=True)
    ko_ablation: Dict[str, dict] = {}
    for tag, rel_in, pre in [
        ("p25_f0_s1b", "p25_f0_s1b/fold0/intact_seed42/vae_epoch_10.pt", False),
        ("p25_f0_s2", "p25_f0_s2/fold0/intact_seed42/vae_epoch_10.pt", True),
    ]:
        ckpt = f"{RUNS_BASE}/{rel_in}"
        ko_tag = {}
        for sub in (2, 4):
            wt_full = _eval_ckpt(ckpt, loader_wt, adj, extra_adj, sub, pre, True, None, args.num_samples, knockout_extrasyn=False, device=device)
            wt_zero = _eval_ckpt(ckpt, loader_wt, adj, extra_adj, sub, pre, True, None, args.num_samples, knockout_extrasyn=True, device=device)
            u31_full = _eval_ckpt(ckpt, loader_u31, adj, extra_adj, sub, pre, True, None, args.num_samples, knockout_extrasyn=False, device=device)
            u31_zero = _eval_ckpt(ckpt, loader_u31, adj, extra_adj, sub, pre, True, None, args.num_samples, knockout_extrasyn=True, device=device)

            def _deltas(a: dict, b: dict) -> dict:
                keys = ("evoked_hop1_r2", "evoked_hop2_r", "evoked_np_only_r2", "evoked_np_only_r", "evoked_np_no_hop12_r2", "evoked_np_no_hop12_r", "kernel_hop1_r", "post_inlier_r2")
                return {k: float(a[k] - b[k]) for k in keys if k in a and k in b}

            ko_tag[f"substeps_{sub}"] = {
                "wt_full": wt_full,
                "wt_extrasyn_zero": wt_zero,
                "wt_delta_full_minus_zero": _deltas(wt_full, wt_zero),
                "unc31_full": u31_full,
                "unc31_extrasyn_zero": u31_zero,
                "unc31_delta_full_minus_zero": _deltas(u31_full, u31_zero),
            }
            print(
                f"  [sub={sub}] {tag} Δ(full-zero): WT np_only_r2={ko_tag[f'substeps_{sub}']['wt_delta_full_minus_zero'].get('evoked_np_only_r2',0):+.5f} "
                f"(np_no_h12={ko_tag[f'substeps_{sub}']['wt_delta_full_minus_zero'].get('evoked_np_no_hop12_r2',0):+.5f}) | "
                f"unc-31 np_only_r2={ko_tag[f'substeps_{sub}']['unc31_delta_full_minus_zero'].get('evoked_np_only_r2',0):+.5f} "
                f"(np_no_h12={ko_tag[f'substeps_{sub}']['unc31_delta_full_minus_zero'].get('evoked_np_no_hop12_r2',0):+.5f})",
                flush=True,
            )
        ko_ablation[tag] = ko_tag

    out = {
        "unc31_cache": args.unc31_cache,
        "wt_cache": args.wt_cache,
        "split_file": args.split_file,
        "num_samples": args.num_samples,
        "unc31_noise_ceiling": nc_u31,
        "wt_to_unc31_zeroshot": u31_zeroshot,
        "insilico_knockout_ablation": ko_ablation,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
