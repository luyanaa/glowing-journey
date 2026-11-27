#!/usr/bin/env python3
"""Step 4.1: Split-half noise ceiling + pre-stim window, horizon, and observation-mask ablations."""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyro
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_c302 import _prestim_baseline, compute_stratified_metrics
from utils import AtlasLoader, RandiTrialDataset, resolve_default_data_path

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


def _compute_strata_masks(
    stim: np.ndarray,
    adjacency: np.ndarray,
    np_adjacency: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return boolean (B, N) masks per anatomical stratum using abs-sum stim detection."""
    n = adjacency.shape[0]
    stim_target = np.abs(stim).sum(axis=1) > 0
    adj1 = adjacency.astype(bool)
    adj2 = ((adj1.astype(np.int32) @ adj1.astype(np.int32)) > 0) & (~adj1) & (~np.eye(n, dtype=bool))
    hop1 = ((stim_target.astype(np.int32) @ adj1.astype(np.int32)) > 0) & (~stim_target)
    hop2 = ((stim_target.astype(np.int32) @ adj2.astype(np.int32)) > 0) & (~stim_target) & (~hop1)
    wired_reach = adj1.copy()
    while True:
        nxt = wired_reach | ((wired_reach.astype(np.int32) @ adj1.astype(np.int32)) > 0)
        if np.array_equal(nxt, wired_reach):
            break
        wired_reach = nxt
    wired_neu = (stim_target.astype(np.int32) @ wired_reach.astype(np.int32)) > 0
    np_hit = (stim_target.astype(np.int32) @ np_adjacency.astype(np.int32)) > 0
    return {
        "stim": stim_target,
        "hop1": hop1,
        "hop2": hop2,
        "np_only": np_hit & (~stim_target) & (~hop1),
        "np_no_hop12": np_hit & (~stim_target) & (~hop1) & (~hop2),
        "np_no_wired_path": np_hit & (~stim_target) & (~wired_neu),
        "unconnected": (~stim_target) & (~hop1) & (~hop2),
    }


def _split_half_ceiling(
    yt_ev: np.ndarray,
    valid: np.ndarray,
    strata: Dict[str, np.ndarray],
    group_keys: List[Tuple],
    unit_ids: Optional[np.ndarray],
    t0: int,
    t_ev_end: int,
    n_splits: int = 20,
    seed: int = 42,
) -> Dict[str, dict]:
    """Split-half reliability (within-recording if unit_ids=None else across unit_ids)."""
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple, List[int]] = {}
    for i, gk in enumerate(group_keys):
        groups.setdefault(gk, []).append(i)

    T_ev = t_ev_end - t0
    out: Dict[str, dict] = {}
    for sname, s_neu in strata.items():
        r_halves = []
        r_sbs = []
        r2_trial_ceilings = []
        loo_single_rs = []
        n_groups_used = 0
        n_pairs_points = 0

        # Pre-check eligible groups and neuron columns observed in >= 2 units
        eligible = []
        for gk, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_arr = np.asarray(idxs, dtype=np.int64)
            u_arr = np.arange(len(idxs)) if unit_ids is None else unit_ids[idx_arr]
            uniq_u = np.unique(u_arr)
            if uniq_u.size < 2:
                continue
            m_ev = valid[idx_arr, t0:t_ev_end, :] & s_neu[idx_arr, None, :]
            # Neuron observed across >= 2 distinct units
            u_obs = np.zeros((uniq_u.size, yt_ev.shape[2]), dtype=bool)
            for ui, uval in enumerate(uniq_u):
                sel = u_arr == uval
                u_obs[ui] = m_ev[sel].any(axis=(0, 1))
            keep_n = np.flatnonzero(u_obs.sum(axis=0) >= 2)
            if keep_n.size == 0:
                continue
            eligible.append((idx_arr, u_arr, uniq_u, keep_n))
            n_groups_used += 1

        if not eligible:
            out[sname] = {"n_groups": 0}
            continue

        # Leave-one-unit-out single-trial baseline correlation & total single-trial variance
        loo_preds, loo_trues = [], []
        for idx_arr, u_arr, uniq_u, keep_n in eligible:
            y_sub = yt_ev[idx_arr, t0:t_ev_end, :][:, :, keep_n]
            m_sub = (valid[idx_arr, t0:t_ev_end, :] & s_neu[idx_arr, None, :])[:, :, keep_n]
            tot_sum = (y_sub * m_sub).sum(axis=0)
            tot_cnt = m_sub.sum(axis=0)
            for b_pos in range(len(idx_arr)):
                uval = u_arr[b_pos]
                if unit_ids is None:
                    other_sum = tot_sum - y_sub[b_pos] * m_sub[b_pos]
                    other_cnt = tot_cnt - m_sub[b_pos].astype(np.int32)
                else:
                    same_u = u_arr == uval
                    other_sum = tot_sum - (y_sub[same_u] * m_sub[same_u]).sum(axis=0)
                    other_cnt = tot_cnt - m_sub[same_u].sum(axis=0)
                ok = m_sub[b_pos] & (other_cnt >= 1)
                if np.any(ok):
                    loo_preds.append(other_sum[ok] / other_cnt[ok])
                    loo_trues.append(y_sub[b_pos][ok])

        loo_r = _safe_r(np.concatenate(loo_trues), np.concatenate(loo_preds)) if loo_trues else 0.0
        var_single = float(np.var(np.concatenate(loo_trues))) if loo_trues else 0.0

        for _ in range(n_splits):
            a_vals, b_vals = [], []
            for idx_arr, u_arr, uniq_u, keep_n in eligible:
                perm = rng.permutation(uniq_u.size)
                half = uniq_u.size // 2
                ua = set(uniq_u[perm[:half]].tolist())
                ub = set(uniq_u[perm[half:2 * half]].tolist())
                if not ub:
                    ub = set(uniq_u[perm[half:]].tolist())
                sel_a = np.array([u in ua for u in u_arr])
                sel_b = np.array([u in ub for u in u_arr])

                y_sub = yt_ev[idx_arr, t0:t_ev_end, :][:, :, keep_n]
                m_sub = (valid[idx_arr, t0:t_ev_end, :] & s_neu[idx_arr, None, :])[:, :, keep_n]
                ca = m_sub[sel_a].sum(axis=0)
                cb = m_sub[sel_b].sum(axis=0)
                ok = (ca >= 1) & (cb >= 1)
                if np.any(ok):
                    ma = (y_sub[sel_a] * m_sub[sel_a]).sum(axis=0)[ok] / ca[ok]
                    mb = (y_sub[sel_b] * m_sub[sel_b]).sum(axis=0)[ok] / cb[ok]
                    a_vals.append(ma)
                    b_vals.append(mb)
            if a_vals:
                aa = np.concatenate(a_vals)
                bb = np.concatenate(b_vals)
                n_pairs_points = max(n_pairs_points, int(aa.size))
                rh = _safe_r(aa, bb)
                r_halves.append(rh)
                r_sbs.append(float(2.0 * rh / (1.0 + rh)) if rh > -0.99 else 0.0)
                cov_ab = float(np.mean((aa - aa.mean()) * (bb - bb.mean())))
                r2_ceil = float(np.clip(cov_ab / max(var_single, 1e-12), 0.0, 1.0)) if var_single > 1e-12 else 0.0
                r2_trial_ceilings.append(r2_ceil)

        out[sname] = {
            "n_groups": int(n_groups_used),
            "n_half_points": int(n_pairs_points),
            "half_split_r_mean": float(np.mean(r_halves)) if r_halves else 0.0,
            "half_split_r_std": float(np.std(r_halves)) if r_halves else 0.0,
            "spearman_brown_r_mean": float(np.mean(r_sbs)) if r_sbs else 0.0,
            "single_trial_r2_ceiling_est": float(np.mean(r2_trial_ceilings)) if r2_trial_ceilings else 0.0,
            "loo_single_trial_r": float(loo_r),
            "single_trial_evoked_std": float(np.sqrt(max(var_single, 0.0))),
        }
    return out


def compute_noise_ceilings(dataset: RandiTrialDataset, fold_split: dict, adjacency: np.ndarray, np_adjacency: np.ndarray, pre_vols: int = 10) -> Dict:
    y = dataset.y.numpy()
    mask = dataset.obs_mask.numpy().astype(bool)
    stim = dataset.stim_inputs.numpy()
    rec_ids = np.array([int(m["rec_id"]) for m in dataset.meta], dtype=np.int64)
    stim_idx = np.array([int(m.get("stim_neuron_idx", -1)) for m in dataset.meta], dtype=np.int64)

    baseline = _prestim_baseline(y, mask, pre_vols, mode="linear")
    yt_ev = y - baseline
    strata = _compute_strata_masks(stim, adjacency, np_adjacency)

    within = _split_half_ceiling(
        yt_ev, mask, strata,
        group_keys=[(int(r), int(s)) for r, s in zip(rec_ids, stim_idx) if True],
        unit_ids=None,
        t0=pre_vols, t_ev_end=pre_vols + 16,
    )
    across = _split_half_ceiling(
        yt_ev, mask, strata,
        group_keys=[(int(s),) for s in stim_idx],
        unit_ids=rec_ids,
        t0=pre_vols, t_ev_end=pre_vols + 16,
    )

    # Also evaluate empirical Fold-0 train->test population pair-mean baseline on locked test split
    tr_idx = [int(i) for i in fold_split["train"] + fold_split.get("val", [])]
    te_idx = [int(i) for i in fold_split["test"]]
    train_means: Dict[int, np.ndarray] = {}
    for s in np.unique(stim_idx[tr_idx]):
        sel = [i for i in tr_idx if stim_idx[i] == s]
        m_s = mask[sel]
        cnt = np.maximum(m_s.sum(axis=0), 1)
        train_means[int(s)] = (yt_ev[sel] * m_s).sum(axis=0) / cnt

    y_te = y[te_idx]
    m_te = mask[te_idx]
    st_te = stim[te_idx]
    base_te = baseline[te_idx]
    pred_pop = base_te.copy()
    for pos, tid in enumerate(te_idx):
        s = int(stim_idx[tid])
        if s in train_means:
            pred_pop[pos, pre_vols:, :] += train_means[s][pre_vols:, :]
    pop_metrics = compute_stratified_metrics(
        y_te, pred_pop, m_te, stim=st_te, pre_vols=pre_vols, adjacency=adjacency,
        evoked_baseline="linear", np_adjacency=np_adjacency, y_pred_counterfactual=base_te,
    )
    return {
        "within_recording": within,
        "across_recording": across,
        "fold0_train_pair_mean_on_test": {
            k: pop_metrics[k] for k in (
                "evoked_stim_r2", "evoked_stim_r",
                "evoked_hop1_r2", "evoked_hop1_r",
                "pair_evoked_hop1_r", "kernel_hop1_r", "kernel_hop1_auc_r", "kernel_hop1_peak_latency_mae_s",
                "evoked_np_only_r2", "evoked_np_no_wired_path_r2",
            ) if k in pop_metrics
        },
    }


def _load_ckpt(checkpoint_path: str, use_prestim: bool, use_extrasyn: bool, device: torch.device):
    pyro.clear_param_store()
    snap = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ms_keys = set(snap["MODEL_STATE"].keys())
    per_neuron_stim_gain = any("stim_gain_vec" in k for k in ms_keys)
    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=use_extrasyn,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
    )
    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=2,
        use_prestim_encoder=bool(use_prestim), pre_vols=10, prestim_rank=16,
        variational_prestim=bool(use_prestim), prestim_oscillatory=False,
        stim_actuator=True, stim_tau_rise=0.08, stim_tau_decay=0.4,
    )
    model = vae.VAElegans(
        rec, rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC), len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    ).to(device)
    model.load_state_dict(snap["MODEL_STATE"])
    pyro.get_param_store().set_state(snap["PARAM_STORE"])
    model.eval()
    return model


@torch.no_grad()
def _posterior_predict_pair(model, stim: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor, pre_obs: torch.Tensor, pre_mask: torch.Tensor, num_samples: int = 8, seed: int = 42):
    """Return posterior predictive mean (pred_stim, pred_zero) of shape (B, T, N)."""
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
    B = stim.shape[0]
    stacked_kw = dict(
        ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
        mask=None, y=None,
        noise_scale=torch.cat([noise, noise], dim=0),
        sample_obs=False,
    )
    if has_pre:
        stacked_kw["pre_stim_obs"] = torch.cat([pre_obs, pre_obs], dim=0)
        stacked_kw["pre_stim_mask"] = torch.cat([pre_mask, pre_mask], dim=0)
        stacked_kw["drift_gate"] = None
    guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_pre:
        guide_kw["pre_stim_obs"] = pre_obs
        guide_kw["pre_stim_mask"] = pre_mask
        guide_kw["drift_gate"] = None

    p_stim_list, p_zero_list = [], []
    from pyro_svi_c302 import extract_and_expand_posterior_samples
    for _ in range(num_samples):
        tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
        ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)
        cond = pyro.condition(model, data=ps)
        out = cond(**stacked_kw)
        p_stim_list.append(out[:B].cpu().numpy())
        p_zero_list.append(out[B:].cpu().numpy())
    return np.mean(p_stim_list, axis=0), np.mean(p_zero_list, axis=0)


def horizon_and_mask_ablations(dataset: RandiTrialDataset, fold_split: dict, adjacency: np.ndarray, np_adjacency: np.ndarray, device: torch.device, num_samples: int = 8) -> Dict:
    te_idx = [int(i) for i in fold_split["test"]]
    ds_te = Subset(dataset, te_idx)
    stim = torch.stack([ds_te[i][0] for i in range(len(ds_te))], dim=0).to(device)
    y = torch.stack([ds_te[i][1] for i in range(len(ds_te))], dim=0).to(device)
    mask = torch.stack([ds_te[i][2] for i in range(len(ds_te))], dim=0).to(device)
    noise = torch.stack([ds_te[i][3] for i in range(len(ds_te))], dim=0).to(device)
    y_np = y.cpu().numpy()
    m_np = mask.cpu().numpy().astype(bool)
    st_np = stim.cpu().numpy()

    checkpoints = [
        ("phase15_f0_s1", f"{RUNS_BASE}/p15_5f_s1/fold0/intact_seed42/vae_epoch_45.pt", False, False),
        ("phase15_f0_s2", f"{RUNS_BASE}/p15_5f_s2/fold0/intact_seed42/vae_epoch_10.pt", True, False),
        ("phase25_f0_s2", f"{RUNS_BASE}/p25_f0_s2/fold0/intact_seed42/vae_epoch_10.pt", True, True),
    ]

    horizons = [2, 4, 10, 20, 50]  # 1, 2, 5, 10, 25 s
    horizon_res: Dict[str, dict] = {}
    for label, ckpt, use_pre, use_ex in checkpoints:
        model = _load_ckpt(ckpt, use_prestim=use_pre, use_extrasyn=use_ex, device=device)
        ps_mean, pz_mean = _posterior_predict_pair(model, stim, mask, noise, y[:, :10, :], mask[:, :10, :], num_samples=num_samples)
        horizon_res[label] = {}
        for H in horizons:
            end_t = 10 + H
            m_h = compute_stratified_metrics(
                y_np[:, :end_t, :], ps_mean[:, :end_t, :], m_np[:, :end_t, :],
                stim=st_np[:, :end_t, :], pre_vols=10, adjacency=adjacency,
                evoked_baseline="mean", evoked_window_vols=min(H, 16),
                y_pred_counterfactual=pz_mean[:, :end_t, :], np_adjacency=np_adjacency,
            )
            horizon_res[label][f"{H * 0.5:.1f}s"] = {
                k: m_h[k] for k in (
                    "post_r2", "post_inlier_r2", "post_inlier_r",
                    "evoked_stim_r2", "evoked_hop1_r2", "evoked_hop1_r", "kernel_hop1_r",
                ) if k in m_h
            }

    # Add simple open-loop Persistence (mean pre-stim carried forward) baseline across horizons
    pre_mean = (y_np[:, :10, :] * m_np[:, :10, :]).sum(axis=1, keepdims=True) / np.maximum(m_np[:, :10, :].sum(axis=1, keepdims=True), 1)
    pers = np.repeat(pre_mean, y_np.shape[1], axis=1)
    horizon_res["persistence_premean"] = {}
    for H in horizons:
        end_t = 10 + H
        m_h = compute_stratified_metrics(
            y_np[:, :end_t, :], pers[:, :end_t, :], m_np[:, :end_t, :],
            stim=st_np[:, :end_t, :], pre_vols=10, adjacency=adjacency,
            evoked_baseline="linear", evoked_window_vols=min(H, 16), np_adjacency=np_adjacency,
        )
        horizon_res["persistence_premean"][f"{H * 0.5:.1f}s"] = {
            k: m_h[k] for k in ("post_r2", "post_inlier_r2", "post_inlier_r", "evoked_hop1_r2") if k in m_h
        }

    # Observation mask ablation on pre_stim_mask (keep 100%, 75%, 50%, 25% of observed neurons per trial)
    mask_fracs = [1.0, 0.75, 0.50, 0.25]
    mask_res: Dict[str, dict] = {}
    rng = np.random.default_rng(42)
    base_pre_obs_neurons = m_np[:, :10, :].any(axis=1)  # (B, N)
    sub_masks = {}
    for frac in mask_fracs:
        if frac >= 0.999:
            sub_masks[frac] = mask[:, :10, :].clone()
        else:
            km = m_np[:, :10, :].copy()
            for b in range(km.shape[0]):
                obs_n = np.flatnonzero(base_pre_obs_neurons[b])
                n_keep = max(1, int(round(frac * obs_n.size)))
                keep = set(rng.choice(obs_n, size=n_keep, replace=False).tolist())
                drop = [c for c in obs_n if c not in keep]
                if drop:
                    km[b, :, drop] = False
            sub_masks[frac] = torch.from_numpy(km).to(device)

    for label, ckpt, use_pre, use_ex in checkpoints[1:]:  # S2 models with pre_stim_encoder
        model = _load_ckpt(ckpt, use_prestim=use_pre, use_extrasyn=use_ex, device=device)
        mask_res[label] = {}
        for frac in mask_fracs:
            pm = sub_masks[frac]
            po = y[:, :10, :] * pm.to(dtype=y.dtype)
            ps_m, pz_m = _posterior_predict_pair(model, stim, mask, noise, po, pm, num_samples=num_samples)
            m_f = compute_stratified_metrics(
                y_np, ps_m, m_np, stim=st_np, pre_vols=10, adjacency=adjacency,
                evoked_baseline="mean", y_pred_counterfactual=pz_m, np_adjacency=np_adjacency,
            )
            mask_res[label][f"{int(round(frac * 100))}%"] = {
                k: m_f[k] for k in (
                    "post_r2", "post_inlier_r2", "post_inlier_r",
                    "evoked_stim_r2", "evoked_hop1_r2", "kernel_hop1_r", "evoked_np_no_wired_path_r2",
                ) if k in m_f
            }

    return {"horizon_ablation": horizon_res, "obs_mask_ablation": mask_res}


def pre_window_ablation(fold_split: dict, data_path: str) -> Dict:
    """Load AtlasLoader(..., causal_preprocess=True) once and sweep pre_vols in [5, 10, 20, 40, 80]."""
    print("\nLoading AtlasLoader(..., causal_preprocess=True) once for pre-stim window ablation...", flush=True)
    t0 = time.time()
    loader = AtlasLoader(path=data_path, strain="wild-type", norm_mode="dff_safe", causal_preprocess=True)
    print(f"Loaded {len(loader.recordings)} causal recordings in {time.time() - t0:.1f}s", flush=True)

    # Identify train vs test rec_ids from Fold-0 split on standard dataset
    std_ds = loader.build_trial_dataset(pre_vols=10, post_vols=50)
    tr_recs = {int(std_ds.meta[i]["rec_id"]) for i in (fold_split["train"] + fold_split.get("val", []))}
    te_recs = {int(std_ds.meta[i]["rec_id"]) for i in fold_split["test"]}

    # Track common (rec_id, event_idx) present even at pre_vols=80
    ds80 = loader.build_trial_dataset(pre_vols=80, post_vols=50)
    keys80 = {(int(m["rec_id"]), int(m["event_idx"])) for m in ds80.meta}

    window_res: Dict[str, dict] = {}
    for pre_v in [5, 10, 20, 40, 80]:
        ds_w = ds80 if pre_v == 80 else loader.build_trial_dataset(pre_vols=pre_v, post_vols=50)
        y_all = ds_w.y.numpy()
        m_all = ds_w.obs_mask.numpy().astype(bool)
        tr_idx = [i for i, m in enumerate(ds_w.meta) if int(m["rec_id"]) in tr_recs]
        te_idx_all = [i for i, m in enumerate(ds_w.meta) if int(m["rec_id"]) in te_recs]
        te_idx_common = [i for i in te_idx_all if (int(ds_w.meta[i]["rec_id"]), int(ds_w.meta[i]["event_idx"])) in keys80]

        # Test causal linear AR/Ridge summary features extracted solely from [0:pre_v] -> post [pre_v:pre_v+50]
        # Per-neuron summary features: [last_val, mean_last5, mean_all, slope] predicting post steps 0..49
        def _extract_feats(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            X_list, Y_list = [], []
            for bi in idxs:
                yv = y_all[bi]
                mv = m_all[bi]
                obs_n = np.flatnonzero(mv[:pre_v].any(axis=0) & mv[pre_v:].any(axis=0))
                for c in obs_n:
                    pre_y = yv[:pre_v, c]
                    pre_m = mv[:pre_v, c]
                    valid_vals = pre_y[pre_m]
                    if valid_vals.size == 0:
                        continue
                    last_v = valid_vals[-1]
                    k5 = min(5, valid_vals.size)
                    mean5 = float(np.mean(valid_vals[-k5:]))
                    mean_all = float(np.mean(valid_vals))
                    t_idx = np.flatnonzero(pre_m).astype(np.float64)
                    if t_idx.size >= 3 and np.var(t_idx) > 1e-8:
                        dt_s = t_idx - t_idx.mean()
                        slope = float(np.sum(dt_s * (valid_vals - mean_all)) / np.sum(dt_s ** 2))
                    else:
                        slope = 0.0
                    X_list.append([last_v, mean5, mean_all, slope, 1.0])
                    Y_list.append(np.where(mv[pre_v:, c], yv[pre_v:, c], np.nan))
            return np.asarray(X_list, dtype=np.float64), np.asarray(Y_list, dtype=np.float64)

        X_tr, Y_tr = _extract_feats(tr_idx)
        X_te_all, Y_te_all = _extract_feats(te_idx_all)
        X_te_com, Y_te_com = _extract_feats(te_idx_common)

        # Fit ridge per future horizon step t_h in 0..49
        pred_all = np.zeros_like(Y_te_all)
        pred_com = np.zeros_like(Y_te_com)
        lam = 100.0
        eye = np.eye(X_tr.shape[1]) * lam
        eye[-1, -1] = 1e-6
        for h in range(50):
            ok_tr = np.isfinite(Y_tr[:, h]) & (np.abs(Y_tr[:, h]) <= 3.0)
            Xh = X_tr[ok_tr]
            yh = Y_tr[ok_tr, h]
            w = np.linalg.solve(Xh.T @ Xh + eye, Xh.T @ yh)
            pred_all[:, h] = X_te_all @ w
            pred_com[:, h] = X_te_com @ w

        def _score(Y_t: np.ndarray, Y_p: np.ndarray) -> dict:
            ok = np.isfinite(Y_t)
            oki = ok & (np.abs(Y_t) <= 3.0)
            return {
                "post_r2": float(r2_score(Y_t[ok], Y_p[ok])),
                "post_inlier_r2": float(r2_score(Y_t[oki], Y_p[oki])),
                "post_inlier_r": _safe_r(Y_t[oki], Y_p[oki]),
                "post_1s_inlier_r2": float(r2_score(Y_t[:, :2][oki[:, :2]], Y_p[:, :2][oki[:, :2]])),
            }

        window_res[f"{pre_v * 0.5:.1f}s"] = {
            "pre_vols": pre_v,
            "total_trials": len(ds_w),
            "test_trials_all": len(te_idx_all),
            "test_trials_common80": len(te_idx_common),
            "ridge_ar_all_test": _score(Y_te_all, pred_all),
            "ridge_ar_common80_test": _score(Y_te_com, pred_com),
        }
        print(
            f"  pre={pre_v * 0.5:4.1f}s ({len(ds_w)} trials, common_test={len(te_idx_common)}): "
            f"common80 post_inlier_r2={window_res[f'{pre_v * 0.5:.1f}s']['ridge_ar_common80_test']['post_inlier_r2']:.4f} "
            f"1s_r2={window_res[f'{pre_v * 0.5:.1f}s']['ridge_ar_common80_test']['post_1s_inlier_r2']:.4f}",
            flush=True,
        )
    return window_res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/randi_causal.npz")
    p.add_argument("--split-file", default=f"{RUNS_BASE}/p15_5f_s1/fold_0.json")
    p.add_argument("--data-path", default=resolve_default_data_path())
    p.add_argument("--out", default=os.path.join(REPO, "report/data/phase4_noise_ceiling_and_ablations.json"))
    p.add_argument("--num-samples", default=8, type=int)
    args = p.parse_args()

    device = torch.device("cpu")
    dataset = RandiTrialDataset.load(args.cache)
    with open(args.split_file) as fh:
        fold_split = json.load(fh)

    intact = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adjacency = intact.synapse.adjacency_matrix(intact.neuronSize)
    np_adjacency = intact.synapse.extrasynaptic_adjacency_matrix(intact.neuronSize)

    print("=== Computing within/across-recording split-half noise ceilings ===", flush=True)
    nc = compute_noise_ceilings(dataset, fold_split, adjacency, np_adjacency, pre_vols=10)
    print(json.dumps(nc, indent=2), flush=True)

    print("\n=== Computing forecast horizon & observation-mask ablations ===", flush=True)
    hm = horizon_and_mask_ablations(dataset, fold_split, adjacency, np_adjacency, device, num_samples=args.num_samples)

    print("\n=== Computing causal pre-stim window ablation ===", flush=True)
    pw = pre_window_ablation(fold_split, args.data_path)

    out = {
        "cache": args.cache,
        "split_file": args.split_file,
        "noise_ceiling": nc,
        "forecast_horizon_ablation": hm["horizon_ablation"],
        "obs_mask_ablation": hm["obs_mask_ablation"],
        "pre_window_ablation": pw,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
