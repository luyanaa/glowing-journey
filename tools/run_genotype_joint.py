#!/usr/bin/env python3
"""Section 6.2: Genotype-conditioned joint Bayesian inference on WT + unc-31.

Jointly fits shared biophysical connectome parameters together with genotype-specific
neuropeptide (gamma_np_mut), monoamine (gamma_ma_mut), and compensatory wired
(gamma_wired_mut) scaling factors on unc-31 relative to WT (= 1.0 reference),
then reports posterior probabilities (P(gamma_np < 1), P(gamma_np < gamma_ma)),
held-out WT & unc-31 metrics, and matched pair Delta K_ij(t) = K_WT(t) - K_unc31(t).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.poutine.messenger import Messenger
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_c302 import (
    _save_snapshot,
    compute_stratified_metrics,
    extract_and_expand_posterior_samples,
)
from pyro_svi_vae import (
    _load_compatible_state_dict,
    _set_extrasyn_only_trainable,
    reweight_hop1_noise,
)
from tools.kernel_metrics import compute_matched_kernel_difference
from utils import RandiTrialDataset


class _CohortObsScaleMessenger(Messenger):
    """Exact per-trial ELBO likelihood reweighting on observed ``obs_*`` sites only.

    Preserves global prior KL terms untouched while replacing uniform ``batch_plate``
    subsampling factor with exact ``importance_weighted`` or ``balanced_50_50``
    weights across unequal-size WT and mutant cohorts.
    """

    def __init__(self, obs_trial_scale: torch.Tensor):
        super().__init__()
        self.obs_trial_scale = obs_trial_scale

    def _pyro_post_sample(self, msg):
        if msg.get("is_observed", False) and str(msg.get("name", "")).startswith("obs_"):
            msg["scale"] = self.obs_trial_scale.to(device=msg["value"].device, dtype=msg["value"].dtype)


def split_unc31_cv_folds(dataset: RandiTrialDataset, n_folds: int = 3) -> List[Tuple[List[int], List[int]]]:
    """Build recording-grouped CV folds (train_idx, test_idx) for unc-31 recordings."""
    groups = np.asarray([int(m["rec_id"]) for m in dataset.meta], dtype=np.int64)
    idx = np.arange(len(dataset))
    unique_recs = np.unique(groups)
    k = max(2, min(int(n_folds), int(unique_recs.size)))
    if unique_recs.size < 2:
        tr, te = split_unc31_by_recording(dataset, test_frac=0.3, seed=42)
        return [(tr, te)]
    gkf = GroupKFold(n_splits=k)
    return [(idx[tr].tolist(), idx[te].tolist()) for tr, te in gkf.split(idx, groups=groups)]


def compute_elbo_obs_weights(
    n_wt_train: int,
    n_u31_train: int,
    b_wt: int,
    b_u31: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-trial observation likelihood scale vector for concatenated [WT, u31] batch."""
    total = float(max(1, n_wt_train + n_u31_train))
    if mode == "importance_weighted":
        w_wt = float(n_wt_train) / float(max(1, b_wt))
        w_u31 = float(n_u31_train) / float(max(1, b_u31))
    elif mode == "balanced_50_50":
        w_wt = (0.5 * total) / float(max(1, b_wt))
        w_u31 = (0.5 * total) / float(max(1, b_u31))
    else:
        raise ValueError(f"Unsupported elbo_balance_mode={mode!r}")
    return torch.cat([
        torch.full((b_wt,), w_wt, dtype=torch.float32, device=device),
        torch.full((b_u31,), w_u31, dtype=torch.float32, device=device),
    ], dim=0)


def compute_joint_identifiability_diagnostics(
    model: vae.VAElegans,
    wt_loader: DataLoader,
    u31_loader: DataLoader,
    pre_vols: int,
    device: torch.device,
) -> Dict[str, object]:
    """Compute Fisher-information cosine correlation matrix across mutant genotype & calib parameters."""
    g = model.guide
    specs = []
    for attr in (
        "gamma_np_mut_loc",
        "gamma_ma_mut_loc",
        "gamma_wired_mut_loc",
        "gcamp_gain_mut_loc",
        "gcamp_offset_mut_loc",
        "stim_gain_mut_loc",
    ):
        if hasattr(g, attr):
            specs.append((attr.replace("_loc", ""), getattr(g, attr)))
    if len(specs) < 2:
        return {"param_names": [s[0] for s in specs], "fisher_corr": []}

    names = [s[0] for s in specs]
    params = [s[1] for s in specs]
    bw = next(iter(wt_loader))
    bu = next(iter(u31_loader))
    stim = torch.cat([bw[0][:8], bu[0][:8]], dim=0).to(device)
    y = torch.cat([bw[1][:8], bu[1][:8]], dim=0).to(device)
    mask = torch.cat([bw[2][:8], bu[2][:8]], dim=0).to(device)
    noise = torch.cat([bw[3][:8], bu[3][:8]], dim=0).to(device)
    geno = torch.cat([
        torch.zeros(min(8, bw[0].shape[0]), dtype=torch.long),
        torch.ones(min(8, bu[0].shape[0]), dtype=torch.long),
    ], dim=0).to(device)

    per_trial_grads = []
    for i in range(stim.shape[0]):
        model.zero_grad()
        g_kw = dict(ExternalInput=stim[i : i + 1], mask=None, y=None, noise_scale=noise[i : i + 1], genotype_id=geno[i : i + 1])
        tr = pyro.poutine.trace(g).get_trace(**g_kw)
        ps = extract_and_expand_posterior_samples(tr, repeat_factor=1)
        y_pred = pyro.condition(model, data=ps)(
            ExternalInput=stim[i : i + 1], mask=None, y=None,
            noise_scale=noise[i : i + 1], genotype_id=geno[i : i + 1], sample_obs=False,
        )
        t0 = min(int(pre_vols), y.shape[1])
        m_i = mask[i : i + 1, t0:, :].to(dtype=y_pred.dtype)
        loss_i = (((y_pred[:, t0:, :] - y[i : i + 1, t0:, :]) ** 2) * m_i).sum() / torch.clamp(m_i.sum(), min=1.0)
        grads = torch.autograd.grad(loss_i, params, allow_unused=True, retain_graph=False)
        vec = torch.tensor([0.0 if gr is None else float(gr.detach().item()) for gr in grads], dtype=torch.float64)
        per_trial_grads.append(vec)

    G = torch.stack(per_trial_grads, dim=0)
    fisher = (G.T @ G).numpy()
    diag = np.sqrt(np.maximum(np.diag(fisher), 1e-18))
    corr = fisher / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return {
        "param_names": names,
        "fisher_diag": [float(x) for x in np.diag(fisher)],
        "fisher_corr": [[float(v) for v in row] for row in corr],
    }


def split_unc31_by_recording(
    dataset: RandiTrialDataset, test_frac: float = 0.3, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Deterministically split unc-31 trials by recording ID."""
    groups = np.asarray([int(m["rec_id"]) for m in dataset.meta], dtype=np.int64)
    idx = np.arange(len(dataset))
    unique_recs = np.unique(groups)
    if unique_recs.size < 2:
        cut = max(1, int(round(len(idx) * (1.0 - test_frac))))
        return idx[:cut].tolist(), idx[cut:].tolist()
    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_frac), random_state=int(seed))
    tr, te = next(gss.split(idx, groups=groups))
    return idx[tr].tolist(), idx[te].tolist()


def evaluate_genotype_cohort(
    model: vae.VAElegans,
    loader: DataLoader,
    genotype_id_val: int,
    adjacency: np.ndarray,
    np_adj: np.ndarray,
    ma_adj: np.ndarray,
    pre_vols: int,
    num_samples: int = 16,
    eval_seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate held-out WT (genotype_id=0) or unc-31 (genotype_id=1) cohort."""
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    pyro.set_rng_seed(eval_seed)
    y_trues, y_preds, y_cfs, masks, stims = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            stim, y, mask, noise = [b.to(device) for b in batch[:4]]
            B = stim.shape[0]
            geno = torch.full((B,), int(genotype_id_val), dtype=torch.long, device=device)
            guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise, genotype_id=geno)
            stacked_kw = dict(
                ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
                mask=None,
                y=None,
                noise_scale=torch.cat([noise, noise], dim=0),
                genotype_id=torch.cat([geno, geno], dim=0),
                sample_obs=False,
            )
            ps_acc, pz_acc = None, None
            for _ in range(max(1, int(num_samples))):
                tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
                ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)
                out = pyro.condition(model, data=ps)(**stacked_kw)
                s_out, z_out = out[:B], out[B:]
                ps_acc = s_out if ps_acc is None else ps_acc + s_out
                pz_acc = z_out if pz_acc is None else pz_acc + z_out
            y_trues.append(y.cpu().numpy())
            y_preds.append((ps_acc / max(1, int(num_samples))).cpu().numpy())
            y_cfs.append((pz_acc / max(1, int(num_samples))).cpu().numpy())
            masks.append(mask.bool().cpu().numpy())
            stims.append(stim.cpu().numpy())

    yt = np.concatenate(y_trues, axis=0)
    yp = np.concatenate(y_preds, axis=0)
    ypc = np.concatenate(y_cfs, axis=0)
    mk = np.concatenate(masks, axis=0)
    st = np.concatenate(stims, axis=0)
    dt_obs = float(getattr(model.decoder, "dt_obs", 0.5))
    metrics = compute_stratified_metrics(
        yt, yp, mk, stim=st, pre_vols=pre_vols, adjacency=adjacency,
        evoked_baseline="mean", y_pred_counterfactual=ypc,
        np_adjacency=np_adj, ma_adjacency=ma_adj, dt_obs=dt_obs,
    )
    return metrics, (yt, yp - ypc, mk, st)


def summarize_genotype_multipliers(model: vae.VAElegans, n_draws: int = 4000, seed: int = 42) -> Dict[str, object]:
    """Summarize variational posteriors for mutant NP/MA/wired/calibration parameters."""
    g = model.guide
    gen = torch.Generator(device=g.gamma_np_mut_loc.device).manual_seed(int(seed))
    with torch.no_grad():
        np_draws = torch.exp(
            g.gamma_np_mut_loc + g.gamma_np_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_np_mut_loc.device)
        ).cpu().numpy()
        ma_draws = torch.exp(
            g.gamma_ma_mut_loc + g.gamma_ma_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_ma_mut_loc.device)
        ).cpu().numpy()
        if hasattr(g, "gamma_wired_mut_loc"):
            wd_draws = torch.exp(
                g.gamma_wired_mut_loc + g.gamma_wired_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_wired_mut_loc.device)
            ).cpu().numpy()
        else:
            wd_draws = np.ones((n_draws,), dtype=np.float32)
    out: Dict[str, object] = {
        "gamma_np_mut_median": float(np.median(np_draws)),
        "gamma_np_mut_ci90": [float(np.quantile(np_draws, 0.05)), float(np.quantile(np_draws, 0.95))],
        "gamma_ma_mut_median": float(np.median(ma_draws)),
        "gamma_ma_mut_ci90": [float(np.quantile(ma_draws, 0.05)), float(np.quantile(ma_draws, 0.95))],
        "gamma_wired_mut_median": float(np.median(wd_draws)),
        "gamma_wired_mut_ci90": [float(np.quantile(wd_draws, 0.05)), float(np.quantile(wd_draws, 0.95))],
        "prob_gamma_np_lt_1": float(np.mean(np_draws < 1.0)),
        "prob_gamma_ma_lt_1": float(np.mean(ma_draws < 1.0)),
        "prob_gamma_np_lt_gamma_ma": float(np.mean(np_draws < ma_draws)),
    }
    if hasattr(g, "tau_np_loc"):
        out["tau_np_median_s"] = float(torch.exp(g.tau_np_loc.detach()).cpu().item())
    if hasattr(g, "tau_ma_loc"):
        out["tau_ma_median_s"] = float(torch.exp(g.tau_ma_loc.detach()).cpu().item())
    if hasattr(g, "gcamp_gain_mut_loc"):
        out["gcamp_gain_mut_median"] = float(torch.exp(g.gcamp_gain_mut_loc.detach()).cpu().item())
    if hasattr(g, "gcamp_offset_mut_loc"):
        out["gcamp_offset_mut_median"] = float(g.gcamp_offset_mut_loc.detach().cpu().item())
    if hasattr(g, "stim_gain_mut_loc"):
        out["stim_gain_mut_median"] = float(torch.exp(g.stim_gain_mut_loc.detach()).cpu().item())
    return out


def _set_joint_extrasyn_trainable(
    model: vae.VAElegans,
    freeze_mod_timescales: bool = False,
    strain_obs_calib: bool = False,
    train_obs_sigma: bool = True,
) -> None:
    """Freeze wired connectome weights while optionally keeping strain calib / freezing tau_np/tau_ma."""
    for name, param in model.named_parameters():
        keep = (
            ("extrasyn" in name)
            or ("gamma_np_mut" in name)
            or ("gamma_ma_mut" in name)
            or ("gamma_wired_mut" in name)
            or ((not freeze_mod_timescales) and (("tau_np" in name) or ("tau_ma" in name)))
            or (strain_obs_calib and (
                ("gcamp_gain_mut" in name) or ("gcamp_offset_mut" in name) or ("stim_gain_mut" in name)
            ))
            or (train_obs_sigma and ("obs_sigma" in name))
        )
        param.requires_grad_(keep)


def _run_single_joint_split(
    args: argparse.Namespace,
    ds_wt: RandiTrialDataset,
    ds_u31: RandiTrialDataset,
    wt_train_idx: List[int],
    wt_test_idx: List[int],
    u31_train_idx: List[int],
    u31_test_idx: List[int],
    device: torch.device,
    fold_tag: str = "fold0",
) -> Dict[str, object]:
    """Run one hardened WT + unc-31 joint fit and return held-out & identifiability metrics."""
    pyro.clear_param_store()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=True,
        per_neuron_stim_gain=True,
        wide_amp_priors=True,
        evoked_syn_priors=True,
        dimensionless=True,
        leak_gauge=("zero_mean_log" if args.heterogeneous_leak_gauge else "unit"),
        split_extrasynaptic=True,
        use_syn_filter=args.syn_filter,
        tau_np_prior=args.tau_np_prior,
        tau_ma_prior=args.tau_ma_prior,
        tau_mod_prior_std=args.tau_mod_prior_std,
        genotype_conditioned=True,
        fix_wired_mut=bool(args.fix_wired_mut),
    )
    adj = step.synapse.adjacency_matrix(step.neuronSize)
    np_adj = step.synapse.np_adjacency_matrix(step.neuronSize)
    ma_adj = step.synapse.ma_adjacency_matrix(step.neuronSize)
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=device)

    snap = torch.load(args.init_from, map_location=device, weights_only=False)
    ms_keys = set(snap["MODEL_STATE"].keys())
    per_neuron_kinetics = any(
        "tau_ca_vec" in k
        or "k_off_vec" in k
        or (k == "guide.tau_ca_loc_unconstrained" and snap["MODEL_STATE"][k].numel() > 1)
        for k in ms_keys
    )

    rec = c302.RecurrentNematode(
        step, dt_obs=0.5, substeps=args.substeps, use_prestim_encoder=False,
        pre_vols=args.pre_vols, per_neuron_kinetics=per_neuron_kinetics, stim_actuator=True,
        stim_tau_rise=0.08, stim_tau_decay=0.4, strain_obs_calib=bool(args.strain_obs_calib),
    )
    model = vae.VAElegans(
        rec, step.neuronSize, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    ).to(device)
    skipped, missing, _ = _load_compatible_state_dict(model, snap["MODEL_STATE"])
    print(f"[{fold_tag}] Warm-start skipped={len(skipped)} missing={len(missing)}", flush=True)
    if args.freeze_wired_connectome:
        _set_joint_extrasyn_trainable(
            model,
            freeze_mod_timescales=bool(args.freeze_mod_timescales),
            strain_obs_calib=bool(args.strain_obs_calib),
        )

    half_b = max(4, int(args.batch_size) // 2)
    wt_tr_ldr = DataLoader(Subset(ds_wt, wt_train_idx), batch_size=half_b, shuffle=True)
    u31_tr_ldr = DataLoader(Subset(ds_u31, u31_train_idx), batch_size=half_b, shuffle=True)
    total_dataset_size = len(wt_train_idx) + len(u31_train_idx)

    optim = pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0})
    for ep in range(int(args.epochs)):
        ep_t0 = time.time()
        loss_sum = 0.0
        n_steps = max(len(wt_tr_ldr), len(u31_tr_ldr))
        wt_it = iter(wt_tr_ldr)
        u31_it = iter(u31_tr_ldr)
        for _ in range(n_steps):
            try:
                bw = next(wt_it)
            except StopIteration:
                wt_it = iter(wt_tr_ldr)
                bw = next(wt_it)
            try:
                bu = next(u31_it)
            except StopIteration:
                u31_it = iter(u31_tr_ldr)
                bu = next(u31_it)
            b_w = int(bw[0].shape[0])
            b_u = int(bu[0].shape[0])
            stim = torch.cat([bw[0], bu[0]], dim=0).to(device)
            y = torch.cat([bw[1], bu[1]], dim=0).to(device)
            mask = torch.cat([bw[2], bu[2]], dim=0).to(device)
            noise = torch.cat([bw[3], bu[3]], dim=0).to(device)
            geno = torch.cat([
                torch.zeros(b_w, dtype=torch.long),
                torch.ones(b_u, dtype=torch.long),
            ], dim=0).to(device)
            noise = reweight_hop1_noise(noise, stim, adj_t, args.hop1_loss_weight, args.stim_loss_weight)
            obs_w = compute_elbo_obs_weights(
                len(wt_train_idx), len(u31_train_idx), b_w, b_u, args.elbo_balance_mode, device
            )
            with _CohortObsScaleMessenger(obs_w):
                svi = SVI(model, model.guide, optim, Trace_ELBO())
                loss_sum += svi.step(
                    ExternalInput=stim, mask=mask, y=y, noise_scale=noise,
                    dataset_size=total_dataset_size, genotype_id=geno,
                )
        print(f"[{fold_tag}] Epoch {ep:03d} | total_elbo={loss_sum / max(1, n_steps):.2f} | dt={time.time() - ep_t0:.2f}s", flush=True)

    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        _save_snapshot(os.path.join(args.snapshot_dir, f"genotype_joint_{fold_tag}_ep{args.epochs}.pt"), model, model.guide, args.epochs)

    model.eval()
    wt_te_ldr = DataLoader(Subset(ds_wt, wt_test_idx), batch_size=max(32, int(args.batch_size)), shuffle=False)
    u31_te_ldr = DataLoader(Subset(ds_u31, u31_test_idx), batch_size=max(32, int(args.batch_size)), shuffle=False)
    wt_metrics, wt_arrs = evaluate_genotype_cohort(
        model, wt_te_ldr, 0, adj, np_adj, ma_adj, args.pre_vols, args.num_samples, args.seed, device
    )
    u31_metrics, u31_arrs = evaluate_genotype_cohort(
        model, u31_te_ldr, 1, adj, np_adj, ma_adj, args.pre_vols, args.num_samples, args.seed, device
    )
    matched_dk = compute_matched_kernel_difference(
        wt_arrs[0], wt_arrs[1], wt_arrs[2], wt_arrs[3],
        u31_arrs[0], u31_arrs[1], u31_arrs[2], u31_arrs[3],
        adjacency=adj, np_adjacency=np_adj, ma_adjacency=ma_adj, pre_vols=args.pre_vols,
    )
    ident = compute_joint_identifiability_diagnostics(model, wt_te_ldr, u31_te_ldr, args.pre_vols, device)
    return {
        "fold": fold_tag,
        "wt_train_trials": len(wt_train_idx),
        "wt_test_trials": len(wt_test_idx),
        "unc31_train_trials": len(u31_train_idx),
        "unc31_test_trials": len(u31_test_idx),
        "unc31_train_recordings": len({int(ds_u31.meta[i]["rec_id"]) for i in u31_train_idx}),
        "unc31_test_recordings": len({int(ds_u31.meta[i]["rec_id"]) for i in u31_test_idx}),
        "posterior_multipliers": summarize_genotype_multipliers(model, seed=args.seed),
        "identifiability": ident,
        "wt_test_metrics": wt_metrics,
        "unc31_test_metrics": u31_metrics,
        "matched_delta_kernel": matched_dk,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Joint WT + unc-31 genotype-conditioned Bayesian inference")
    p.add_argument("--wt-cache", required=True)
    p.add_argument("--wt-split-file", required=True)
    p.add_argument("--unc31-cache", required=True)
    p.add_argument("--init-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--substeps", default=4, type=int)
    p.add_argument("--pre-vols", default=40, type=int)
    p.add_argument("--lr", default=0.004, type=float)
    p.add_argument("--stim-loss-weight", default=2.5, type=float)
    p.add_argument("--hop1-loss-weight", default=1.7, type=float)
    p.add_argument("--heterogeneous-leak-gauge", action="store_true")
    p.add_argument("--syn-filter", action="store_true")
    p.add_argument("--tau-np-prior", default=4.0, type=float)
    p.add_argument("--tau-ma-prior", default=1.0, type=float)
    p.add_argument("--tau-mod-prior-std", default=0.5, type=float)
    p.add_argument("--freeze-wired-connectome", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fix-wired-mut", action="store_true",
                   help="Pin gamma_wired_mut = 1.0 (prevent compensatory wired conductance absorption)")
    p.add_argument("--strain-obs-calib", action="store_true",
                   help="Allow global strain-level gcamp_gain/offset and stim_gain calibration on unc-31")
    p.add_argument("--freeze-mod-timescales", action="store_true",
                   help="Freeze tau_np and tau_ma at WT values during joint fit")
    p.add_argument("--elbo-balance-mode", choices=["importance_weighted", "balanced_50_50"], default="importance_weighted")
    p.add_argument("--n-cv-folds", default=1, type=int,
                   help="Recording-grouped CV folds over unc-31 (>1 runs GroupKFold summary)")
    p.add_argument("--num-samples", default=16, type=int)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_wt = RandiTrialDataset.load(args.wt_cache)
    ds_u31 = RandiTrialDataset.load(args.unc31_cache)
    with open(args.wt_split_file) as fh:
        wt_split = json.load(fh)
    wt_train_idx = [int(i) for i in wt_split["train"]]
    wt_test_idx = [int(i) for i in wt_split["test"]]

    n_cv = max(1, int(args.n_cv_folds))
    if n_cv > 1:
        u31_folds = split_unc31_cv_folds(ds_u31, n_folds=n_cv)
    else:
        u31_folds = [split_unc31_by_recording(ds_u31, test_frac=0.3, seed=args.seed)]

    fold_results = []
    for fi, (u31_tr, u31_te) in enumerate(u31_folds):
        res_f = _run_single_joint_split(
            args, ds_wt, ds_u31, wt_train_idx, wt_test_idx, u31_tr, u31_te, device, fold_tag=f"fold{fi}"
        )
        fold_results.append(res_f)

    if len(fold_results) == 1:
        out = dict(fold_results[0])
        out.update({
            "wt_cache": args.wt_cache,
            "unc31_cache": args.unc31_cache,
            "init_from": args.init_from,
            "fix_wired_mut": bool(args.fix_wired_mut),
            "strain_obs_calib": bool(args.strain_obs_calib),
            "freeze_mod_timescales": bool(args.freeze_mod_timescales),
            "elbo_balance_mode": args.elbo_balance_mode,
            "wall_s": float(time.time() - t0),
        })
    else:
        np_meds = [f["posterior_multipliers"]["gamma_np_mut_median"] for f in fold_results]
        ma_meds = [f["posterior_multipliers"]["gamma_ma_mut_median"] for f in fold_results]
        p_lt = [f["posterior_multipliers"]["prob_gamma_np_lt_gamma_ma"] for f in fold_results]
        u31_h1 = [f["unc31_test_metrics"].get("evoked_hop1_r2", float("nan")) for f in fold_results]
        out = {
            "wt_cache": args.wt_cache,
            "unc31_cache": args.unc31_cache,
            "init_from": args.init_from,
            "n_cv_folds": len(fold_results),
            "fix_wired_mut": bool(args.fix_wired_mut),
            "strain_obs_calib": bool(args.strain_obs_calib),
            "freeze_mod_timescales": bool(args.freeze_mod_timescales),
            "elbo_balance_mode": args.elbo_balance_mode,
            "cv_summary": {
                "gamma_np_mut_mean": float(np.mean(np_meds)),
                "gamma_np_mut_std": float(np.std(np_meds, ddof=1)),
                "gamma_ma_mut_mean": float(np.mean(ma_meds)),
                "gamma_ma_mut_std": float(np.std(ma_meds, ddof=1)),
                "prob_gamma_np_lt_gamma_ma_mean": float(np.mean(p_lt)),
                "unc31_evoked_hop1_r2_mean": float(np.nanmean(u31_h1)),
                "unc31_evoked_hop1_r2_std": float(np.nanstd(u31_h1, ddof=1)),
            },
            "folds": fold_results,
            "wall_s": float(time.time() - t0),
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
