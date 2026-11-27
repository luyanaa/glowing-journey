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
from sklearn.model_selection import GroupShuffleSplit
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


def summarize_genotype_multipliers(model: vae.VAElegans, n_draws: int = 4000, seed: int = 42) -> Dict[str, float]:
    """Summarize variational posteriors for mutant NP/MA/wired multipliers."""
    g = model.guide
    gen = torch.Generator(device=g.gamma_np_mut_loc.device).manual_seed(int(seed))
    with torch.no_grad():
        np_draws = torch.exp(
            g.gamma_np_mut_loc + g.gamma_np_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_np_mut_loc.device)
        ).cpu().numpy()
        ma_draws = torch.exp(
            g.gamma_ma_mut_loc + g.gamma_ma_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_ma_mut_loc.device)
        ).cpu().numpy()
        wd_draws = torch.exp(
            g.gamma_wired_mut_loc + g.gamma_wired_mut_scale * torch.randn((n_draws,), generator=gen, device=g.gamma_wired_mut_loc.device)
        ).cpu().numpy()
    out = {
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
    return out


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
    p.add_argument("--num-samples", default=16, type=int)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    t0 = time.time()
    pyro.clear_param_store()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_wt = RandiTrialDataset.load(args.wt_cache)
    ds_u31 = RandiTrialDataset.load(args.unc31_cache)
    with open(args.wt_split_file) as fh:
        wt_split = json.load(fh)
    wt_train_idx = [int(i) for i in wt_split["train"]]
    wt_test_idx = [int(i) for i in wt_split["test"]]
    u31_train_idx, u31_test_idx = split_unc31_by_recording(ds_u31, test_frac=0.3, seed=args.seed)

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
        stim_tau_rise=0.08, stim_tau_decay=0.4,
    )
    model = vae.VAElegans(
        rec, step.neuronSize, len(step.synapse.Wicks_SRC), len(step.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    ).to(device)
    skipped, missing, unexpected = _load_compatible_state_dict(model, snap["MODEL_STATE"])
    print(f"Warm-start from {args.init_from}: skipped={len(skipped)} missing={len(missing)}", flush=True)
    if args.freeze_wired_connectome:
        _set_extrasyn_only_trainable(model, True)

    half_b = max(4, int(args.batch_size) // 2)
    wt_tr_ldr = DataLoader(Subset(ds_wt, wt_train_idx), batch_size=half_b, shuffle=True)
    u31_tr_ldr = DataLoader(Subset(ds_u31, u31_train_idx), batch_size=half_b, shuffle=True)
    total_dataset_size = len(wt_train_idx) + len(u31_train_idx)

    svi = SVI(model, model.guide, pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0}), Trace_ELBO())
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
            stim = torch.cat([bw[0], bu[0]], dim=0).to(device)
            y = torch.cat([bw[1], bu[1]], dim=0).to(device)
            mask = torch.cat([bw[2], bu[2]], dim=0).to(device)
            noise = torch.cat([bw[3], bu[3]], dim=0).to(device)
            geno = torch.cat([
                torch.zeros(bw[0].shape[0], dtype=torch.long),
                torch.ones(bu[0].shape[0], dtype=torch.long),
            ], dim=0).to(device)
            noise = reweight_hop1_noise(noise, stim, adj_t, args.hop1_loss_weight, args.stim_loss_weight)
            loss_sum += svi.step(
                ExternalInput=stim, mask=mask, y=y, noise_scale=noise,
                dataset_size=total_dataset_size, genotype_id=geno,
            )
        print(f"Epoch {ep:03d} | total_elbo={loss_sum / max(1, n_steps):.2f} | dt={time.time() - ep_t0:.2f}s", flush=True)

    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        _save_snapshot(os.path.join(args.snapshot_dir, f"genotype_joint_ep{args.epochs}.pt"), model, model.guide, args.epochs)

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
    out = {
        "wt_cache": args.wt_cache,
        "unc31_cache": args.unc31_cache,
        "init_from": args.init_from,
        "wt_train_trials": len(wt_train_idx),
        "wt_test_trials": len(wt_test_idx),
        "unc31_train_trials": len(u31_train_idx),
        "unc31_test_trials": len(u31_test_idx),
        "posterior_multipliers": summarize_genotype_multipliers(model, seed=args.seed),
        "wt_test_metrics": wt_metrics,
        "unc31_test_metrics": u31_metrics,
        "matched_delta_kernel": matched_dk,
        "wall_s": float(time.time() - t0),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
