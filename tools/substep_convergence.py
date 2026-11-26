#!/usr/bin/env python3
"""Step 4.0d: Substep convergence audit across substeps in [2, 4, 8, 16, 32].

Evaluates:
  1. Trajectory & evoked GCaMP peak/latency/AUC drift vs substeps=32 (32 test trials).
  2. Loss gradient cosine similarity w.r.t. biophysical global parameters vs substeps=32.
  3. Full Fold-0 test metrics (641 trials) across substeps in [2, 4, 8, 16].
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pyro
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import c302
import vae
from pyro_svi_vae import evaluate_predictions
from utils import RandiTrialDataset

RUNS_BASE = "/Users/yanlu/Documents/glowing-journey_runs/phase25_decoupled_extrasyn_fold0_2026-11-25"


def _build_model(checkpoint_path: str, substeps: int, use_prestim: bool, use_extrasyn: bool, device: torch.device):
    pyro.clear_param_store()
    snap = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ms_keys = set(snap["MODEL_STATE"].keys())
    per_neuron_stim_gain = any("stim_gain_vec" in k for k in ms_keys)
    per_neuron_kinetics = any("tau_ca_vec" in k or "k_off_vec" in k for k in ms_keys)
    per_edge_release = any("V_rest_edge" in k for k in ms_keys)
    step = c302.readConnectome(
        os.path.join(REPO, "data/CElegansNeuronTables.xls"),
        use_extrasynaptic=use_extrasyn,
        per_edge_release=per_edge_release,
        per_neuron_stim_gain=per_neuron_stim_gain,
        wide_amp_priors=True,
    )
    rec = c302.RecurrentNematode(
        step,
        dt_obs=0.5,
        substeps=int(substeps),
        use_prestim_encoder=bool(use_prestim),
        pre_vols=10,
        prestim_rank=16,
        variational_prestim=bool(use_prestim),
        prestim_oscillatory=False,
        per_neuron_kinetics=per_neuron_kinetics,
        per_edge_release=per_edge_release,
        stim_actuator=True,
        stim_tau_rise=0.08,
        stim_tau_decay=0.4,
    )
    model = vae.VAElegans(
        rec,
        rec.model.Neuron.neuronSize,
        len(rec.model.synapse.Wicks_SRC),
        len(rec.model.synapse.Gap_Junction_SRC),
        direct_global_guide=True,
    ).to(device)
    model.load_state_dict(snap["MODEL_STATE"])
    pyro.get_param_store().set_state(snap["PARAM_STORE"])
    return model


def _sample_fixed_latents(model, stim: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor, y: torch.Tensor, seed: int = 42) -> Dict[str, torch.Tensor]:
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
    kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_pre:
        kw["pre_stim_obs"] = y[:, :10, :]
        kw["pre_stim_mask"] = mask[:, :10, :]
    with torch.no_grad():
        tr = pyro.poutine.trace(model.guide).get_trace(**kw)
    return {
        name: site["value"].detach().clone()
        for name, site in tr.nodes.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm())
    nb = float(b.norm())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(torch.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))


def audit_checkpoint(
    name: str,
    checkpoint_path: str,
    dataset: RandiTrialDataset,
    test_idx: List[int],
    probe_idx: List[int],
    adjacency: np.ndarray,
    np_adjacency: np.ndarray,
    use_prestim: bool,
    use_extrasyn: bool,
    device: torch.device,
    substeps_traj: List[int],
    substeps_eval: List[int],
    num_eval_samples: int,
) -> Dict:
    print(f"\n=== Auditing {name} ({checkpoint_path}) ===", flush=True)
    probe_ds = Subset(dataset, probe_idx)
    stim_p = torch.stack([probe_ds[i][0] for i in range(len(probe_ds))], dim=0).to(device)
    y_p = torch.stack([probe_ds[i][1] for i in range(len(probe_ds))], dim=0).to(device)
    m_p = torch.stack([probe_ds[i][2] for i in range(len(probe_ds))], dim=0).to(device)
    noise_p = torch.stack([probe_ds[i][3] for i in range(len(probe_ds))], dim=0).to(device)

    # Fix one posterior draw from guide so differences reflect pure integrator resolution.
    m0 = _build_model(checkpoint_path, substeps=substeps_traj[0], use_prestim=use_prestim, use_extrasyn=use_extrasyn, device=device)
    ps_fixed = _sample_fixed_latents(m0, stim_p, m_p, noise_p, y_p, seed=42)

    adj_t = torch.from_numpy(adjacency).to(device=device, dtype=torch.float32)
    stim_target_p = stim_p.abs().sum(dim=1) > 0  # (B, N)
    hop1_p = (stim_target_p.to(dtype=torch.float32) @ adj_t > 0) & (~stim_target_p)

    traj_by_s: Dict[int, np.ndarray] = {}
    evoked_by_s: Dict[int, np.ndarray] = {}
    grad_vecs: Dict[int, Dict[str, torch.Tensor]] = {}
    traj_time_s: Dict[int, float] = {}

    for s in substeps_traj:
        t_start = time.time()
        model = _build_model(checkpoint_path, substeps=s, use_prestim=use_prestim, use_extrasyn=use_extrasyn, device=device)
        model.eval()
        has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
        kw = dict(ExternalInput=stim_p, mask=None, y=None, noise_scale=noise_p, sample_obs=False)
        kw0 = dict(ExternalInput=torch.zeros_like(stim_p), mask=None, y=None, noise_scale=noise_p, sample_obs=False)
        if has_pre:
            kw["pre_stim_obs"] = y_p[:, :10, :]
            kw["pre_stim_mask"] = m_p[:, :10, :]
            kw0["pre_stim_obs"] = y_p[:, :10, :]
            kw0["pre_stim_mask"] = m_p[:, :10, :]

        # Forward trajectory sensitivity (no grad)
        with torch.no_grad():
            cond = pyro.condition(model, data=ps_fixed)
            pred = cond(**kw)
            pred0 = cond(**kw0)
        traj_by_s[s] = pred.cpu().numpy()
        evoked_by_s[s] = (pred - pred0).cpu().numpy()

        # Exact fixed-seed SVI ELBO gradient w.r.t. biophysical global guide params
        model.train()
        target_names = (
            "guide.wicks_g_max_z", "guide.wicks_g_max_global_z",
            "guide.gap_g_syn_z", "guide.gap_g_syn_global_z",
            "guide.stim_gain_loc", "guide.stim_gain_vec_loc",
            "guide.extrasyn_gain_z",
        )
        for pname, p in pyro.get_param_store().named_parameters():
            if pname in target_names:
                p.requires_grad_(True)
            if p.grad is not None:
                p.grad.zero_()
        pyro.set_rng_seed(42)
        torch.manual_seed(42)
        kw_elbo = dict(ExternalInput=stim_p, mask=m_p, y=y_p, noise_scale=noise_p)
        if has_pre:
            kw_elbo["pre_stim_obs"] = y_p[:, :10, :]
            kw_elbo["pre_stim_mask"] = m_p[:, :10, :]
        pyro.infer.Trace_ELBO().loss_and_grads(model, model.guide, **kw_elbo)
        grad_vecs[s] = {
            pname: param.grad.detach().clone().cpu()
            for pname, param in pyro.get_param_store().named_parameters()
            if pname in target_names and param.grad is not None
        }
        traj_time_s[s] = float(time.time() - t_start)
        print(f"  probe substeps={s}: elapsed={traj_time_s[s]:.2f}s", flush=True)

    ref_s = max(substeps_traj)
    ref_traj = traj_by_s[ref_s]
    ref_ev = evoked_by_s[ref_s]
    ref_ev_hop1 = ref_ev[:, 10:26, :][hop1_p.cpu().numpy()[:, None, :].repeat(16, axis=1)]
    ref_ev_stim = ref_ev[:, 10:26, :][stim_target_p.cpu().numpy()[:, None, :].repeat(16, axis=1)]

    # Reference mean 1-hop evoked curve across valid hop1 pairs for latency/AUC check
    m_hop1_3d = hop1_p.cpu().numpy()[:, None, :] & m_p[:, 10:26, :].cpu().numpy()
    stim_3d = stim_target_p.cpu().numpy()[:, None, :] & m_p[:, 10:26, :].cpu().numpy()

    probe_comparison: Dict[str, dict] = {}
    for s in substeps_traj:
        tr = traj_by_s[s]
        ev = evoked_by_s[s]
        diff = tr - ref_traj
        rmse_vs_ref = float(np.sqrt(np.mean(diff ** 2)))
        max_abs_vs_ref = float(np.max(np.abs(diff)))

        ev_diff_hop1 = ev[:, 10:26, :][m_hop1_3d] - ref_ev[:, 10:26, :][m_hop1_3d]
        ev_hop1_rmse = float(np.sqrt(np.mean(ev_diff_hop1 ** 2))) if ev_diff_hop1.size else 0.0
        ev_hop1_rel = float(ev_hop1_rmse / max(float(np.std(ref_ev[:, 10:26, :][m_hop1_3d])), 1e-12))

        # Per-trial-neuron peak amp and latency on strong ref evoked responses (|peak| > 0.01)
        ev_h = ev[:, 10:26, :].transpose(0, 2, 1)[hop1_p.cpu().numpy()]  # (K, 16)
        rf_h = ref_ev[:, 10:26, :].transpose(0, 2, 1)[hop1_p.cpu().numpy()]
        pk_rf = np.max(np.abs(rf_h), axis=1)
        strong = pk_rf > 0.01
        if np.any(strong):
            lat_s = np.argmax(np.abs(ev_h[strong]), axis=1) * 0.5
            lat_r = np.argmax(np.abs(rf_h[strong]), axis=1) * 0.5
            lat_mae_s = float(np.mean(np.abs(lat_s - lat_r)))
            pk_ev = np.max(np.abs(ev_h[strong]), axis=1)
            pk_rel_err = float(np.mean(np.abs(pk_ev - pk_rf[strong]) / pk_rf[strong]))
            auc_ev = np.sum(ev_h[strong], axis=1) * 0.5
            auc_rf = np.sum(rf_h[strong], axis=1) * 0.5
            auc_rel_err = float(np.mean(np.abs(auc_ev - auc_rf)) / max(float(np.mean(np.abs(auc_rf))), 1e-12))
        else:
            lat_mae_s, pk_rel_err, auc_rel_err = 0.0, 0.0, 0.0

        # Direct stim peak relative error
        ev_st = ev[:, 10:26, :].transpose(0, 2, 1)[stim_target_p.cpu().numpy()]
        rf_st = ref_ev[:, 10:26, :].transpose(0, 2, 1)[stim_target_p.cpu().numpy()]
        stim_pk_rel = float(np.mean(np.abs(np.max(np.abs(ev_st), axis=1) - np.max(np.abs(rf_st), axis=1))) / max(float(np.mean(np.max(np.abs(rf_st), axis=1))), 1e-12))

        g_cos = {}
        for k in grad_vecs[ref_s]:
            if k in grad_vecs[s]:
                g_cos[k] = _cosine(grad_vecs[s][k], grad_vecs[ref_s][k])
        all_s = torch.cat([grad_vecs[s][k].reshape(-1) for k in sorted(grad_vecs[ref_s]) if k in grad_vecs[s]])
        all_r = torch.cat([grad_vecs[ref_s][k].reshape(-1) for k in sorted(grad_vecs[ref_s]) if k in grad_vecs[s]])
        g_cos["overall"] = _cosine(all_s, all_r)

        probe_comparison[str(s)] = {
            "traj_rmse_vs_ref": rmse_vs_ref,
            "traj_max_abs_vs_ref": max_abs_vs_ref,
            "evoked_hop1_rmse_vs_ref": ev_hop1_rmse,
            "evoked_hop1_rel_rmse_vs_ref": ev_hop1_rel,
            "stim_peak_rel_err_vs_ref": stim_pk_rel,
            "hop1_peak_rel_err_vs_ref": pk_rel_err,
            "hop1_latency_mae_s_vs_ref": lat_mae_s,
            "hop1_auc_rel_err_vs_ref": auc_rel_err,
            "grad_cosine_vs_ref": g_cos,
            "probe_wall_s": traj_time_s[s],
        }

    # Full Fold-0 test set evaluation
    loader = DataLoader(Subset(dataset, test_idx), batch_size=len(test_idx), shuffle=False)
    eval_by_substep: Dict[str, dict] = {}
    for s in substeps_eval:
        t0 = time.time()
        model = _build_model(checkpoint_path, substeps=s, use_prestim=use_prestim, use_extrasyn=use_extrasyn, device=device)
        model.eval()
        np.random.seed(42)
        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        m = evaluate_predictions(
            model,
            loader,
            device,
            guide=model.guide,
            num_samples=num_eval_samples,
            pre_vols=10,
            adjacency=adjacency,
            np_adjacency=np_adjacency,
        )
        elapsed = float(time.time() - t0)
        keep_keys = [
            "r2", "post_inlier_r2", "post_inlier_r",
            "evoked_stim_r2", "evoked_stim_r",
            "evoked_hop1_r2", "evoked_hop1_r", "evoked_hop1_amp_ratio",
            "pair_evoked_hop1_r", "pair_evoked_hop1_r2", "pair_evoked_hop1_sign_acc",
            "kernel_hop1_n_pairs", "kernel_hop1_r", "kernel_hop1_r2", "kernel_hop1_pair_curve_r_mean",
            "kernel_hop1_auc_r", "kernel_hop1_peak_amp_r", "kernel_hop1_peak_latency_mae_s", "kernel_hop1_fft_r",
            "evoked_np_only_r2", "evoked_np_no_hop12_r2", "evoked_np_no_wired_path_r2",
        ]
        summary_m = {k: m[k] for k in keep_keys if k in m}
        summary_m["eval_wall_s"] = elapsed
        eval_by_substep[str(s)] = summary_m
        print(
            f"  full Fold-0 eval substeps={s}: evoked_hop1_r2={m.get('evoked_hop1_r2'):.5f} "
            f"evoked_stim_r2={m.get('evoked_stim_r2'):.5f} kernel_hop1_r={m.get('kernel_hop1_r', float('nan')):.4f} "
            f"post_inlier_r2={m.get('post_inlier_r2'):.5f} ({elapsed:.1f}s)",
            flush=True,
        )

    return {
        "checkpoint": checkpoint_path,
        "use_prestim": use_prestim,
        "use_extrasyn": use_extrasyn,
        "ref_substeps": ref_s,
        "probe_comparison": probe_comparison,
        "full_test_eval": eval_by_substep,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/randi_causal.npz")
    p.add_argument("--split-file", default=f"{RUNS_BASE}/p15_5f_s1/fold_0.json")
    p.add_argument("--out", default=os.path.join(REPO, "report/data/substep_convergence_audit.json"))
    p.add_argument("--num-samples", default=4, type=int)
    args = p.parse_args()

    device = torch.device("cpu")
    dataset = RandiTrialDataset.load(args.cache)
    with open(args.split_file) as fh:
        split = json.load(fh)
    test_idx = [int(i) for i in split["test"]]

    # Deterministic 32-trial probe subset covering diverse stimulated neurons
    probe_idx = test_idx[::max(1, len(test_idx) // 32)][:32]

    intact = c302.readConnectome(os.path.join(REPO, "data/CElegansNeuronTables.xls"))
    adjacency = intact.synapse.adjacency_matrix(intact.neuronSize)
    np_adjacency = intact.synapse.extrasynaptic_adjacency_matrix(intact.neuronSize)

    models = [
        ("phase15_f0_s1_45ep", f"{RUNS_BASE}/p15_5f_s1/fold0/intact_seed42/vae_epoch_45.pt", False, False),
        ("phase15_f0_s2", f"{RUNS_BASE}/p15_5f_s2/fold0/intact_seed42/vae_epoch_10.pt", True, False),
        ("phase25_f0_s1b", f"{RUNS_BASE}/p25_f0_s1b/fold0/intact_seed42/vae_epoch_10.pt", False, True),
    ]

    results = {
        "cache": args.cache,
        "split_file": args.split_file,
        "test_trials": len(test_idx),
        "probe_trials": len(probe_idx),
        "num_eval_samples": args.num_samples,
        "models": {},
    }
    for label, ckpt, use_pre, use_ex in models:
        results["models"][label] = audit_checkpoint(
            label,
            ckpt,
            dataset,
            test_idx,
            probe_idx,
            adjacency,
            np_adjacency,
            use_prestim=use_pre,
            use_extrasyn=use_ex,
            device=device,
            substeps_traj=[2, 4, 8, 16, 32],
            substeps_eval=[2, 4, 8, 16],
            num_eval_samples=args.num_samples,
        )

    # Flag Re-plan trigger check
    replan_flags = []
    for label, d in results["models"].items():
        cos_2 = d["probe_comparison"]["2"]["grad_cosine_vs_ref"]["overall"]
        r2_2 = d["full_test_eval"]["2"].get("evoked_hop1_r2", 0.0)
        r2_16 = d["full_test_eval"]["16"].get("evoked_hop1_r2", 0.0)
        if cos_2 < 0.7 or (r2_2 > 0 and r2_16 <= 0) or (r2_2 <= 0 and r2_16 > 0):
            replan_flags.append({
                "model": label,
                "grad_cosine_2_vs_32": cos_2,
                "evoked_hop1_r2_s2": r2_2,
                "evoked_hop1_r2_s16": r2_16,
            })
    results["replan_triggers"] = replan_flags

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {args.out}; replan_triggers={replan_flags}", flush=True)


if __name__ == "__main__":
    main()
