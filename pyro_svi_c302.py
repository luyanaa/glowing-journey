import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pyro
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

import c302
from utils import (
    AtlasLoader,
    RandiTrialDataset,
    resolve_default_data_path,
    responseGenerator,
    split_trial_dataset,
)


DEFAULT_DATA_PATH = resolve_default_data_path()


def _save_snapshot(snapshot_path: str, model, guide, epoch: int) -> None:
    target_model = model.module if hasattr(model, "module") else model
    snapshot = {
        "MODEL_STATE": target_model.state_dict(),
        "GUIDE_STATE": guide.state_dict() if (guide is not None and hasattr(guide, "state_dict")) else None,
        "PARAM_STORE": pyro.get_param_store().get_state(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")


def _safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or float(np.std(y_true)) <= 1e-8 or float(np.std(y_pred)) <= 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _extract_adjacency(model) -> Optional[np.ndarray]:
    step = getattr(model, "model", None)
    if step is None and hasattr(model, "decoder"):
        step = getattr(model.decoder, "model", None)
    if step is None or not hasattr(step, "synapse") or not hasattr(step.synapse, "adjacency_matrix"):
        return None
    n_neurons = int(getattr(step, "neuronSize", 300))
    return step.synapse.adjacency_matrix(n_neurons)


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    stim: Optional[np.ndarray] = None,
    pre_vols: int = 10,
    adjacency: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute overall, inlier (|y|<=3), direct-stimulated, and connectome hop-stratified metrics."""
    valid = mask.astype(bool)
    if not np.any(valid):
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan")}

    yt = y_true[valid]
    yp = y_pred[valid]
    out: Dict[str, float] = {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "pearson_r": _safe_pearson_r(yt, yp),
        "n_obs": int(yt.size),
    }

    inlier = valid & (np.abs(y_true) <= 3.0)
    if np.any(inlier):
        yt_i, yp_i = y_true[inlier], y_pred[inlier]
        out["inlier_r2"] = float(r2_score(yt_i, yp_i))
        out["inlier_mae"] = float(mean_absolute_error(yt_i, yp_i))
        out["inlier_rmse"] = float(np.sqrt(mean_squared_error(yt_i, yp_i)))
        out["inlier_r"] = _safe_pearson_r(yt_i, yp_i)

    # Strict future-only (post-stimulus t >= pre_vols) metrics for fair causal pre-stim comparison
    if y_true.ndim == 3 and 0 < int(pre_vols) < y_true.shape[1]:
        t0_post = int(pre_vols)
        post_valid = np.zeros_like(valid)
        post_valid[:, t0_post:, :] = valid[:, t0_post:, :]
        if np.any(post_valid):
            yt_p, yp_p = y_true[post_valid], y_pred[post_valid]
            out["post_r2"] = float(r2_score(yt_p, yp_p))
            out["post_mae"] = float(mean_absolute_error(yt_p, yp_p))
            out["post_rmse"] = float(np.sqrt(mean_squared_error(yt_p, yp_p)))
            out["post_r"] = _safe_pearson_r(yt_p, yp_p)
        post_inlier = post_valid & (np.abs(y_true) <= 3.0)
        if np.any(post_inlier):
            yt_pi, yp_pi = y_true[post_inlier], y_pred[post_inlier]
            out["post_inlier_r2"] = float(r2_score(yt_pi, yp_pi))
            out["post_inlier_mae"] = float(mean_absolute_error(yt_pi, yp_pi))
            out["post_inlier_rmse"] = float(np.sqrt(mean_squared_error(yt_pi, yp_pi)))
            out["post_inlier_r"] = _safe_pearson_r(yt_pi, yp_pi)

    # Baseline-subtracted "evoked" signals: remove each trial/neuron's
    # pre-stimulus mean before comparing post-stimulus responses.  This avoids
    # rewarding a model that merely reproduces the slow pre-stimulus baseline.
    yt_ev = yp_ev = None
    if y_true.ndim == 3 and 0 < int(pre_vols) < y_true.shape[1]:
        pre_valid = valid[:, :int(pre_vols), :]
        pre_count = np.maximum(pre_valid.sum(axis=1), 1)
        base_true = (y_true[:, :int(pre_vols), :] * pre_valid).sum(axis=1) / pre_count
        base_pred = (y_pred[:, :int(pre_vols), :] * pre_valid).sum(axis=1) / pre_count
        yt_ev = y_true - base_true[:, None, :]
        yp_ev = y_pred - base_pred[:, None, :]
        post_ev = np.zeros_like(valid)
        post_ev[:, int(pre_vols):, :] = valid[:, int(pre_vols):, :]
        if np.any(post_ev):
            yt_e, yp_e = yt_ev[post_ev], yp_ev[post_ev]
            out["evoked_post_r2"] = float(r2_score(yt_e, yp_e))
            out["evoked_post_r"] = _safe_pearson_r(yt_e, yp_e)

    if stim is not None and y_true.ndim == 3:
        n_trials, n_t, n_neurons = y_true.shape
        t0 = min(max(0, int(pre_vols)), max(0, n_t - 1))
        # Cover full nuclear GCaMP6s rise + peak (~10.5s) + early decay (~17.5s post-stim at 2 Hz)
        t1_stim = min(n_t, t0 + 35)
        t1_down = min(n_t, t0 + 35)

        stim_target = stim.sum(axis=1) > 0  # (B, N)
        evoked_stim = np.zeros_like(valid)
        evoked_stim[:, t0:t1_stim, :] = stim_target[:, None, :] & valid[:, t0:t1_stim, :]
        if np.any(evoked_stim):
            yt_s, yp_s = y_true[evoked_stim], y_pred[evoked_stim]
            out["stim_count"] = int(yt_s.size)
            out["stim_r2"] = float(r2_score(yt_s, yp_s))
            out["stim_mae"] = float(mean_absolute_error(yt_s, yp_s))
            out["stim_rmse"] = float(np.sqrt(mean_squared_error(yt_s, yp_s)))
            out["stim_r"] = _safe_pearson_r(yt_s, yp_s)
            out["stim_true_mean"] = float(yt_s.mean())
            out["stim_pred_mean"] = float(yp_s.mean())
            if yt_ev is not None:
                yt_se, yp_se = yt_ev[evoked_stim], yp_ev[evoked_stim]
                out["evoked_stim_r2"] = float(r2_score(yt_se, yp_se))
                out["evoked_stim_r"] = _safe_pearson_r(yt_se, yp_se)
            evoked_stim_in = evoked_stim & (np.abs(y_true) <= 3.0)
            if np.any(evoked_stim_in):
                out["stim_inlier_r"] = _safe_pearson_r(y_true[evoked_stim_in], y_pred[evoked_stim_in])

        if adjacency is not None and adjacency.shape == (n_neurons, n_neurons):
            adj1 = adjacency.astype(bool)
            adj2 = ((adj1.astype(np.int32) @ adj1.astype(np.int32)) > 0) & (~adj1) & (~np.eye(n_neurons, dtype=bool))
            # (B, N) downstream boolean indicators per trial
            hop1_neu = (stim_target.astype(np.int32) @ adj1.astype(np.int32)) > 0
            hop1_neu &= ~stim_target
            hop2_neu = (stim_target.astype(np.int32) @ adj2.astype(np.int32)) > 0
            hop2_neu &= (~stim_target) & (~hop1_neu)
            unconn_neu = (~stim_target) & (~hop1_neu) & (~hop2_neu)

            for prefix, neu_sel in [
                ("hop1", hop1_neu),
                ("hop2", hop2_neu),
                ("unconnected", unconn_neu),
            ]:
                sub_mask = np.zeros_like(valid)
                sub_mask[:, t0:t1_down, :] = neu_sel[:, None, :] & valid[:, t0:t1_down, :]
                if np.any(sub_mask):
                    yt_d, yp_d = y_true[sub_mask], y_pred[sub_mask]
                    out[f"{prefix}_count"] = int(yt_d.size)
                    out[f"{prefix}_r2"] = float(r2_score(yt_d, yp_d))
                    out[f"{prefix}_mae"] = float(mean_absolute_error(yt_d, yp_d))
                    out[f"{prefix}_rmse"] = float(np.sqrt(mean_squared_error(yt_d, yp_d)))
                    out[f"{prefix}_r"] = _safe_pearson_r(yt_d, yp_d)
                    if yt_ev is not None:
                        yt_de, yp_de = yt_ev[sub_mask], yp_ev[sub_mask]
                        out[f"evoked_{prefix}_r2"] = float(r2_score(yt_de, yp_de))
                        out[f"evoked_{prefix}_r"] = _safe_pearson_r(yt_de, yp_de)
                    sub_in = sub_mask & (np.abs(y_true) <= 3.0)
                    if np.any(sub_in):
                        out[f"{prefix}_inlier_r"] = _safe_pearson_r(y_true[sub_in], y_pred[sub_in])

    return out


def compute_drift_gate(stim: torch.Tensor, adjacency: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
    """Build a causal per-neuron gate for pre-stimulus drift using only stimulus schedule + connectome."""
    if mode == "none" or adjacency is None:
        return None
    stim_target = stim.sum(dim=1) > 0
    if mode == "direct":
        return (~stim_target).to(dtype=stim.dtype)
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)
    return (~(stim_target | hop1)).to(dtype=stim.dtype)


def evaluate_predictions(
    model,
    data_loader: DataLoader,
    device: torch.device,
    guide=None,
    posterior_samples: dict = None,
    num_samples: int = 8,
    pre_vols: int = 10,
    adjacency: Optional[np.ndarray] = None,
    drift_gate_mode: str = "none",
) -> Dict[str, float]:
    """Evaluate posterior predictive GCaMP fit with overall, stimulated, and downstream metrics."""
    if posterior_samples is not None:
        predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=("_RETURN",))
    else:
        predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("_RETURN",))

    has_prestim = bool(
        getattr(model, "use_prestim_encoder", False)
        or getattr(getattr(model, "decoder", None), "use_prestim_encoder", False)
    )
    adj = adjacency if adjacency is not None else _extract_adjacency(model)
    adj_t = (
        torch.as_tensor(adj, dtype=torch.float32, device=device)
        if (adj is not None and has_prestim and drift_gate_mode != "none")
        else None
    )
    y_trues = []
    y_preds = []
    masks = []
    stims = []
    with torch.no_grad():
        for batch in data_loader:
            stim, y, mask, noise = [b.to(device) for b in batch]
            if has_prestim:
                t_cut = min(int(pre_vols), y.shape[1])
                gate = compute_drift_gate(stim, adj_t, drift_gate_mode) if adj_t is not None else None
                samples = predictive(
                    ExternalInput=stim,
                    mask=None,
                    y=None,
                    noise_scale=noise,
                    pre_stim_obs=y[:, :t_cut, :],
                    pre_stim_mask=mask[:, :t_cut, :],
                    drift_gate=gate,
                )
            else:
                samples = predictive(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
            pred_mean = samples["_RETURN"].mean(dim=0)
            y_trues.append(y.cpu().numpy())
            y_preds.append(pred_mean.cpu().numpy())
            masks.append(mask.bool().cpu().numpy())
            stims.append(stim.cpu().numpy())

    if not y_trues:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    return compute_stratified_metrics(
        np.concatenate(y_trues, axis=0),
        np.concatenate(y_preds, axis=0),
        np.concatenate(masks, axis=0),
        stim=np.concatenate(stims, axis=0),
        pre_vols=pre_vols,
        adjacency=adj,
    )


def _build_datasets(args) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    split_info: Dict[str, int] = {"grouped": 0, "train_recordings": 0, "test_recordings": 0}
    if args.source == "randi":
        if args.dataset_cache and os.path.isfile(args.dataset_cache):
            dataset = RandiTrialDataset.load(args.dataset_cache)
        else:
            loader = AtlasLoader(
                path=args.data_path,
                strain=args.strain,
                norm_mode=args.norm_mode,
                max_recordings=args.max_recordings,
            )
            dataset = loader.build_trial_dataset(
                pre_vols=args.pre_vols,
                post_vols=args.post_vols,
                stim_duration_s=args.stim_duration,
            )
            if args.dataset_cache:
                dataset.save(args.dataset_cache)
        train_idx, test_idx, split_info = split_trial_dataset(
            dataset,
            test_size=args.test_size,
            random_state=42,
            group_by_recording=not getattr(args, "trial_split", False),
            max_trials=args.max_trials,
        )
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)
    else:
        x_all, y_all, m_all = responseGenerator(
            folder="./wormfunconn/atlas/",
            strain=args.strain,
            size=args.synthetic_len,
            dt=args.dt_obs,
        ).Dataset(return_mask=True)
        noise_all = torch.full((x_all.shape[0], x_all.shape[2]), 0.05, dtype=torch.float32)
        idx = list(range(len(x_all)))
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=42)
        train_ds = _TupleDataset(
            x_all[train_idx], y_all[train_idx], m_all[train_idx], noise_all[train_idx]
        )
        test_ds = _TupleDataset(
            x_all[test_idx], y_all[test_idx], m_all[test_idx], noise_all[test_idx]
        )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=len(train_ds) >= args.batch_size,
        pin_memory=pin,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(args.batch_size, len(test_ds)),
        shuffle=False,
        pin_memory=pin,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader, split_info


class _TupleDataset(torch.utils.data.Dataset):
    def __init__(self, stim, y, mask, noise):
        self.stim = stim
        self.y = y
        self.mask = mask
        self.noise = noise

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.stim[idx], self.y[idx], self.mask[idx], self.noise[idx]


def init_to_prior_median(site=None):
    """Initialize AutoGuide locations at exact prior medians (handles Independent & TransformedDistribution)."""
    if site is None:
        return init_to_prior_median
    fn = site["fn"]
    while isinstance(fn, pyro.distributions.Independent):
        fn = fn.base_dist
    if isinstance(fn, torch.distributions.TransformedDistribution):
        val = fn.base_dist.mean.clone()
        for tr in fn.transforms:
            val = tr(val)
        return val
    if hasattr(fn, "mean"):
        return fn.mean.clone()
    return pyro.infer.autoguide.init_to_median(site)


def format_eval_summary(metrics: Dict[str, float]) -> str:
    parts = [
        f"R2={metrics.get('r2', float('nan')):.4f}",
        f"MAE={metrics.get('mae', float('nan')):.4f}",
        f"RMSE={metrics.get('rmse', float('nan')):.4f}",
    ]
    if "inlier_r2" in metrics:
        parts.append(f"inlier_R2={metrics['inlier_r2']:.4f}")
    if "post_inlier_r2" in metrics:
        parts.append(f"post_inR2={metrics['post_inlier_r2']:.4f}")
    if "stim_r2" in metrics:
        parts.append(f"stim_R2={metrics['stim_r2']:.4f}(r={metrics['stim_r']:.3f})")
    if "hop1_r" in metrics:
        parts.append(f"hop1_r={metrics['hop1_r']:.3f}(R2={metrics['hop1_r2']:.4f})")
    if "hop2_r" in metrics:
        parts.append(f"hop2_r={metrics['hop2_r']:.3f}")
    if "unconnected_r" in metrics:
        parts.append(f"unconn_r={metrics['unconnected_r']:.3f}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit C. elegans connectome + GCaMP model with Pyro SVI/MCMC")
    parser.add_argument("total_epochs", nargs="?", type=int, default=10, help="Total epochs to train (SVI)")
    parser.add_argument("save_every", nargs="?", type=int, default=5, help="Checkpoint frequency in epochs")
    parser.add_argument("--batch_size", "--batch-size", default=16, type=int, help="Mini-batch size")
    parser.add_argument("--source", choices=["randi", "synthetic"], default="randi")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--dataset-cache", default=None, type=str, help="Optional .npz cache for preprocessed trials")
    parser.add_argument("--strain", default="wild-type", type=str)
    parser.add_argument("--norm-mode", choices=["dff_safe", "dff_strict", "robust_z"], default="dff_safe")
    parser.add_argument("--mode", choices=["svi", "mcmc"], default="svi")
    parser.add_argument("--guide", choices=["diag", "lowrank"], default="diag")
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--substeps", default=4, type=int)
    parser.add_argument("--dt-obs", default=0.5, type=float)
    parser.add_argument("--pre-vols", default=10, type=int)
    parser.add_argument("--post-vols", default=50, type=int)
    parser.add_argument("--stim-duration", default=0.5, type=float)
    parser.add_argument("--synthetic-len", default=100, type=int)
    parser.add_argument("--test-size", default=0.1, type=float)
    parser.add_argument("--max-recordings", default=None, type=int)
    parser.add_argument("--max-trials", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--use-bnn", action="store_true")
    parser.add_argument("--use-extrasynaptic", action="store_true")
    parser.add_argument("--use-sensory-models", action="store_true")
    parser.add_argument("--use-prestim-encoder", action="store_true")
    parser.add_argument("--prestim-rank", default=16, type=int)
    parser.add_argument("--likelihood", choices=["student_t", "normal"], default="student_t")
    parser.add_argument("--mcmc-samples", default=20, type=int)
    parser.add_argument("--mcmc-warmup", default=20, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--trial-split", action="store_true", help="Disable recording-level grouping in split")
    parser.add_argument("--resume", default=None, type=str, help="Optional snapshot .pt path to resume SVI")
    parser.add_argument("--snapshot-dir", default=".", type=str)
    args = parser.parse_args()

    pyro.clear_param_store()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    os.makedirs(args.snapshot_dir, exist_ok=True)

    train_loader, test_loader, split_info = _build_datasets(args)
    n_train_trials = len(train_loader.dataset)
    print(
        f"Dataset ready ({args.source}): train={n_train_trials} trials "
        f"({split_info['train_recordings']} recs, {len(train_loader)} batches), "
        f"test={len(test_loader.dataset)} trials ({split_info['test_recordings']} recs, "
        f"{len(test_loader)} batches, grouped={bool(split_info['grouped'])})"
    )

    step_model = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_sensory_models=args.use_sensory_models,
        use_extrasynaptic=args.use_extrasynaptic,
        use_bnn_residual=args.use_bnn,
    )
    model = c302.RecurrentNematode(
        step_model,
        dt_obs=args.dt_obs,
        substeps=args.substeps,
        likelihood=args.likelihood,
        use_prestim_encoder=args.use_prestim_encoder,
        pre_vols=args.pre_vols,
        prestim_rank=args.prestim_rank,
    ).to(device)
    summary(model)

    if args.mode == "mcmc":
        stim_b, y_b, mask_b, noise_b = next(iter(train_loader))
        stim_b, y_b, mask_b, noise_b = (
            stim_b.to(device),
            y_b.to(device),
            mask_b.to(device),
            noise_b.to(device),
        )
        nuts_kernel = NUTS(model, jit_compile=False, max_tree_depth=5)
        mcmc = MCMC(nuts_kernel, num_samples=args.mcmc_samples, warmup_steps=args.mcmc_warmup)
        mcmc.run(ExternalInput=stim_b, mask=mask_b, y=y_b, noise_scale=noise_b)
        metrics = evaluate_predictions(
            model, test_loader, device, posterior_samples=mcmc.get_samples()
        )
        print(f"MCMC Test Metrics: {metrics}")
        return

    if args.guide == "lowrank":
        guide = AutoLowRankMultivariateNormal(
            model, rank=16, init_loc_fn=init_to_prior_median, init_scale=0.02
        )
    else:
        guide = AutoDiagonalNormal(
            model, init_loc_fn=init_to_prior_median, init_scale=0.02
        )

    optim = pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optim, Trace_ELBO())

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        stim0, y0, mask0, noise0 = next(iter(train_loader))
        _ = guide(
            ExternalInput=stim0.to(device),
            mask=mask0.to(device),
            y=y0.to(device),
            noise_scale=noise0.to(device),
            dataset_size=n_train_trials,
        )
        snap = torch.load(args.resume, map_location=device, weights_only=False)
        pyro.get_param_store().set_state(snap["PARAM_STORE"])
        model.load_state_dict(snap["MODEL_STATE"])
        if snap.get("GUIDE_STATE") is not None:
            guide.load_state_dict(snap["GUIDE_STATE"])
        start_epoch = int(snap.get("EPOCHS_RUN", 0))
        print(f"Resumed from {args.resume} (starting at epoch {start_epoch})")

    for epoch in range(start_epoch, start_epoch + args.total_epochs):
        t0 = time.time()
        total_loss = 0.0
        scaled_obs = 0.0
        for stim, y, mask, noise in train_loader:
            stim = stim.to(device)
            y = y.to(device)
            mask = mask.to(device)
            noise = noise.to(device)
            bsz = max(1, int(stim.shape[0]))
            scale_factor = float(n_train_trials) / float(bsz) if n_train_trials > bsz else 1.0
            total_loss += svi.step(
                ExternalInput=stim,
                mask=mask,
                y=y,
                noise_scale=noise,
                dataset_size=n_train_trials,
            )
            scaled_obs += float(mask.sum().item()) * scale_factor

        avg_loss = total_loss / max(1.0, scaled_obs)
        dt_ep = time.time() - t0
        print(f"Epoch {epoch:03d} | elbo/obs={avg_loss:.4f} | total_loss={total_loss:.2f} | dt={dt_ep:.2f}s")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            metrics = evaluate_predictions(
                model, test_loader, device, guide=guide, num_samples=4, pre_vols=args.pre_vols
            )
            print(f"  Eval | {format_eval_summary(metrics)}")
            snap_path = os.path.join(args.snapshot_dir, f"snapshot_epoch_{epoch+1}.pt")
            _save_snapshot(snap_path, model, guide, epoch + 1)
            metrics_path = os.path.join(args.snapshot_dir, f"metrics_epoch_{epoch+1}.json")
            with open(metrics_path, "w") as mf:
                json.dump({"epoch": epoch + 1, "elbo_per_obs": avg_loss, **metrics}, mf, indent=2)


if __name__ == "__main__":
    main()
