import argparse
import os
import time
from typing import Dict, Tuple

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
from utils import AtlasLoader, PsuedoDataset, RandiTrialDataset, responseGenerator


DEFAULT_DATA_PATH = "/Users/yanlu/Google Drive/マイドライブ/exported_data"


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


def evaluate_predictions(
    model,
    data_loader: DataLoader,
    device: torch.device,
    guide=None,
    posterior_samples: dict = None,
    num_samples: int = 8,
) -> Dict[str, float]:
    """Evaluate posterior predictive GCaMP fit on masked observed entries."""
    if posterior_samples is not None:
        predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=("_RETURN",))
    else:
        predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=("_RETURN",))

    y_trues = []
    y_preds = []
    with torch.no_grad():
        for batch in data_loader:
            stim, y, mask, noise = [b.to(device) for b in batch]
            samples = predictive(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
            pred_mean = samples["_RETURN"].mean(dim=0)
            valid = mask.bool()
            if torch.any(valid):
                y_trues.append(y[valid].cpu().numpy())
                y_preds.append(pred_mean[valid].cpu().numpy())

    if not y_trues:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    y_true_np = np.concatenate(y_trues, axis=0)
    y_pred_np = np.concatenate(y_preds, axis=0)
    return {
        "r2": float(r2_score(y_true_np, y_pred_np)),
        "mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
    }


def _build_datasets(args) -> Tuple[DataLoader, DataLoader]:
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
        if args.max_trials is not None and len(dataset) > args.max_trials:
            indices = list(range(args.max_trials))
        else:
            indices = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
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
        train_ds = PsuedoDataset(
            x_all[train_idx], y_all[train_idx], m_all[train_idx], noise_all[train_idx]
        )
        # Reorder to match RandiTrialDataset (stim, y, mask, noise)
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
    return train_loader, test_loader


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
    """Initialize AutoGuide locations at exact prior medians (unwrap Independent, not LogNormal)."""
    if site is None:
        return init_to_prior_median
    fn = site["fn"]
    while isinstance(fn, pyro.distributions.Independent):
        fn = fn.base_dist
    if isinstance(fn, torch.distributions.LogNormal):
        return torch.exp(fn.loc).clone()
    if hasattr(fn, "mean"):
        return fn.mean.clone()
    return pyro.infer.autoguide.init_to_median(site)


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
    parser.add_argument("--likelihood", choices=["student_t", "normal"], default="student_t")
    parser.add_argument("--mcmc-samples", default=20, type=int)
    parser.add_argument("--mcmc-warmup", default=20, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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

    train_loader, test_loader = _build_datasets(args)
    print(f"Dataset ready ({args.source}): train batches={len(train_loader)}, test batches={len(test_loader)}")

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

    for epoch in range(args.total_epochs):
        t0 = time.time()
        total_loss = 0.0
        n_obs = 0
        for stim, y, mask, noise in train_loader:
            stim = stim.to(device)
            y = y.to(device)
            mask = mask.to(device)
            noise = noise.to(device)
            total_loss += svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
            n_obs += int(mask.sum().item())

        avg_loss = total_loss / max(1, n_obs)
        dt_ep = time.time() - t0
        print(f"Epoch {epoch:03d} | elbo/obs={avg_loss:.4f} | total_loss={total_loss:.2f} | dt={dt_ep:.2f}s")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            metrics = evaluate_predictions(model, test_loader, device, guide=guide, num_samples=4)
            print(f"  Eval | R2={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
            snap_path = os.path.join(args.snapshot_dir, f"snapshot_epoch_{epoch+1}.pt")
            _save_snapshot(snap_path, model, guide, epoch + 1)


if __name__ == "__main__":
    main()
