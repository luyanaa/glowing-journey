import argparse
import os

import pyro
from pyro.infer import SVI, Trace_ELBO
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

import c302
from pyro_svi_c302 import DEFAULT_DATA_PATH, _save_snapshot, evaluate_predictions
from utils import AtlasLoader, RandiTrialDataset, responseGenerator
import vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAElegans amortized variational connectome model")
    parser.add_argument("total_epochs", nargs="?", type=int, default=10, help="Total epochs to train")
    parser.add_argument("save_every", nargs="?", type=int, default=5, help="Checkpoint frequency in epochs")
    parser.add_argument("--batch_size", "--batch-size", default=16, type=int)
    parser.add_argument("--source", choices=["randi", "synthetic"], default="randi")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--dataset-cache", default=None, type=str)
    parser.add_argument("--strain", default="wild-type", type=str)
    parser.add_argument("--norm-mode", choices=["dff_safe", "dff_strict", "robust_z"], default="dff_safe")
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--substeps", default=4, type=int)
    parser.add_argument("--dt-obs", default=0.5, type=float)
    parser.add_argument("--pre-vols", default=10, type=int)
    parser.add_argument("--post-vols", default=50, type=int)
    parser.add_argument("--max-recordings", default=None, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--snapshot-dir", default=".", type=str)
    args = parser.parse_args()

    pyro.clear_param_store()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.snapshot_dir, exist_ok=True)

    if args.source == "randi":
        if args.dataset_cache and os.path.isfile(args.dataset_cache):
            dataset = RandiTrialDataset.load(args.dataset_cache)
        else:
            dataset = AtlasLoader(
                path=args.data_path,
                strain=args.strain,
                norm_mode=args.norm_mode,
                max_recordings=args.max_recordings,
            ).build_trial_dataset(pre_vols=args.pre_vols, post_vols=args.post_vols)
            if args.dataset_cache:
                dataset.save(args.dataset_cache)
        indices = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)
    else:
        x_all, y_all, m_all = responseGenerator(
            folder="./wormfunconn/atlas/", strain=args.strain, size=100, dt=args.dt_obs
        ).Dataset(return_mask=True)
        noise_all = torch.full((x_all.shape[0], x_all.shape[2]), 0.05, dtype=torch.float32)
        from pyro_svi_c302 import _TupleDataset

        idx = list(range(len(x_all)))
        train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
        train_ds = _TupleDataset(x_all[train_idx], y_all[train_idx], m_all[train_idx], noise_all[train_idx])
        test_ds = _TupleDataset(x_all[test_idx], y_all[test_idx], m_all[test_idx], noise_all[test_idx])

    pin = torch.cuda.is_available()
    train_data = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        pin_memory=pin,
        shuffle=True,
        drop_last=len(train_ds) >= args.batch_size,
        num_workers=args.num_workers,
    )
    test_data = DataLoader(
        test_ds,
        batch_size=min(args.batch_size, len(test_ds)),
        pin_memory=pin,
        shuffle=False,
        num_workers=args.num_workers,
    )

    step_model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    rec_model = c302.RecurrentNematode(step_model, dt_obs=args.dt_obs, substeps=args.substeps)
    model = vae.VAElegans(
        rec_model,
        rec_model.model.Neuron.neuronSize,
        len(rec_model.model.synapse.Wicks_SRC),
        len(rec_model.model.synapse.Gap_Junction_SRC),
    ).to(device)
    summary(model)

    optim = pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0})
    svi = SVI(model, model.guide, optim, Trace_ELBO())

    for epoch in range(args.total_epochs):
        loss = 0.0
        n_obs = 0
        for stim, y, mask, noise in train_data:
            stim = stim.to(device)
            y = y.to(device)
            mask = mask.to(device)
            noise = noise.to(device)
            loss += svi.step(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
            n_obs += int(mask.sum().item())
        print(f"Epoch {epoch:03d} | elbo/obs={loss / max(1, n_obs):.4f} | total_loss={loss:.2f}")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            metrics = evaluate_predictions(model, test_data, device, guide=model.guide, num_samples=2)
            print(f"  Eval | R2={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
            _save_snapshot(os.path.join(args.snapshot_dir, f"vae_epoch_{epoch+1}.pt"), model, model.guide, epoch + 1)


if __name__ == "__main__":
    main()
