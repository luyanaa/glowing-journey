import argparse
import json
import os
import time

import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

import c302
from pyro_svi_c302 import (
    DEFAULT_DATA_PATH,
    _save_snapshot,
    evaluate_predictions,
    format_eval_summary,
)
from utils import (
    AtlasLoader,
    RandiTrialDataset,
    responseGenerator,
    split_trial_dataset,
    split_train_validation_indices,
    ToyoshimaLoader,
    resolve_default_toyoshima_path,
)
import vae


def compute_drift_gate(stim: torch.Tensor, adjacency: torch.Tensor, mode: str) -> torch.Tensor:
    """Build a causal per-neuron gate for the pre-stimulus drift terms.

    Uses only the known optogenetic stimulus schedule and the fixed connectome;
    no future GCaMP observation is read.  ``direct`` gates only direct stimulus
    targets, ``direct_hop1`` also gates their 1-hop postsynaptic partners.
    """
    if mode == "none":
        return None
    stim_target = stim.sum(dim=1) > 0
    if mode == "direct":
        return (~stim_target).to(dtype=stim.dtype)
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)
    return (~(stim_target | hop1)).to(dtype=stim.dtype)


def reweight_hop1_noise(
    noise: torch.Tensor, stim: torch.Tensor, adjacency: torch.Tensor, weight: float
) -> torch.Tensor:
    """Shrink the observation noise (increase likelihood weight) on 1-hop neurons."""
    if weight <= 1.0:
        return noise
    stim_target = stim.sum(dim=1) > 0
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)
    factor = torch.where(
        hop1,
        torch.full_like(noise, 1.0 / float(weight)),
        torch.ones_like(noise),
    )
    return noise * factor


def _load_compatible_state_dict(module, state):
    """Load only keys whose names and shapes match the current module."""
    current = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return skipped, list(missing), list(unexpected)


def _expand_scalar_kinetics(module, state):
    """Initialize per-neuron tau_ca/k_off loc+scale from a scalar v3 checkpoint."""
    pairs = [
        ("guide.tau_ca_loc_unconstrained", "tau_ca_loc"),
        ("guide.tau_ca_scale_unconstrained", "tau_ca_scale"),
        ("guide.k_off_loc_unconstrained", "k_off_loc"),
        ("guide.k_off_scale_unconstrained", "k_off_scale"),
    ]
    expanded = []
    for key, attr in pairs:
        if key in state and state[key].numel() == 1:
            target = getattr(module.guide, attr)
            with torch.no_grad():
                target.fill_(float(state[key].detach().cpu()))
            expanded.append(key)
    return expanded


def _set_kinetics_only_trainable(module, kinetics_only: bool):
    for name, param in module.named_parameters():
        if kinetics_only:
            param.requires_grad_("tau_ca" in name or "k_off" in name)
        else:
            param.requires_grad_(True)


def _set_edge_only_trainable(module, edge_only: bool):
    for name, param in module.named_parameters():
        if edge_only:
            param.requires_grad_("v_rest_edge" in name or "v_slope_edge" in name)
        else:
            param.requires_grad_(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAElegans amortized variational connectome model")
    parser.add_argument("total_epochs", nargs="?", type=int, default=10, help="Total epochs to train")
    parser.add_argument("save_every", nargs="?", type=int, default=5, help="Checkpoint frequency in epochs")
    parser.add_argument("--batch_size", "--batch-size", default=16, type=int)
    parser.add_argument("--source", choices=["randi", "toyoshima", "synthetic"], default="randi")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--toyoshima-path", default=resolve_default_toyoshima_path(), type=str)
    parser.add_argument("--toyoshima-targets", default="ASH,AWC,ASE", type=str,
                        help="Comma-separated operational NaCl sensory targets")
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
    parser.add_argument("--use-prestim-encoder", action="store_true")
    parser.add_argument("--prestim-rank", default=16, type=int)
    parser.add_argument("--drift-gate", choices=["none", "direct", "direct_hop1"], default="none",
                        help="Gate pre-stimulus drift on direct/1-hop stimulus targets")
    parser.add_argument("--hop1-loss-weight", default=1.0, type=float,
                        help="Relative likelihood weight on 1-hop downstream neurons")
    parser.add_argument("--per-neuron-kinetics", action="store_true",
                        help="Learn per-neuron tau_ca and k_off (hierarchical around v3 global values)")
    parser.add_argument("--per-edge-release", action="store_true",
                        help="Learn per-edge V_rest/V_slope residuals on top of per-neuron release values")
    parser.add_argument("--per-neuron-stim-gain", action="store_true",
                        help="Learn a constrained stimulus gain per neuron")
    parser.add_argument("--stim-actuator", action="store_true",
                        help="Apply a first-order rise/decay actuator before the membrane")
    parser.add_argument("--stim-tau-rise", default=0.08, type=float)
    parser.add_argument("--stim-tau-decay", default=0.4, type=float)
    parser.add_argument("--freeze-nonedge-epochs", default=0, type=int,
                        help="Freeze all parameters except per-edge release residuals for this many epochs")
    parser.add_argument("--init-from", default=None, type=str,
                        help="Warm-start checkpoint; scalar tau_ca/k_off are expanded to per-neuron vectors")
    parser.add_argument("--freeze-nonkinetics-epochs", default=0, type=int,
                        help="Freeze all parameters except guide tau_ca/k_off for this many epochs")
    parser.add_argument("--trial-split", action="store_true", help="Disable recording-level grouping in split")
    parser.add_argument("--seed", default=42, type=int, help="Training RNG seed")
    parser.add_argument("--split-seed", default=42, type=int, help="Fixed data split seed")
    parser.add_argument("--test-size", default=0.1, type=float,
                        help="Recording-grouped test fraction (default 0.1)")
    parser.add_argument("--val-fraction", default=0.15, type=float,
                        help="Recording-grouped validation fraction held out from training")
    parser.add_argument("--eval-test-each-save", action="store_true",
                        help="Evaluate the locked test set at every save (default: final save only)")
    parser.add_argument("--causal-preprocess", action="store_true",
                        help="Use past-only Hampel/baseline/normalization for the Randi preprocessing")
    parser.add_argument("--snapshot-dir", default=".", type=str)
    args = parser.parse_args()

    pyro.clear_param_store()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
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
                causal_preprocess=args.causal_preprocess,
            ).build_trial_dataset(pre_vols=args.pre_vols, post_vols=args.post_vols)
            if args.dataset_cache:
                dataset.save(args.dataset_cache)
        source_label = "randi"
    elif args.source == "toyoshima":
        if args.dataset_cache and os.path.isfile(args.dataset_cache):
            dataset = RandiTrialDataset.load(args.dataset_cache)
        else:
            targets = tuple(t.strip() for t in args.toyoshima_targets.split(",") if t.strip())
            dataset = ToyoshimaLoader(
                args.toyoshima_path,
                target_fs=1.0 / float(args.dt_obs),
                stim_targets=targets,
                max_recordings=args.max_recordings,
            ).build_trial_dataset(pre_vols=args.pre_vols, post_vols=args.post_vols)
            if args.dataset_cache:
                dataset.save(args.dataset_cache)
        source_label = "toyoshima"
    else:
        x_all, y_all, m_all = responseGenerator(
            folder="./wormfunconn/atlas/", strain=args.strain, size=100, dt=args.dt_obs
        ).Dataset(return_mask=True)
        noise_all = torch.full((x_all.shape[0], x_all.shape[2]), 0.05, dtype=torch.float32)
        from pyro_svi_c302 import _TupleDataset

        idx = list(range(len(x_all)))
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=42)
        train_ds = _TupleDataset(x_all[train_idx], y_all[train_idx], m_all[train_idx], noise_all[train_idx])
        test_ds = _TupleDataset(x_all[test_idx], y_all[test_idx], m_all[test_idx], noise_all[test_idx])
        val_ds = None
        source_label = "synthetic"

    if args.source != "synthetic":
        train_idx, test_idx, split_info = split_trial_dataset(
            dataset,
            test_size=args.test_size,
            random_state=args.split_seed,
            group_by_recording=not args.trial_split,
        )
        test_ds = Subset(dataset, test_idx)
        if args.val_fraction > 0:
            train_core_idx, val_idx, val_info = split_train_validation_indices(
                dataset, train_idx, val_size=args.val_fraction, random_state=args.split_seed,
                group_by_recording=not args.trial_split,
            )
            val_ds = Subset(dataset, val_idx)
        else:
            train_core_idx = list(train_idx)
            val_idx = []
            val_info = {"val_recordings": 0}
            val_ds = None
        train_ds = Subset(dataset, train_core_idx)
        print(
            f"Dataset ready ({source_label}): train={len(train_ds)} trials "
            f"({split_info['train_recordings']} recs, val={val_info['val_recordings']} recs), "
            f"val={0 if val_ds is None else len(val_ds)} trials, test={len(test_ds)} trials "
            f"({split_info['test_recordings']} recs, grouped={bool(split_info['grouped'])})"
        )

    pin = torch.cuda.is_available()
    train_data = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        pin_memory=pin,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=len(train_ds) >= args.batch_size,
        num_workers=args.num_workers,
    )
    val_data = (
        DataLoader(
            val_ds,
            batch_size=min(args.batch_size, len(val_ds)),
            pin_memory=pin,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_ds is not None
        else None
    )
    test_data = DataLoader(
        test_ds,
        batch_size=min(args.batch_size, len(test_ds)),
        pin_memory=pin,
        shuffle=False,
        num_workers=args.num_workers,
    )

    step_model = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        per_edge_release=args.per_edge_release,
        per_neuron_stim_gain=args.per_neuron_stim_gain,
    )
    adj_t = torch.as_tensor(
        step_model.synapse.adjacency_matrix(step_model.neuronSize),
        dtype=torch.float32,
        device=device,
    )
    rec_model = c302.RecurrentNematode(
        step_model,
        dt_obs=args.dt_obs,
        substeps=args.substeps,
        use_prestim_encoder=args.use_prestim_encoder,
        pre_vols=args.pre_vols,
        prestim_rank=args.prestim_rank,
        per_neuron_kinetics=args.per_neuron_kinetics,
        per_edge_release=args.per_edge_release,
        stim_actuator=args.stim_actuator,
        stim_tau_rise=args.stim_tau_rise,
        stim_tau_decay=args.stim_tau_decay,
    )
    model = vae.VAElegans(
        rec_model,
        rec_model.model.Neuron.neuronSize,
        len(rec_model.model.synapse.Wicks_SRC),
        len(rec_model.model.synapse.Gap_Junction_SRC),
    ).to(device)
    if args.init_from:
        snap = torch.load(args.init_from, map_location=device, weights_only=False)
        state = snap["MODEL_STATE"]
        skipped, missing, unexpected = _load_compatible_state_dict(model, state)
        expanded = _expand_scalar_kinetics(model, state) if args.per_neuron_kinetics else []
        print(
            f"Warm start from {args.init_from}: loaded compatible={len(state)-len(skipped)} "
            f"skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)} "
            f"expanded_kinetics={expanded}",
            flush=True,
        )
    summary(model)

    optim = pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0})
    svi = SVI(model, model.guide, optim, Trace_ELBO())

    n_train_trials = len(train_ds)
    freeze_epochs = max(0, int(args.freeze_nonkinetics_epochs))
    edge_freeze_epochs = max(0, int(args.freeze_nonedge_epochs))
    for epoch in range(args.total_epochs):
        if freeze_epochs:
            frozen = epoch < freeze_epochs
            _set_kinetics_only_trainable(model, frozen)
            if epoch == freeze_epochs:
                print("Unfreezing all parameters after kinetics-only warm start", flush=True)
        if edge_freeze_epochs:
            frozen = epoch < edge_freeze_epochs
            _set_edge_only_trainable(model, frozen)
            if epoch == edge_freeze_epochs:
                print("Unfreezing all parameters after edge-only warm start", flush=True)
        t0 = time.time()
        loss = 0.0
        scaled_obs = 0.0
        for stim, y, mask, noise in train_data:
            stim = stim.to(device)
            y = y.to(device)
            mask = mask.to(device)
            noise = noise.to(device)
            bsz = max(1, int(stim.shape[0]))
            scale_factor = float(n_train_trials) / float(bsz) if n_train_trials > bsz else 1.0
            drift_gate = compute_drift_gate(stim, adj_t, args.drift_gate) if args.use_prestim_encoder else None
            noise = reweight_hop1_noise(noise, stim, adj_t, args.hop1_loss_weight)
            loss += svi.step(
                ExternalInput=stim,
                mask=mask,
                y=y,
                noise_scale=noise,
                dataset_size=n_train_trials,
                drift_gate=drift_gate,
            )
            scaled_obs += float(mask.sum().item()) * scale_factor
        avg_loss = loss / max(1.0, scaled_obs)
        dt_ep = time.time() - t0
        print(f"Epoch {epoch:03d} | elbo/obs={avg_loss:.4f} | total_loss={loss:.2f} | dt={dt_ep:.2f}s")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            gate_mode = args.drift_gate if args.use_prestim_encoder else "none"
            val_metrics = {}
            if val_data is not None:
                val_metrics = evaluate_predictions(
                    model,
                    val_data,
                    device,
                    guide=model.guide,
                    num_samples=4,
                    pre_vols=args.pre_vols,
                    drift_gate_mode=gate_mode,
                )
                print(f"  Val  | {format_eval_summary(val_metrics)}")
            test_metrics = {}
            if args.eval_test_each_save or (epoch + 1) == args.total_epochs:
                test_metrics = evaluate_predictions(
                    model,
                    test_data,
                    device,
                    guide=model.guide,
                    num_samples=4,
                    pre_vols=args.pre_vols,
                    drift_gate_mode=gate_mode,
                )
                print(f"  Test | {format_eval_summary(test_metrics)}")
            _save_snapshot(os.path.join(args.snapshot_dir, f"vae_epoch_{epoch+1}.pt"), model, model.guide, epoch + 1)
            metrics_path = os.path.join(args.snapshot_dir, f"metrics_epoch_{epoch+1}.json")
            with open(metrics_path, "w") as mf:
                json.dump({"epoch": epoch + 1, "elbo_per_obs": avg_loss,
                           "val": val_metrics, "test": test_metrics}, mf, indent=2)


if __name__ == "__main__":
    main()
