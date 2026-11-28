import argparse
import json
import os
import time

import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchinfo import summary

import c302
from pyro_svi_c302 import (
    DEFAULT_DATA_PATH,
    _save_snapshot,
    evaluate_predictions,
    extract_and_expand_posterior_samples,
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
    stim_target = stim.abs().sum(dim=1) > 0
    if mode == "direct":
        return (~stim_target).to(dtype=stim.dtype)
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)
    return (~(stim_target | hop1)).to(dtype=stim.dtype)


def reweight_hop1_noise(
    noise: torch.Tensor,
    stim: torch.Tensor,
    adjacency: torch.Tensor,
    weight: float,
    stim_weight: float = 1.0,
) -> torch.Tensor:
    """Shrink observation noise (increase likelihood weight) on 1-hop and stim targets.

    ``weight`` controls the 1-hop factor; ``stim_weight`` independently controls
    the direct stimulus targets, which are only ~1.4% of post-stim observations
    and otherwise disappear into the background likelihood.
    """
    stim_target = stim.abs().sum(dim=1) > 0
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)
    factor = torch.ones_like(noise)
    if stim_weight > 1.0:
        factor = torch.where(
            stim_target,
            torch.full_like(noise, 1.0 / float(stim_weight)),
            factor,
        )
    if weight > 1.0:
        factor = torch.where(
            hop1,
            torch.full_like(noise, 1.0 / float(weight)),
            factor,
        )
    return noise * factor


def _prestim_baseline_torch(y: torch.Tensor, mask: torch.Tensor, pre_vols: int) -> torch.Tensor:
    """Differentiable causal pre-stim linear extrapolation baseline per (B, N)."""
    t0 = max(1, min(int(pre_vols), y.shape[1]))
    n_t = y.shape[1]
    w = mask[:, :t0, :].to(dtype=y.dtype)
    denom = torch.clamp(w.sum(dim=1), min=1.0)
    pre_t = torch.arange(t0, device=y.device, dtype=y.dtype).view(1, t0, 1)
    mean_t = (w * pre_t).sum(dim=1) / denom
    mean_y = (w * y[:, :t0, :]).sum(dim=1) / denom
    cov = (w * (pre_t - mean_t.unsqueeze(1)) * (y[:, :t0, :] - mean_y.unsqueeze(1))).sum(dim=1)
    var = (w * (pre_t - mean_t.unsqueeze(1)) ** 2).sum(dim=1)
    slope = torch.where(var > 1e-8, cov / torch.clamp(var, min=1e-12), torch.zeros_like(var))
    intercept = mean_y - slope * mean_t
    t_all = torch.arange(n_t, device=y.device, dtype=y.dtype).view(1, n_t, 1)
    return intercept.unsqueeze(1) + slope.unsqueeze(1) * t_all


def compute_evoked_kernel_aux_loss(
    y_pred_stim: torch.Tensor,
    y_pred_zero: torch.Tensor,
    y_true: torch.Tensor,
    obs_mask: torch.Tensor,
    stim: torch.Tensor,
    adjacency: torch.Tensor,
    pre_vols: int = 10,
    evoked_window_vols: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable pair-level 1-hop evoked kernel & temporal-derivative losses.

    Returns ``(kernel_loss, dynamics_loss)`` comparing paired counterfactual curves
    ``y_pred_stim - y_pred_zero`` against pre-stim detrended ``y_true - baseline``
    averaged within minibatch stimulus-signature groups on observed 1-hop targets.
    """
    t0 = int(pre_vols)
    t1 = min(y_true.shape[1], t0 + int(evoked_window_vols))
    zero = y_pred_stim.new_zeros(())
    if t1 <= t0 + 1:
        return zero, zero

    baseline = _prestim_baseline_torch(y_true, obs_mask, pre_vols)
    yt_ev = (y_true - baseline)[:, t0:t1, :]
    yp_ev = (y_pred_stim - y_pred_zero)[:, t0:t1, :]
    valid = obs_mask[:, t0:t1, :].bool()

    stim_target = stim.abs().sum(dim=1) > 0  # (B, N)
    hop1 = (stim_target.to(dtype=stim.dtype) @ adjacency > 0) & (~stim_target)

    # Group trials in minibatch by stimulated neuron signature
    sigs: dict[tuple[int, ...], list[int]] = {}
    st_cpu = stim_target.detach().cpu().numpy()
    for b in range(stim.shape[0]):
        s_tuple = tuple(np.flatnonzero(st_cpu[b]).tolist())
        if s_tuple:
            sigs.setdefault(s_tuple, []).append(b)

    k_losses, d_losses = [], []
    for grp in sigs.values():
        m_grp = valid[grp] & hop1[grp].unsqueeze(1)
        if not torch.any(m_grp):
            continue
        cnt = m_grp.sum(dim=0)  # (T_ev, N)
        ok_n = cnt.any(dim=0)
        if not torch.any(ok_n):
            continue
        cnt_k = torch.clamp(cnt[:, ok_n].to(dtype=yp_ev.dtype), min=1.0)
        m_k = m_grp[:, :, ok_n].to(dtype=yp_ev.dtype)
        k_true = (yt_ev[grp][:, :, ok_n] * m_k).sum(dim=0) / cnt_k
        k_pred = (yp_ev[grp][:, :, ok_n] * m_k).sum(dim=0) / cnt_k
        step_ok = cnt[:, ok_n] >= 1
        k_losses.append(F.smooth_l1_loss(k_pred[step_ok], k_true[step_ok], beta=0.1))
        diff_ok = step_ok[1:] & step_ok[:-1]
        if torch.any(diff_ok):
            dk_p = k_pred[1:] - k_pred[:-1]
            dk_t = k_true[1:] - k_true[:-1]
            d_losses.append(F.smooth_l1_loss(dk_p[diff_ok], dk_t[diff_ok], beta=0.05))

    kernel_loss = torch.stack(k_losses).mean() if k_losses else zero
    dyn_loss = torch.stack(d_losses).mean() if d_losses else zero
    return kernel_loss, dyn_loss


def _aux_kernel_step(
    model,
    aux_opt: torch.optim.Optimizer,
    stim: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
    adj_t: torch.Tensor,
    pre_vols: int,
    kernel_w: float,
    dyn_w: float,
    drift_gate: torch.Tensor | None = None,
) -> float:
    """Single optimization step on differentiable paired-counterfactual kernel/dynamics auxiliary loss."""
    aux_opt.zero_grad()
    B, n_t, _ = stim.shape
    has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
    has_ssm = getattr(model.decoder, "latent_ssm", None) is not None
    guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_pre or has_ssm:
        guide_kw["pre_stim_obs"] = y[:, :pre_vols, :]
        guide_kw["pre_stim_mask"] = mask[:, :pre_vols, :]
        guide_kw["drift_gate"] = drift_gate

    tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
    ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)

    stacked_kw = dict(
        ExternalInput=torch.cat([stim, torch.zeros_like(stim)], dim=0),
        mask=None,
        y=None,
        noise_scale=torch.cat([noise, noise], dim=0),
        sample_obs=False,
    )
    if has_pre or has_ssm:
        stacked_kw["pre_stim_obs"] = torch.cat([y[:, :pre_vols, :], y[:, :pre_vols, :]], dim=0)
        stacked_kw["pre_stim_mask"] = torch.cat([mask[:, :pre_vols, :], mask[:, :pre_vols, :]], dim=0)
        if drift_gate is not None:
            g_cat = torch.cat([drift_gate, drift_gate], dim=1 if drift_gate.dim() == 3 else 0) if drift_gate.dim() >= 2 else drift_gate
            stacked_kw["drift_gate"] = g_cat
    if has_ssm:
        z_dim = int(getattr(model.decoder.latent_ssm, "total_z_dim", model.decoder.latent_ssm.z_dim))
        eps = torch.randn((B, n_t + 1, z_dim), device=stim.device, dtype=stim.dtype)
        stacked_kw["ssm_eps"] = torch.cat([eps, eps], dim=0)

    out = pyro.condition(model, data=ps)(**stacked_kw)
    k_loss, d_loss = compute_evoked_kernel_aux_loss(
        out[:B], out[B:], y, mask, stim, adj_t, pre_vols=pre_vols
    )
    tot = float(kernel_w) * k_loss + float(dyn_w) * d_loss
    if tot.requires_grad:
        tot.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        aux_opt.step()
    return float(tot.detach().item())


class _WormIdSubset(Dataset):
    """Dataset wrapper attaching compact per-worm integer IDs to trial tuples."""

    def __init__(self, subset: Subset, worm_ids: list):
        self.subset = subset
        self.worm_ids = torch.as_tensor(worm_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        return (*item, self.worm_ids[idx])


def compute_multi_horizon_rollout_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    obs_mask: torch.Tensor,
    pre_vols: int = 10,
    horizons: tuple = (1, 2, 4, 8, 16, 32, 64),
) -> torch.Tensor:
    """Multi-horizon open-loop prediction & increment consistency loss across 0.5s-32s scales."""
    n_t = y_pred.shape[1]
    t0 = max(0, min(int(pre_vols), n_t - 1))
    m = obs_mask.to(dtype=y_pred.dtype, device=y_pred.device)
    total = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
    n_active = 0
    for h in horizons:
        step_h = int(h)
        if t0 + step_h >= n_t:
            continue
        t_end = min(n_t, t0 + step_h)
        m_w = m[:, t0:t_end, :]
        w_sum = m_w.sum()
        if w_sum.item() > 0:
            err_w = ((y_pred[:, t0:t_end, :] - y_true[:, t0:t_end, :]) ** 2) * m_w
            total = total + err_w.sum() / w_sum
            n_active += 1
        # Increment matching across lag h for post-stim trajectory dynamics
        if t0 + step_h < n_t:
            dp = y_pred[:, t0 + step_h :, :] - y_pred[:, t0 : n_t - step_h, :]
            dt_t = y_true[:, t0 + step_h :, :] - y_true[:, t0 : n_t - step_h, :]
            m_dp = m[:, t0 + step_h :, :] * m[:, t0 : n_t - step_h, :]
            dp_sum = m_dp.sum()
            if dp_sum.item() > 0:
                total = total + (((dp - dt_t) ** 2) * m_dp).sum() / dp_sum
                n_active += 1
    if n_active > 0:
        total = total / float(n_active)
    return total


def compute_latent_overshooting_loss(
    model,
    posterior_samples: dict,
    stim: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    pre_vols: int = 10,
    drift_gate=None,
    worm_id=None,
    z_prior_post: torch.Tensor = None,
) -> torch.Tensor:
    """Align multi-step open-loop prior SSM rollout with causal/smoothing variational posterior."""
    dec = model.decoder
    ssm = getattr(dec, "latent_ssm", None)
    if ssm is None:
        return torch.zeros((), device=stim.device, dtype=stim.dtype)
    if z_prior_post is None:
        bsz, n_t, _ = stim.shape
        z_dim = int(getattr(ssm, "total_z_dim", ssm.z_dim))
        eps0 = torch.zeros((bsz, n_t + 1, z_dim), device=stim.device, dtype=stim.dtype)
        m_kw = dict(
            stim_input=stim,
            voltage_clamp=None,
            dt_obs=float(dec.dt_obs),
            pre_stim_obs=y[:, :pre_vols, :],
            pre_stim_mask=mask[:, :pre_vols, :],
            drift_gate=drift_gate,
            return_trajectories=False,
            ssm_eps=eps0,
            worm_id=worm_id,
            return_latent_traj=True,
        )
        with dec._pyro_context, pyro.poutine.condition(data=posterior_samples):
            out = dec._rollout(**m_kw)
        z_prior_post = out[3]
    if z_prior_post is None or not getattr(ssm, "variational_traj", False):
        return torch.zeros((), device=stim.device, dtype=stim.dtype)
    t0 = max(0, min(int(pre_vols), y.shape[1]))
    if t0 >= y.shape[1]:
        return torch.zeros((), device=stim.device, dtype=stim.dtype)
    q_loc_seq, _ = ssm.encode_traj_posterior(
        y[:, t0:, :], mask[:, t0:, :], stim[:, t0:, :], allow_smoother=bool(ssm.use_smoother)
    )
    t_len = min(z_prior_post.shape[1], q_loc_seq.shape[1])
    if t_len == 0:
        return torch.zeros((), device=stim.device, dtype=stim.dtype)
    return F.mse_loss(z_prior_post[:, :t_len, :], q_loc_seq[:, :t_len, :].detach())


def _aux_rollout_overshoot_step(
    model,
    aux_opt,
    stim: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
    pre_vols: int,
    rollout_w: float,
    overshoot_w: float,
    drift_gate=None,
    worm_id=None,
) -> float:
    """Single optimization step on deterministic open-loop multi-horizon & overshooting loss."""
    aux_opt.zero_grad()
    B, n_t, _ = stim.shape
    has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
    ssm = getattr(model.decoder, "latent_ssm", None)
    has_ssm = ssm is not None
    guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_pre or has_ssm:
        guide_kw["pre_stim_obs"] = y[:, :pre_vols, :]
        guide_kw["pre_stim_mask"] = mask[:, :pre_vols, :]
        guide_kw["drift_gate"] = drift_gate
    tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
    ps = extract_and_expand_posterior_samples(tr, repeat_factor=1)
    eps0 = (
        torch.zeros((B, n_t + 1, int(getattr(ssm, "total_z_dim", ssm.z_dim))), device=stim.device, dtype=stim.dtype)
        if has_ssm
        else None
    )
    with model.decoder._pyro_context, pyro.poutine.condition(data=ps):
        roll_out = model.decoder._rollout(
            stim,
            None,
            float(model.decoder.dt_obs),
            pre_stim_obs=y[:, :pre_vols, :] if (has_pre or has_ssm) else None,
            pre_stim_mask=mask[:, :pre_vols, :] if (has_pre or has_ssm) else None,
            drift_gate=drift_gate if (has_pre or has_ssm) else None,
            prestim_z=ps.get("prestim_z"),
            return_trajectories=False,
            ssm_eps=eps0,
            ssm_z0_sample=ps.get("ssm_z0"),
            ssm_a0_sample=ps.get("ssm_a0"),
            worm_id=worm_id,
            return_latent_traj=has_ssm,
        )
    y_open = roll_out[0]
    z_post = roll_out[3] if has_ssm and len(roll_out) > 3 else None

    tot = torch.zeros((), device=stim.device, dtype=stim.dtype)
    if float(rollout_w) > 0.0:
        tot = tot + float(rollout_w) * compute_multi_horizon_rollout_loss(y_open, y, mask, pre_vols=pre_vols)
    if float(overshoot_w) > 0.0 and has_ssm:
        tot = tot + float(overshoot_w) * compute_latent_overshooting_loss(
            model, ps, stim, y, mask, pre_vols=pre_vols, drift_gate=drift_gate,
            worm_id=worm_id, z_prior_post=z_post,
        )
    if tot.requires_grad:
        tot.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        aux_opt.step()
    return float(tot.detach().item())


def _aux_combined_step(
    model,
    aux_opt: torch.optim.Optimizer,
    stim: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    noise: torch.Tensor,
    adj_t: torch.Tensor,
    pre_vols: int,
    kernel_w: float,
    dyn_w: float,
    rollout_w: float,
    overshoot_w: float,
    drift_gate=None,
    worm_id=None,
) -> float:
    """Single fused optimization step covering kernel, dynamics, multi-horizon rollout, and overshooting losses."""
    aux_opt.zero_grad()
    B, n_t, _ = stim.shape
    has_pre = bool(getattr(model.decoder, "use_prestim_encoder", False))
    ssm = getattr(model.decoder, "latent_ssm", None)
    has_ssm = ssm is not None
    guide_kw = dict(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
    if has_pre or has_ssm:
        guide_kw["pre_stim_obs"] = y[:, :pre_vols, :]
        guide_kw["pre_stim_mask"] = mask[:, :pre_vols, :]
        guide_kw["drift_gate"] = drift_gate

    tr = pyro.poutine.trace(model.guide).get_trace(**guide_kw)
    ps = extract_and_expand_posterior_samples(tr, repeat_factor=2)

    stim_2 = torch.cat([stim, torch.zeros_like(stim)], dim=0)
    pre_obs_2 = torch.cat([y[:, :pre_vols, :], y[:, :pre_vols, :]], dim=0) if (has_pre or has_ssm) else None
    pre_msk_2 = torch.cat([mask[:, :pre_vols, :], mask[:, :pre_vols, :]], dim=0) if (has_pre or has_ssm) else None
    drift_gate_2 = None
    if (has_pre or has_ssm) and drift_gate is not None:
        if drift_gate.dim() >= 2:
            drift_gate_2 = torch.cat([drift_gate, drift_gate], dim=1 if drift_gate.dim() == 3 else 0)
        else:
            drift_gate_2 = drift_gate
    worm_id_2 = torch.cat([worm_id, worm_id], dim=0) if worm_id is not None else None
    eps0 = (
        torch.zeros((2 * B, n_t + 1, int(getattr(ssm, "total_z_dim", ssm.z_dim))), device=stim.device, dtype=stim.dtype)
        if has_ssm
        else None
    )

    with model.decoder._pyro_context, pyro.poutine.condition(data=ps):
        roll_out = model.decoder._rollout(
            stim_2,
            None,
            float(model.decoder.dt_obs),
            pre_stim_obs=pre_obs_2,
            pre_stim_mask=pre_msk_2,
            drift_gate=drift_gate_2,
            prestim_z=ps.get("prestim_z"),
            return_trajectories=False,
            ssm_eps=eps0,
            ssm_z0_sample=ps.get("ssm_z0"),
            ssm_a0_sample=ps.get("ssm_a0"),
            worm_id=worm_id_2,
            return_latent_traj=has_ssm,
        )
    y_stim, y_zero = roll_out[0][:B], roll_out[0][B:]
    z_post_stim = roll_out[3][:B] if (has_ssm and len(roll_out) > 3 and roll_out[3] is not None) else None

    k_loss, d_loss = compute_evoked_kernel_aux_loss(
        y_stim, y_zero, y, mask, stim, adj_t, pre_vols=pre_vols
    )
    tot = float(kernel_w) * k_loss + float(dyn_w) * d_loss
    if float(rollout_w) > 0.0:
        tot = tot + float(rollout_w) * compute_multi_horizon_rollout_loss(y_stim, y, mask, pre_vols=pre_vols)
    if float(overshoot_w) > 0.0 and has_ssm:
        tot = tot + float(overshoot_w) * compute_latent_overshooting_loss(
            model, ps, stim, y, mask, pre_vols=pre_vols, drift_gate=drift_gate,
            worm_id=worm_id, z_prior_post=z_post_stim,
        )
    if tot.requires_grad:
        tot.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        aux_opt.step()
    return float(tot.detach().item())


def _validate_cache_provenance(provenance: dict, expected: dict, cache_path: str) -> None:
    """Fail fast if a dataset cache does not match the requested preprocessing contract."""
    if not provenance:
        raise ValueError(
            f"Dataset cache {cache_path!r} is missing provenance metadata; delete and rebuild it."
        )
    mismatches = []
    for k, v in expected.items():
        if provenance.get(k) != v:
            mismatches.append(f"{k}: cache={provenance.get(k)!r} vs requested={v!r}")
    if mismatches:
        raise ValueError(
            f"Dataset cache {cache_path!r} provenance mismatch ({'; '.join(mismatches)})."
        )


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
        ("guide.tau_ca_loc_unconstrained", "tau_ca_loc_unconstrained"),
        ("guide.tau_ca_scale_unconstrained", "tau_ca_scale_unconstrained"),
        ("guide.k_off_loc_unconstrained", "k_off_loc_unconstrained"),
        ("guide.k_off_scale_unconstrained", "k_off_scale_unconstrained"),
    ]
    expanded = []
    for key, attr in pairs:
        if key in state and state[key].numel() == 1:
            # Write the *unconstrained* PyroParam. Accessing ``guide.tau_ca_scale``
            # returns the constrained tensor from ExpTransform; filling that
            # temporary silently leaves the trainable parameter unchanged.
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


def _set_prestim_only_trainable(module, prestim_only: bool, train_obs_sigma: bool = True):
    """Freeze all non-prestim/non-latent-SSM parameters for Stage-2 fitting."""
    for name, param in module.named_parameters():
        if prestim_only:
            keep = (
                ("prestim_encoder" in name)
                or ("prestim_q" in name)
                or ("latent_ssm" in name)
                or ("U_chem" in name)
                or ("U_stim" in name)
                or ("U_gain" in name)
                or ("worm_b" in name)
                or (train_obs_sigma and "obs_sigma" in name)
            )
            param.requires_grad_(keep)
        else:
            param.requires_grad_(True)


def _set_extrasyn_only_trainable(module, extrasyn_only: bool, train_obs_sigma: bool = True):
    """Freeze wired connectome/stim/GCaMP parameters for Stage-1b extrasynaptic fitting."""
    for name, param in module.named_parameters():
        if extrasyn_only:
            keep = (
                ("extrasyn" in name)
                or ("tau_np" in name)
                or ("tau_ma" in name)
                or ("gamma_np_mut" in name)
                or ("gamma_ma_mut" in name)
                or ("gamma_wired_mut" in name)
                or (train_obs_sigma and "obs_sigma" in name)
            )
            param.requires_grad_(keep)
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
    parser.add_argument("--stim-loss-weight", default=1.0, type=float,
                        help="Relative likelihood weight on direct stimulus targets")
    parser.add_argument("--per-neuron-kinetics", action="store_true",
                        help="Learn per-neuron tau_ca and k_off (hierarchical around v3 global values)")
    parser.add_argument("--per-edge-release", action="store_true",
                        help="Learn per-edge V_rest/V_slope residuals on top of per-neuron release values")
    parser.add_argument("--per-neuron-stim-gain", action="store_true",
                        help="Learn a constrained stimulus gain per neuron")
    parser.add_argument("--wide-amp-priors", action="store_true",
                        help="Widen stim_gain, chemical g_max/gap spans, and dual-action ACh/Glu E_rev priors")
    parser.add_argument("--evoked-syn-priors", action="store_true",
                        help="Keep V_rest=-42 mV while sharpening V_slope (8 mV) and setting g_max median to 0.015 nS (std 0.95)")
    parser.add_argument("--direct-global-guide", action="store_true",
                        help="Use non-amortized PyroParam q(theta_global) and restrict amortization to q_phi(z | y_pre)")
    parser.add_argument("--use-extrasynaptic", action="store_true",
                        help="Enable neuropeptide and monoamine extrasynaptic pathway (edgelist_NP + MA)")
    parser.add_argument("--split-extrasynaptic", action="store_true",
                        help="Split edgelist_NP (neuropeptide) and edgelist_MA (monoamine) with separate slow time constants")
    parser.add_argument("--syn-filter", action="store_true",
                        help="Enable first-order chemical synaptic low-pass activation filter tau_syn")
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
    parser.add_argument("--freeze-non-prestim", action="store_true",
                        help="Stage-2 decoupled mode: freeze connectome/stim/GCaMP params and train only prestim_encoder (+ obs_sigma)")
    parser.add_argument("--freeze-wired-connectome", action="store_true",
                        help="Stage-1b decoupled mode: freeze wired connectome/stim/GCaMP and train only extrasyn_gain (+ obs_sigma)")
    parser.add_argument("--prestim-oscillatory", action="store_true",
                        help="Augment CausalPreStimEncoder with strictly causal damped oscillatory latent modes")
    parser.add_argument("--dimensionless-physiology", action="store_true",
                        help="Fix leak conductance scale and sample tau_m=C/G and normalized conductances kappa=g/G")
    parser.add_argument("--heterogeneous-leak-gauge", action="store_true",
                        help="Use global zero-mean log leak gauge (geom_mean(G)=1) preserving gap reciprocity")
    parser.add_argument("--latent-ssm", action="store_true",
                        help="Enable autonomous discrete-time stochastic LatentNeuralSSM residual")
    parser.add_argument("--latent-ssm-dim", default=16, type=int,
                        help="Latent state dimension for LatentNeuralSSM (default: 16)")
    parser.add_argument("--ssm-encoder", choices=["pooling", "gru"], default="pooling",
                        help="Pre-stimulus sequence encoder for LatentNeuralSSM")
    parser.add_argument("--ssm-split-latent", action="store_true",
                        help="Separate neural state z_neural (current only) from autonomous z_artifact (obs residual only)")
    parser.add_argument("--ssm-artifact-dim", default=8, type=int,
                        help="Latent dimension for autonomous z_artifact when --ssm-split-latent is enabled")
    parser.add_argument("--ssm-variational-traj", action="store_true",
                        help="Sample variational latent trajectory posterior q_phi(z_t | z_{t-1}, y_{<=t})")
    parser.add_argument("--ssm-smoother", action="store_true",
                        help="Use bidirectional smoothing posterior during training when --ssm-variational-traj is enabled")
    parser.add_argument("--ssm-state-observer", action="store_true",
                        help="Enable Phase 8A continuous causal physiological state assimilation over t < inject_t")
    parser.add_argument("--ssm-min-decay", default=0.0, type=float,
                        help="Minimum transition decay eigenvalue floor in LatentNeuralSSM (0 = disabled)")
    parser.add_argument("--rollout-loss-weight", default=0.0, type=float,
                        help="Multi-horizon open-loop prediction loss weight (0 disables)")
    parser.add_argument("--overshoot-loss-weight", default=0.0, type=float,
                        help="Open-loop vs smoothing latent overshooting loss weight (0 disables)")
    parser.add_argument("--worm-latent-dim", default=0, type=int,
                        help="Trainable generative worm latent factor dimension (0 disables)")
    parser.add_argument("--shuffle-disjoint", action="store_true",
                        help="When --shuffle-connectome is set, use degree-preserving 0-overlap disjoint rewiring")
    parser.add_argument("--history-vols", default=0, type=int,
                        help="Optional trailing pre-stim window passed to SSM encoder (0 = use full pre_vols)")
    parser.add_argument("--tau-np-prior", default=4.0, type=float,
                        help="Prior median (seconds) for split neuropeptide timescale tau_np")
    parser.add_argument("--tau-ma-prior", default=1.0, type=float,
                        help="Prior median (seconds) for split monoamine timescale tau_ma")
    parser.add_argument("--tau-mod-prior-std", default=0.5, type=float,
                        help="Log-normal prior std for tau_np and tau_ma")
    parser.add_argument("--use-model-adj-for-supervision", action="store_true",
                        help="Legacy behavior: use shuffled connectome for loss reweighting/aux/val instead of intact biological adjacency")
    parser.add_argument("--kernel-loss-weight", default=0.0, type=float,
                        help="Auxiliary pair-level 1-hop response kernel K_ij(t) loss weight (0 disables)")
    parser.add_argument("--dynamics-loss-weight", default=0.0, type=float,
                        help="Auxiliary pair-level 1-hop temporal derivative dK_ij/dt loss weight (0 disables)")
    parser.add_argument("--trial-split", action="store_true", help="Disable recording-level grouping in split")
    parser.add_argument("--seed", default=42, type=int, help="Training RNG seed")
    parser.add_argument("--split-seed", default=42, type=int, help="Fixed data split seed")
    parser.add_argument("--test-size", default=0.1, type=float,
                        help="Recording-grouped test fraction (default 0.1)")
    parser.add_argument("--shuffle-connectome", default=None, type=int,
                        help="Fixed seed for a connectivity-shuffle control (None = intact graph)")
    parser.add_argument("--split-file", default=None, type=str,
                        help="JSON with train/val/test index lists (nested CV fold); overrides test-size/val-fraction")
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
        expected_prov = {
            "source": "randi",
            "strain": str(args.strain),
            "norm_mode": str(args.norm_mode),
            "causal_preprocess": bool(args.causal_preprocess),
            "pre_vols": int(args.pre_vols),
            "post_vols": int(args.post_vols),
        }
        if args.dataset_cache and os.path.isfile(args.dataset_cache):
            dataset = RandiTrialDataset.load(args.dataset_cache)
            _validate_cache_provenance(dataset.provenance, expected_prov, args.dataset_cache)
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
        expected_prov = {
            "source": "toyoshima",
            "pre_vols": int(args.pre_vols),
            "post_vols": int(args.post_vols),
        }
        if args.dataset_cache and os.path.isfile(args.dataset_cache):
            dataset = RandiTrialDataset.load(args.dataset_cache)
            _validate_cache_provenance(dataset.provenance, expected_prov, args.dataset_cache)
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
        if args.split_file:
            with open(args.split_file, "r") as fh:
                split = json.load(fh)
            train_idx = [int(i) for i in split["train"]]
            val_idx = [int(i) for i in split.get("val", [])]
            test_idx = [int(i) for i in split["test"]]
            split_info = {
                "grouped": 1,
                "train_recordings": len({dataset.meta[i]["rec_id"] for i in train_idx + val_idx}),
                "test_recordings": len({dataset.meta[i]["rec_id"] for i in test_idx}),
                "split_file": args.split_file,
            }
            val_info = {"val_recordings": len({dataset.meta[i]["rec_id"] for i in val_idx})}
            train_core_idx = list(train_idx)
            train_ds = Subset(dataset, train_core_idx)
            test_ds = Subset(dataset, test_idx)
            val_ds = Subset(dataset, val_idx) if val_idx else None
            print(
                f"Dataset ready ({source_label}, split-file): train={len(train_ds)} trials, "
                f"val={0 if val_ds is None else len(val_ds)} trials, test={len(test_ds)} trials"
            )
        else:
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

    num_train_worms = 0
    if args.worm_latent_dim > 0 and args.source != "synthetic":
        rec_map = {}
        worm_ids_tr = []
        for idx_i in train_core_idx:
            rid = int(dataset.meta[idx_i]["rec_id"])
            if rid not in rec_map:
                rec_map[rid] = len(rec_map)
            worm_ids_tr.append(rec_map[rid])
        num_train_worms = len(rec_map)
        train_ds = _WormIdSubset(train_ds, worm_ids_tr)

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
    eval_bs = max(int(args.batch_size), 256)
    val_data = (
        DataLoader(
            val_ds,
            batch_size=min(eval_bs, len(val_ds)),
            pin_memory=pin,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_ds is not None
        else None
    )
    test_data = DataLoader(
        test_ds,
        batch_size=min(eval_bs, len(test_ds)),
        pin_memory=pin,
        shuffle=False,
        num_workers=args.num_workers,
    )

    eff_extrasyn = bool(args.use_extrasynaptic or args.split_extrasynaptic)
    leak_gauge = "zero_mean_log" if args.heterogeneous_leak_gauge else "unit"
    step_model = c302.readConnectome(
        "./data/CElegansNeuronTables.xls",
        use_extrasynaptic=eff_extrasyn,
        per_edge_release=args.per_edge_release,
        per_neuron_stim_gain=args.per_neuron_stim_gain,
        wide_amp_priors=args.wide_amp_priors,
        evoked_syn_priors=args.evoked_syn_priors,
        dimensionless=args.dimensionless_physiology,
        leak_gauge=leak_gauge,
        split_extrasynaptic=args.split_extrasynaptic,
        use_syn_filter=args.syn_filter,
        tau_np_prior=args.tau_np_prior,
        tau_ma_prior=args.tau_ma_prior,
        tau_mod_prior_std=args.tau_mod_prior_std,
        shuffle_seed=args.shuffle_connectome,
        shuffle_disjoint=args.shuffle_disjoint,
    )
    if args.shuffle_connectome is not None and not args.use_model_adj_for_supervision:
        intact_ref = c302.readConnectome(
            "./data/CElegansNeuronTables.xls",
            use_extrasynaptic=True,
            split_extrasynaptic=True,
        )
        supervision_adj = intact_ref.synapse.adjacency_matrix(intact_ref.neuronSize)
        sup_np_adj = intact_ref.synapse.np_adjacency_matrix(intact_ref.neuronSize)
        sup_ma_adj = intact_ref.synapse.ma_adjacency_matrix(intact_ref.neuronSize)
    else:
        supervision_adj = step_model.synapse.adjacency_matrix(step_model.neuronSize)
        sup_np_adj = step_model.synapse.np_adjacency_matrix(step_model.neuronSize)
        sup_ma_adj = step_model.synapse.ma_adjacency_matrix(step_model.neuronSize)
    adj_t = torch.as_tensor(supervision_adj, dtype=torch.float32, device=device)
    rec_model = c302.RecurrentNematode(
        step_model,
        dt_obs=args.dt_obs,
        substeps=args.substeps,
        use_prestim_encoder=args.use_prestim_encoder,
        pre_vols=args.pre_vols,
        prestim_rank=args.prestim_rank,
        variational_prestim=bool(args.direct_global_guide and args.use_prestim_encoder),
        prestim_oscillatory=args.prestim_oscillatory,
        per_neuron_kinetics=args.per_neuron_kinetics,
        per_edge_release=args.per_edge_release,
        stim_actuator=args.stim_actuator,
        stim_tau_rise=args.stim_tau_rise,
        stim_tau_decay=args.stim_tau_decay,
        use_latent_ssm=args.latent_ssm,
        latent_ssm_dim=args.latent_ssm_dim,
        ssm_encoder=args.ssm_encoder,
        ssm_split_latent=args.ssm_split_latent,
        ssm_artifact_dim=args.ssm_artifact_dim,
        ssm_variational_traj=args.ssm_variational_traj,
        ssm_smoother=args.ssm_smoother,
        history_vols=args.history_vols,
        ssm_state_observer=args.ssm_state_observer,
        ssm_min_decay=args.ssm_min_decay,
        worm_latent_dim=args.worm_latent_dim,
        num_worms=num_train_worms,
    )
    model = vae.VAElegans(
        rec_model,
        rec_model.model.Neuron.neuronSize,
        len(rec_model.model.synapse.Wicks_SRC),
        len(rec_model.model.synapse.Gap_Junction_SRC),
        direct_global_guide=args.direct_global_guide,
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
    if args.freeze_non_prestim and args.freeze_wired_connectome:
        raise ValueError("--freeze-non-prestim and --freeze-wired-connectome cannot be combined")
    if args.freeze_non_prestim:
        if not (args.use_prestim_encoder or args.latent_ssm):
            raise ValueError("--freeze-non-prestim requires --use-prestim-encoder or --latent-ssm")
        _set_prestim_only_trainable(model, True)
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"Stage-2 prestim-only mode: trainable={n_train_p} params, frozen={n_frozen_p} params",
            flush=True,
        )
    elif args.freeze_wired_connectome:
        if not eff_extrasyn:
            raise ValueError("--freeze-wired-connectome requires --use-extrasynaptic or --split-extrasynaptic")
        _set_extrasyn_only_trainable(model, True)
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"Stage-1b extrasyn-only mode: trainable={n_train_p} params, frozen={n_frozen_p} params",
            flush=True,
        )
    summary(model)

    optim = pyro.optim.ClippedAdam({"lr": args.lr, "clip_norm": 10.0})
    svi = SVI(model, model.guide, optim, Trace_ELBO())
    use_kernel_aux = (float(args.kernel_loss_weight) > 0.0) or (float(args.dynamics_loss_weight) > 0.0)
    use_rollout_aux = (float(args.rollout_loss_weight) > 0.0) or (float(args.overshoot_loss_weight) > 0.0)
    use_aux = use_kernel_aux or use_rollout_aux
    aux_opt = (
        torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.5)
        if use_aux
        else None
    )

    n_train_trials = len(train_ds)
    freeze_epochs = max(0, int(args.freeze_nonkinetics_epochs))
    edge_freeze_epochs = max(0, int(args.freeze_nonedge_epochs))
    for epoch in range(args.total_epochs):
        if args.freeze_non_prestim:
            _set_prestim_only_trainable(model, True)
        elif args.freeze_wired_connectome:
            _set_extrasyn_only_trainable(model, True)
        elif freeze_epochs:
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
        aux_loss_sum = 0.0
        n_batches_ep = 0
        scaled_obs_t = torch.zeros((), device=device, dtype=torch.float32)
        for batch in train_data:
            stim, y, mask, noise = batch[:4]
            worm_id = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
            stim = stim.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            noise = noise.to(device, non_blocking=True)
            bsz = max(1, int(stim.shape[0]))
            scale_factor = float(n_train_trials) / float(bsz) if n_train_trials > bsz else 1.0
            drift_gate = compute_drift_gate(stim, adj_t, args.drift_gate) if args.use_prestim_encoder else None
            noise = reweight_hop1_noise(
                noise, stim, adj_t, args.hop1_loss_weight, args.stim_loss_weight
            )
            svi_kw = dict(
                ExternalInput=stim,
                mask=mask,
                y=y,
                noise_scale=noise,
                dataset_size=n_train_trials,
                drift_gate=drift_gate,
            )
            if worm_id is not None:
                svi_kw["worm_id"] = worm_id
            loss += svi.step(**svi_kw)
            if use_kernel_aux and use_rollout_aux and aux_opt is not None:
                aux_loss_sum += _aux_combined_step(
                    model, aux_opt, stim, y, mask, noise, adj_t, args.pre_vols,
                    args.kernel_loss_weight, args.dynamics_loss_weight,
                    args.rollout_loss_weight, args.overshoot_loss_weight,
                    drift_gate=drift_gate, worm_id=worm_id,
                )
            elif use_kernel_aux and aux_opt is not None:
                aux_loss_sum += _aux_kernel_step(
                    model, aux_opt, stim, y, mask, noise, adj_t,
                    args.pre_vols, args.kernel_loss_weight, args.dynamics_loss_weight, drift_gate=drift_gate,
                )
            elif use_rollout_aux and aux_opt is not None:
                aux_loss_sum += _aux_rollout_overshoot_step(
                    model, aux_opt, stim, y, mask, noise,
                    args.pre_vols, args.rollout_loss_weight, args.overshoot_loss_weight,
                    drift_gate=drift_gate, worm_id=worm_id,
                )
            n_batches_ep += 1
            scaled_obs_t = scaled_obs_t + mask.sum().to(dtype=torch.float32) * scale_factor
        scaled_obs = float(scaled_obs_t.item())
        avg_loss = loss / max(1.0, scaled_obs)
        avg_aux = aux_loss_sum / max(1, n_batches_ep)
        dt_ep = time.time() - t0
        aux_str = f" | aux={avg_aux:.5f}" if use_aux else ""
        print(f"Epoch {epoch:03d} | elbo/obs={avg_loss:.4f} | total_loss={loss:.2f}{aux_str} | dt={dt_ep:.2f}s")
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
                    adjacency=supervision_adj,
                    drift_gate_mode=gate_mode,
                    np_adjacency=sup_np_adj,
                    ma_adjacency=sup_ma_adj,
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
                    adjacency=supervision_adj,
                    drift_gate_mode=gate_mode,
                    np_adjacency=sup_np_adj,
                    ma_adjacency=sup_ma_adj,
                )
                print(f"  Test | {format_eval_summary(test_metrics)}")
            _save_snapshot(os.path.join(args.snapshot_dir, f"vae_epoch_{epoch+1}.pt"), model, model.guide, epoch + 1)
            metrics_path = os.path.join(args.snapshot_dir, f"metrics_epoch_{epoch+1}.json")
            with open(metrics_path, "w") as mf:
                json.dump({"epoch": epoch + 1, "elbo_per_obs": avg_loss, "aux_loss": avg_aux if use_aux else 0.0,
                           "val": val_metrics, "test": test_metrics}, mf, indent=2)


if __name__ == "__main__":
    main()
