import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy.ndimage import median_filter, percentile_filter, uniform_filter1d
from scipy.signal import savgol_filter
import torch
from torch.utils.data import Dataset


DEFAULT_ACONNECTOME_IDS_PATH = "./pumpprobe/pumpprobe/aconnectome_ids.txt"
DEFAULT_TIME_STEP = 0.05


def load_canonical_neurons(
    aconnectome_path: str = DEFAULT_ACONNECTOME_IDS_PATH,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load the 300 canonical C. elegans neuron names and name->index map."""
    neuron_names = np.loadtxt(aconnectome_path, dtype=str)[:, 1]
    name_to_idx = {str(name): idx for idx, name in enumerate(neuron_names)}
    return neuron_names, name_to_idx


def canonicalize_neuron_label(label: str) -> str:
    """Normalize raw label text to match canonical connectome IDs."""
    cleaned = label.strip()
    if cleaned == "AWCOFF":
        return "AWCL"
    if cleaned == "AWCON":
        return "AWCR"
    return cleaned


def _interpolate_nonfinite_1d(trace: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Linearly interpolate non-finite samples for sliding-window filtering only."""
    out = trace.copy()
    idx = np.arange(len(trace))
    out[~finite] = np.interp(idx[~finite], idx[finite], trace[finite])
    return out


def _estimate_hf_sigma(traces: np.ndarray) -> np.ndarray:
    """Estimate frame-to-frame noise std robustly from first-difference MAD."""
    if traces.shape[0] < 2:
        return np.ones(traces.shape[1], dtype=np.float64)
    diffs = np.diff(traces, axis=0)
    diff_med = np.median(diffs, axis=0)
    mad_diff = np.median(np.abs(diffs - diff_med), axis=0)
    return 1.4826 * mad_diff / np.sqrt(2.0)


def _causal_preprocess_gcamp(
    raw: np.ndarray,
    finite_mask: np.ndarray,
    valid_cols: np.ndarray,
    dt: float,
    hampel_window_sec: float,
    hampel_n_sigma: float,
    baseline_window_sec: float,
    baseline_quantile: float,
    norm_mode: str,
    min_f0: float,
    min_f0_snr: float,
    drop_invalid_f0: bool,
    smooth_window: Optional[int],
    clip_abs: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past-only Hampel/baseline/normalization for causal pre-stim inputs."""
    import pandas as pd

    n_t = raw.shape[0]
    n_cols = raw.shape[1]
    valid_mask = finite_mask.copy()
    normalized = np.zeros_like(raw, dtype=np.float64)
    noise_scale = np.ones(n_cols, dtype=np.float64)

    hampel_win = max(5, int(round(hampel_window_sec / max(dt, 1e-3))))
    if hampel_win % 2 == 0:
        hampel_win += 1
    hampel_win = min(hampel_win, n_t)
    base_win = max(5, int(round(baseline_window_sec / max(dt, 1e-3))))
    if base_win % 2 == 0:
        base_win += 1
    base_win = min(base_win, n_t)
    smooth_win = max(3, min(base_win // 5, 15))

    for k, col_idx in enumerate(valid_cols):
        raw_col = raw[:, col_idx]
        # Forward-fill only: leading gaps become 0 and remain masked.
        s = pd.Series(raw_col).ffill().fillna(0.0)
        med = s.rolling(hampel_win, min_periods=1).median()
        absdev = (s - med).abs().rolling(hampel_win, min_periods=1).median()
        local_scale = np.maximum(1.4826 * absdev.to_numpy(), 1e-6)
        spikes = np.abs(s.to_numpy() - med.to_numpy()) > hampel_n_sigma * local_scale
        despiked = np.where(spikes, med.to_numpy(), s.to_numpy())

        baseline = (
            pd.Series(despiked)
            .rolling(base_win, min_periods=1)
            .quantile(float(baseline_quantile))
            .to_numpy()
        )
        baseline = pd.Series(baseline).rolling(smooth_win, min_periods=1).mean().to_numpy()
        delta_f = despiked - baseline

        if smooth_window is not None and smooth_window >= 3:
            win = int(smooth_window)
            if win % 2 == 0:
                win += 1
            delta_f = pd.Series(delta_f).rolling(win, min_periods=1).mean().to_numpy()

        # Noise/F0 validity must not depend on the later recording.  Use the
        # first ~5 s only; the full baseline window may be the whole recording
        # for short sessions, which would reintroduce future leakage.
        noise_win = max(5, int(round(5.0 / max(dt, 1e-3))))
        first_n = min(n_t, noise_win)
        sigma_hf = (
            float(_estimate_hf_sigma(despiked[:first_n, None])[0])
            if first_n >= 2
            else 1.0
        )
        if norm_mode in ("dff_safe", "dff_strict"):
            scale_t = pd.Series(baseline).rolling(base_win, min_periods=1).median().to_numpy()
            scale_t = np.maximum(scale_t, min_f0)
            f0 = float(np.median(baseline[:first_n]))
            f0_ok = f0 > max(float(min_f0), float(min_f0_snr) * sigma_hf)
            if (norm_mode == "dff_strict" or drop_invalid_f0) and not f0_ok:
                # Invalid baseline: keep the column masked instead of
                # silently normalizing by an unstable F0.
                valid_mask[:, col_idx] = False
                continue
        elif norm_mode == "robust_z":
            # Robust scale from the causal delta_f distribution, not from the
            # absolute first difference (which is spike-sensitive).
            delta_series = pd.Series(delta_f)
            delta_med = delta_series.rolling(base_win, min_periods=1).median()
            delta_mad = (delta_series - delta_med).abs().rolling(base_win, min_periods=1).median()
            scale_t = np.maximum(1.4826 * delta_mad.to_numpy(), 1e-6)
            if sigma_hf <= 1e-6:
                valid_mask[:, col_idx] = False
                continue
        else:
            raise ValueError(f"Unknown norm_mode={norm_mode!r}")

        out = delta_f / scale_t
        if clip_abs is not None:
            out = np.clip(out, -clip_abs, clip_abs)
        mask_col = finite_mask[:, col_idx]
        out[~mask_col] = 0.0
        normalized[:, col_idx] = out
        valid_mask[:, col_idx] = mask_col
        # Estimate noise from the first causal baseline window, never from the
        # full recording or from scale_t[-1] (both would use future frames).
        scale_ref = float(np.median(scale_t[:first_n]))
        noise_scale[col_idx] = float(
            np.clip(sigma_hf / max(scale_ref, 1e-6), 0.01, 5.0)
        )

    return normalized.astype(np.float32), valid_mask, noise_scale.astype(np.float32)


def preprocess_gcamp_recording(
    raw_data: np.ndarray,
    dt: float = 0.5,
    hampel_window_sec: float = 5.5,
    hampel_n_sigma: float = 4.5,
    baseline_window_sec: float = 60.0,
    baseline_quantile: float = 0.20,
    norm_mode: str = "dff_safe",
    min_f0: float = 1.0,
    min_f0_snr: float = 3.0,
    drop_invalid_f0: bool = True,
    smooth_window: Optional[int] = None,
    smooth_polyorder: int = 2,
    min_finite_frac: float = 0.5,
    clip_abs: Optional[float] = 10.0,
    causal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Despike, detrend, and normalize a raw GCaMP recording matrix (T, N).

    Unlike the legacy ``wormdatamodel.Signal`` pipeline, this:
      1. Preserves raw ``NaN``/``Inf`` locations in ``valid_mask`` instead of
         silently treating interpolated gaps as observations.
      2. Uses a bidirectional rolling Hampel filter to remove both positive
         flash artifacts and negative tracking dropouts.
      3. Removes slow photobleaching / camera-offset drift via a smoothed
         rolling low-quantile baseline.
      4. Handles negative or near-zero baseline traces safely without sign
         inversion or division blowups.

    Returns:
        norm_data: (T, N) float32 preprocessed GCaMP signal (invalid entries 0).
        valid_mask: (T, N) bool mask of trustworthy observations.
        noise_scale: (N,) float32 estimated noise std per neuron in output units.
    """
    raw = np.asarray(raw_data, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    n_t, n_cols = raw.shape

    finite_mask = np.isfinite(raw)
    valid_mask = finite_mask.copy()
    norm_data = np.zeros_like(raw, dtype=np.float64)
    noise_scale = np.ones(n_cols, dtype=np.float64)

    min_samples = max(10, int(np.ceil(min_finite_frac * n_t)))
    col_has_data = np.sum(finite_mask, axis=0) >= min_samples
    valid_cols = np.where(col_has_data)[0]
    if valid_cols.size == 0 or n_t < 5:
        return (
            norm_data.astype(np.float32),
            np.zeros_like(valid_mask, dtype=bool),
            noise_scale.astype(np.float32),
        )

    work = np.zeros((n_t, valid_cols.size), dtype=np.float64)
    for k, col_idx in enumerate(valid_cols):
        f_col = finite_mask[:, col_idx]
        if np.all(f_col):
            work[:, k] = raw[:, col_idx]
        else:
            work[:, k] = _interpolate_nonfinite_1d(raw[:, col_idx], f_col)

    if causal:
        return _causal_preprocess_gcamp(
            raw,
            finite_mask,
            valid_cols,
            dt,
            hampel_window_sec,
            hampel_n_sigma,
            baseline_window_sec,
            baseline_quantile,
            norm_mode,
            min_f0,
            min_f0_snr,
            drop_invalid_f0,
            smooth_window,
            clip_abs,
        )

    # 1. Bidirectional Hampel despiking
    hampel_win = max(5, int(round(hampel_window_sec / max(dt, 1e-3))))
    if hampel_win % 2 == 0:
        hampel_win += 1
    hampel_win = min(hampel_win, n_t if n_t % 2 == 1 else max(3, n_t - 1))

    sigma_hf_raw = _estimate_hf_sigma(work)
    med = median_filter(work, size=(hampel_win, 1), mode="nearest")
    mad = median_filter(np.abs(work - med), size=(hampel_win, 1), mode="nearest")
    local_scale = np.maximum(1.4826 * mad, np.maximum(0.5 * sigma_hf_raw[None, :], 1e-6))
    spikes = np.abs(work - med) > (hampel_n_sigma * local_scale)
    despiked = np.where(spikes, med, work)

    # 2. Slow photobleaching / offset drift removal via rolling low quantile
    base_win = max(11, int(round(baseline_window_sec / max(dt, 1e-3))))
    if base_win % 2 == 0:
        base_win += 1
    base_win = min(base_win, n_t if n_t % 2 == 1 else max(5, n_t - 1))
    q_stride = max(1, min(base_win // 12, 10)) if (n_t >= 200 and base_win >= 31) else 1
    if q_stride > 1 and n_t > 2 * q_stride:
        pad_l = base_win // 2
        pad_r = base_win - 1 - pad_l
        xp = np.pad(despiked, ((pad_l, pad_r), (0, 0)), mode="edge")
        win_view = np.lib.stride_tricks.sliding_window_view(xp, base_win, axis=0)
        q_idx = np.arange(0, n_t, q_stride)
        if q_idx[-1] != n_t - 1:
            q_idx = np.append(q_idx, n_t - 1)
        qs = np.quantile(win_view[q_idx], float(baseline_quantile), axis=-1)
        pos = np.interp(np.arange(n_t), q_idx, np.arange(len(q_idx)))
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, len(q_idx) - 1)
        w = (pos - i0)[:, None]
        baseline = (1.0 - w) * qs[i0] + w * qs[i1]
    else:
        baseline = percentile_filter(
            despiked,
            percentile=float(baseline_quantile * 100.0),
            size=(base_win, 1),
            mode="nearest",
        )
    smooth_base_win = max(3, min(base_win // 5, 15))
    baseline = uniform_filter1d(baseline, size=smooth_base_win, axis=0, mode="nearest")
    delta_f = despiked - baseline

    # 3. Optional zero-phase smoothing (disabled by default to preserve rise times)
    if smooth_window is not None and smooth_window >= 5:
        win = int(smooth_window)
        if win % 2 == 0:
            win += 1
        win = min(win, n_t if n_t % 2 == 1 else n_t - 1)
        if win > smooth_polyorder:
            delta_f = savgol_filter(
                delta_f,
                window_length=win,
                polyorder=smooth_polyorder,
                axis=0,
                mode="interp",
            )

    # 4. High-frequency noise & safe normalization
    sigma_hf = np.maximum(_estimate_hf_sigma(despiked), 1e-6)
    f0 = np.median(baseline, axis=0)
    f0_ok = f0 > np.maximum(min_f0, min_f0_snr * sigma_hf)

    if norm_mode == "robust_z":
        scale = sigma_hf
        col_keep = sigma_hf > 1e-6
    elif norm_mode in ("dff_safe", "dff_strict"):
        scale = np.where(f0_ok, f0, np.maximum(5.0 * sigma_hf, min_f0))
        if norm_mode == "dff_strict" or drop_invalid_f0:
            col_keep = f0_ok & (sigma_hf > 1e-6)
        else:
            col_keep = sigma_hf > 1e-6
    else:
        raise ValueError(f"Unknown norm_mode={norm_mode!r}; expected 'dff_safe', 'dff_strict', or 'robust_z'.")

    normalized = delta_f / scale[None, :]
    if clip_abs is not None:
        normalized = np.clip(normalized, -clip_abs, clip_abs)
    est_noise = np.clip(sigma_hf / scale, 0.01, 5.0)

    valid_mask[:, :] = False
    for k, col_idx in enumerate(valid_cols):
        if not col_keep[k]:
            continue
        col_mask = finite_mask[:, col_idx]
        norm_data[col_mask, col_idx] = normalized[col_mask, k]
        valid_mask[:, col_idx] = col_mask
        noise_scale[col_idx] = est_noise[k]

    return (
        norm_data.astype(np.float32),
        valid_mask,
        noise_scale.astype(np.float32),
    )


def _convolve_causal_fallback(rf: np.ndarray, stim: np.ndarray, dt: float) -> np.ndarray:
    """Pure-NumPy causal convolution fallback matching dt * sum_{s<=t} rf[t-s] stim[s]."""
    n_t = len(stim)
    return np.convolve(rf, stim)[:n_t] * dt


def _import_wormfunconn():
    """Import wormfunconn lazily, installing a NumPy convolution fallback if C++ .so is absent."""
    import sys
    import types

    pkg_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wormfunconn")
    if os.path.isdir(os.path.join(pkg_parent, "wormfunconn")) and pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    existing = sys.modules.get("wormfunconn")
    if existing is not None and not hasattr(existing, "FunctionalAtlas"):
        sys.modules.pop("wormfunconn", None)

    try:
        import wormfunconn as wfc
        return wfc
    except ImportError:
        import importlib.util

        integ_path = os.path.join(pkg_parent, "wormfunconn", "integration.py")
        spec = importlib.util.spec_from_file_location("wormfunconn_integration_py", integ_path)
        integ_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integ_mod)
        integral_py = integ_mod.integral

        def _conv_py(a: np.ndarray, b: np.ndarray, dt: float, k: int = 8) -> np.ndarray:
            return _convolve_causal_fallback(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), float(dt))

        def _conv1_py(a: np.ndarray, b: np.ndarray, n: int, dt: float, k: int = 8) -> float:
            return float(np.sum(a[: n + 1] * b[: n + 1][::-1]) * dt)

        sys.modules.pop("wormfunconn", None)
        sys.modules.setdefault(
            "wormfunconn._integration",
            types.SimpleNamespace(integral=integral_py),
        )
        sys.modules.setdefault(
            "wormfunconn._convolution",
            types.SimpleNamespace(convolution=_conv_py, convolution1=_conv1_py),
        )
        import wormfunconn as wfc
        return wfc


class responseGenerator:
    """Generate synthetic responses from the wormfunconn FunctionalAtlas."""

    def __init__(
        self,
        folder: str,
        strain: str = "wild-type",
        stim_type: str = "rectangular",
        dt: float = DEFAULT_TIME_STEP,
        size: int = 1000,
        stim_duration: float = 1.0,
        aconnectome_path: str = DEFAULT_ACONNECTOME_IDS_PATH,
    ):
        wfc = _import_wormfunconn()
        self._funatlas = wfc.FunctionalAtlas.from_file(folder, strain)
        self.respList = self._funatlas.get_neuron_ids()
        self.stimList = self._funatlas.get_neuron_ids(stim=True)
        self.dt = float(dt)
        self.stim_type = stim_type
        self.size = int(size)
        self.stim_duration = float(stim_duration)
        self.aconnectome_path = aconnectome_path

    def step(self, stim_neu_id: str, resp_neu_ids: Sequence[str], points: int):
        if self.stim_type in ("rectangular", "delta"):
            kwargs = {"duration": self.stim_duration} if self.stim_type == "rectangular" else {}
        else:
            kwargs = {}
        stim = self._funatlas.get_standard_stimulus(
            points, dt=self.dt, stim_type=self.stim_type, **kwargs
        )
        resp, labels, confidences, _ = self._funatlas.get_responses(
            stim, self.dt, stim_neu_id, resp_neu_ids=resp_neu_ids
        )
        return stim, resp, labels

    def Dataset(self, return_mask: bool = False):
        neuron_names, name_to_idx = load_canonical_neurons(self.aconnectome_path)
        n_neurons = len(neuron_names)

        def _build_one(i: int):
            stim_neu_id = str(self.stimList[i])
            _stim, _resp, _labels = self.step(stim_neu_id, self.respList, self.size)
            stim = np.zeros((n_neurons, self.size), dtype=np.float32)
            resp = np.zeros((n_neurons, self.size), dtype=np.float32)
            mask = np.zeros((n_neurons, self.size), dtype=bool)

            if stim_neu_id in name_to_idx:
                s_idx = name_to_idx[stim_neu_id]
                stim[s_idx] = _stim
                resp[s_idx] = _stim
                mask[s_idx] = True

            for index, lbl in enumerate(_labels):
                lbl_str = canonicalize_neuron_label(str(lbl))
                if lbl_str in name_to_idx:
                    r_idx = name_to_idx[lbl_str]
                    resp[r_idx] = _resp[index]
                    mask[r_idx] = True
            return stim, resp, mask

        result = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(_build_one)(i) for i in range(len(self.stimList))
        )
        stims = np.stack([r[0] for r in result], axis=0)
        resps = np.stack([r[1] for r in result], axis=0)
        masks = np.stack([r[2] for r in result], axis=0)

        x = torch.from_numpy(stims).swapaxes(1, 2).contiguous()
        y = torch.from_numpy(resps).swapaxes(1, 2).contiguous()
        m = torch.from_numpy(masks).swapaxes(1, 2).contiguous()
        if return_mask:
            return x, y, m
        return x, y


class PsuedoDataset(Dataset):
    """Simple tensor tuple dataset (supports optional obs_mask and noise_scale)."""

    def __init__(self, x1, x2, y, mask=None, noise_scale=None) -> None:
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.mask = mask
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        items = [self.x1[index], self.x2[index], self.y[index]]
        if self.mask is not None:
            items.append(self.mask[index])
        if self.noise_scale is not None:
            items.append(self.noise_scale[index])
        return tuple(items)


class RandiTrialDataset(Dataset):
    """Trial-segmented, preprocessed Randi et al. dataset aligned to 300 canonical neurons."""

    def __init__(
        self,
        stim_inputs: torch.Tensor,
        y: torch.Tensor,
        obs_mask: torch.Tensor,
        noise_scale: torch.Tensor,
        dts: torch.Tensor,
        meta: List[dict],
    ):
        self.stim_inputs = stim_inputs
        self.y = y
        self.obs_mask = obs_mask
        self.noise_scale = noise_scale
        self.dts = dts
        self.meta = meta

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.stim_inputs[idx],
            self.y[idx],
            self.obs_mask[idx],
            self.noise_scale[idx],
        )

    def save(self, path: str) -> None:
        """Save preprocessed trial dataset compactly as compressed NPZ."""
        import json

        stim_nz = torch.nonzero(self.stim_inputs, as_tuple=False).cpu().numpy().astype(np.int32)
        stim_vals = self.stim_inputs[stim_nz[:, 0], stim_nz[:, 1], stim_nz[:, 2]].cpu().numpy().astype(np.float32)
        np.savez_compressed(
            path,
            stim_shape=np.array(self.stim_inputs.shape, dtype=np.int64),
            stim_nz=stim_nz,
            stim_vals=stim_vals,
            y=self.y.cpu().numpy().astype(np.float32),
            obs_mask=self.obs_mask.cpu().numpy().astype(bool),
            noise_scale=self.noise_scale.cpu().numpy().astype(np.float32),
            dts=self.dts.cpu().numpy().astype(np.float32),
            meta_json=np.array([json.dumps(self.meta)]),
        )

    @classmethod
    def load(cls, path: str) -> "RandiTrialDataset":
        """Load preprocessed trial dataset saved with ``RandiTrialDataset.save``."""
        import json

        with np.load(path, allow_pickle=False) as d:
            stim_shape = tuple(int(x) for x in d["stim_shape"])
            stim = torch.zeros(stim_shape, dtype=torch.float32)
            stim_nz = d["stim_nz"]
            if len(stim_nz) > 0:
                stim[stim_nz[:, 0], stim_nz[:, 1], stim_nz[:, 2]] = torch.from_numpy(d["stim_vals"])
            return cls(
                stim_inputs=stim,
                y=torch.from_numpy(d["y"].astype(np.float32)),
                obs_mask=torch.from_numpy(d["obs_mask"].astype(bool)),
                noise_scale=torch.from_numpy(d["noise_scale"].astype(np.float32)),
                dts=torch.from_numpy(d["dts"].astype(np.float32)),
                meta=json.loads(str(d["meta_json"][0])),
            )


class AtlasLoader(Dataset):
    """Load, denoise, normalize, and align exported Randi et al. recordings.

    Supports both pointing directly at a directory containing ``*_gcamp.txt``
    files (such as ``.../exported_data``) and pointing at a parent folder
    containing ``exported_data/`` or ``exported_data_unc31/``.
    """

    def __init__(
        self,
        path: str,
        strain: str = "wild-type",
        device: str = "cpu",
        aconnectome_path: str = DEFAULT_ACONNECTOME_IDS_PATH,
        norm_mode: str = "dff_safe",
        drop_invalid_f0: bool = True,
        smooth_window: Optional[int] = None,
        max_recordings: Optional[int] = None,
        causal_preprocess: bool = False,
    ):
        self.device = device
        self.causal_preprocess = bool(causal_preprocess)
        self.norm_mode = norm_mode
        self.neuron_names, self.name_to_idx = load_canonical_neurons(aconnectome_path)
        self.n_neurons = len(self.neuron_names)

        resolved_dir = self._resolve_data_dir(path, strain)
        self.data_dir = resolved_dir

        rec_ids = self._discover_recording_ids(resolved_dir)
        if max_recordings is not None:
            rec_ids = rec_ids[:max_recordings]
        if not rec_ids:
            raise FileNotFoundError(f"No '*_gcamp.txt' recordings found in {resolved_dir}")

        self.recordings: List[dict] = []
        self.x = []
        self.y = []

        for rec_id in rec_ids:
            rec = self._load_single_recording(
                resolved_dir,
                rec_id,
                norm_mode=norm_mode,
                drop_invalid_f0=drop_invalid_f0,
                smooth_window=smooth_window,
                causal=self.causal_preprocess,
            )
            self.recordings.append(rec)
            self.x.append((rec["stim_neurons"], rec["stim_volume_i"]))
            self.y.append((rec["t"], rec["y"], rec["obs_mask"], rec["noise_scale"]))

    @staticmethod
    def _resolve_data_dir(path: str, strain: str) -> str:
        if os.path.isdir(path) and any(f.endswith("_gcamp.txt") for f in os.listdir(path)):
            return path
        if strain == "wild-type":
            candidate = os.path.join(path, "exported_data")
        elif strain in ("unc-31", "unc31"):
            candidate = os.path.join(path, "exported_data_unc31")
        else:
            raise NotImplementedError(f"Unsupported strain {strain!r}")
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Data directory not found: {candidate}")
        return candidate

    @staticmethod
    def _discover_recording_ids(data_dir: str) -> List[int]:
        ids = []
        for fname in os.listdir(data_dir):
            match = re.match(r"^(\d+)_gcamp\.txt$", fname)
            if match:
                ids.append(int(match.group(1)))
        return sorted(ids)

    @staticmethod
    def _load_txt_matrix(path: str) -> np.ndarray:
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    ncols = len(stripped.split())
                    break
            else:
                return np.empty((0, 0), dtype=np.float64)
        try:
            arr = np.fromfile(path, sep=" ", dtype=np.float64)
            if arr.size % ncols == 0 and arr.size > 0:
                return arr.reshape(-1, ncols)
        except Exception:
            pass
        arr = np.loadtxt(path)
        return arr[:, None] if arr.ndim == 1 else arr

    def _load_single_recording(
        self,
        data_dir: str,
        rec_id: int,
        norm_mode: str,
        drop_invalid_f0: bool,
        smooth_window: Optional[int],
        causal: bool = False,
    ) -> dict:
        prefix = os.path.join(data_dir, f"{rec_id}_")
        raw_gcamp = self._load_txt_matrix(prefix + "gcamp.txt")
        t = np.atleast_1d(np.loadtxt(prefix + "t.txt")).astype(np.float64)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.5

        stim_neu = np.atleast_1d(np.loadtxt(prefix + "stim_neurons.txt", dtype=np.int64))
        stim_vol_i = np.atleast_1d(np.loadtxt(prefix + "stim_volume_i.txt", dtype=np.int64))

        with open(prefix + "labels.txt", "r") as f:
            raw_labels = [canonicalize_neuron_label(line) for line in f]

        n_t = raw_gcamp.shape[0]
        y_canon = np.zeros((n_t, self.n_neurons), dtype=np.float32)
        obs_mask = np.zeros((n_t, self.n_neurons), dtype=bool)
        noise_scale = np.ones(self.n_neurons, dtype=np.float32)

        # Only preprocess columns that map onto canonical connectome neurons;
        # preprocess_gcamp_recording operates independently per column.
        mapped_orig_cols = []
        mapped_canon_idxs = []
        col_to_canon: Dict[int, int] = {}
        for col_idx, lbl in enumerate(raw_labels):
            if col_idx >= raw_gcamp.shape[1]:
                break
            if lbl in self.name_to_idx:
                c_idx = self.name_to_idx[lbl]
                col_to_canon[col_idx] = c_idx
                mapped_orig_cols.append(col_idx)
                mapped_canon_idxs.append(c_idx)

        if mapped_orig_cols:
            norm_gcamp, valid_mask, col_noise = preprocess_gcamp_recording(
                raw_gcamp[:, mapped_orig_cols],
                dt=dt,
                norm_mode=norm_mode,
                drop_invalid_f0=drop_invalid_f0,
                smooth_window=smooth_window,
                causal=causal,
            )
            for sub_idx, c_idx in enumerate(mapped_canon_idxs):
                # Only overwrite if new column has more valid frames
                if np.sum(valid_mask[:, sub_idx]) > np.sum(obs_mask[:, c_idx]):
                    y_canon[:, c_idx] = norm_gcamp[:, sub_idx]
                    obs_mask[:, c_idx] = valid_mask[:, sub_idx]
                    noise_scale[c_idx] = col_noise[sub_idx]

        # Clean stimulation events: drop negative sentinel targets, out-of-range
        # volume indices, and trailing non-monotonic wrap-arounds (e.g. rec 33).
        clean_events = []
        last_vol = -1
        for ev_idx, (s_col, vol) in enumerate(zip(stim_neu.tolist(), stim_vol_i.tolist())):
            if vol < 0 or vol >= n_t:
                continue
            if vol <= last_vol:
                continue
            last_vol = vol
            s_canon = col_to_canon.get(int(s_col), -1) if s_col >= 0 else -1
            clean_events.append((ev_idx, int(vol), int(s_col), int(s_canon)))

        return {
            "rec_id": rec_id,
            "t": torch.from_numpy(t.astype(np.float32)),
            "dt": dt,
            "y": torch.from_numpy(y_canon),
            "obs_mask": torch.from_numpy(obs_mask),
            "noise_scale": torch.from_numpy(noise_scale),
            "stim_neurons": torch.from_numpy(stim_neu),
            "stim_volume_i": torch.from_numpy(stim_vol_i),
            "clean_events": clean_events,
            "n_mapped_neurons": len(col_to_canon),
        }

    def build_trial_dataset(
        self,
        pre_vols: int = 10,
        post_vols: int = 50,
        stim_duration_s: float = 0.5,
        min_observed_neurons: int = 5,
        rebaseline_on_pre: bool = True,
        max_pre_abs: float = 5.0,
        max_pre_dev: Optional[float] = 2.0,
        trial_clip_abs: Optional[float] = 10.0,
    ) -> RandiTrialDataset:
        """Slice aligned (T_trial, 300) optogenetic perturbation trials."""
        trial_len = pre_vols + post_vols
        stim_list = []
        y_list = []
        mask_list = []
        noise_list = []
        dt_list = []
        meta_list = []

        for rec in self.recordings:
            y_full = rec["y"].numpy()
            mask_full = rec["obs_mask"].numpy()
            noise_vec = rec["noise_scale"].numpy()
            dt = rec["dt"]
            n_t = y_full.shape[0]
            pulse_steps = max(1, int(round(stim_duration_s / max(dt, 1e-3))))
            events = rec["clean_events"]

            for ev_pos, (ev_idx, vol, _, s_canon) in enumerate(events):
                if s_canon < 0:
                    continue
                v_start = vol - pre_vols
                v_end = vol + post_vols
                if v_start < 0 or v_end > n_t:
                    continue

                y_seg = y_full[v_start:v_end].copy()
                m_seg = mask_full[v_start:v_end].copy()
                stim_seg = np.zeros((trial_len, self.n_neurons), dtype=np.float32)

                # Encode primary stimulus and any additional mapped stimuli inside window;
                # mask out frames starting at any unmapped/failed stimulus inside window.
                for _, o_vol, _, o_canon in events:
                    if v_start <= o_vol < v_end:
                        rel_t = o_vol - v_start
                        if o_canon >= 0:
                            end_pulse = min(trial_len, rel_t + pulse_steps)
                            stim_seg[rel_t:end_pulse, o_canon] = 1.0
                        elif o_vol > vol:
                            m_seg[rel_t:, :] = False

                # Check target neuron itself or enough downstream neurons are observed
                if int(np.sum(np.any(m_seg, axis=0))) < min_observed_neurons:
                    continue

                if rebaseline_on_pre and pre_vols >= 3:
                    # Subtract per-trial pre-stimulus median on observed neurons to
                    # remove residual slow trial-to-trial offset before stimulation.
                    # Reject neurons whose pre-stimulus median itself is an extreme artifact
                    # and re-clip after subtraction so outliers cannot double to +/- 2*clip_abs.
                    for c in np.where(np.any(m_seg[:pre_vols], axis=0))[0]:
                        pre_valid = y_seg[:pre_vols, c][m_seg[:pre_vols, c]]
                        if pre_valid.size >= 3:
                            pre_med = float(np.median(pre_valid))
                            pre_max_dev = float(np.max(np.abs(pre_valid - pre_med)))
                            if abs(pre_med) > max_pre_abs or (
                                max_pre_dev is not None and pre_max_dev > max_pre_dev
                            ):
                                m_seg[:, c] = False
                                y_seg[:, c] = 0.0
                                continue
                            y_seg[:, c] -= pre_med
                            if trial_clip_abs is not None:
                                y_seg[:, c] = np.clip(y_seg[:, c], -trial_clip_abs, trial_clip_abs)
                            y_seg[~m_seg[:, c], c] = 0.0

                if int(np.sum(np.any(m_seg, axis=0))) < min_observed_neurons:
                    continue

                stim_list.append(torch.from_numpy(stim_seg))
                y_list.append(torch.from_numpy(y_seg))
                mask_list.append(torch.from_numpy(m_seg))
                noise_list.append(torch.from_numpy(noise_vec.copy()))
                dt_list.append(dt)
                meta_list.append(
                    {
                        "rec_id": rec["rec_id"],
                        "event_idx": ev_idx,
                        "stim_vol": vol,
                        "stim_neuron": str(self.neuron_names[s_canon]),
                        "stim_neuron_idx": s_canon,
                    }
                )

        if not y_list:
            raise RuntimeError("No valid trials extracted from dataset.")

        return RandiTrialDataset(
            stim_inputs=torch.stack(stim_list, dim=0),
            y=torch.stack(y_list, dim=0),
            obs_mask=torch.stack(mask_list, dim=0),
            noise_scale=torch.stack(noise_list, dim=0),
            dts=torch.tensor(dt_list, dtype=torch.float32),
            meta=meta_list,
        )

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class WormTensor(AtlasLoader):
    """Backward-compatible alias for directory-based GCaMP loading."""

    def __init__(self, path: str, device: str = "cpu", **kwargs):
        super().__init__(path=path, strain="wild-type", device=device, **kwargs)


class ToyoshimaLoader:
    """Load Toyoshima et al. (2022) NaCl whole-brain GCaMP data as a second dataset.

    The export contains ``N_ratio.csv`` (time x neurons), ``N_uniqNames.csv``
    (column labels), and ``stimulation_timing.xlsx`` (NaCl switching times).
    The loader resamples each recording to ``target_fs`` (default 2 Hz),
    maps available neuron IDs onto the 300-neuron canonical space, and cuts
    trials around each NaCl onset.

    The NaCl stimulus is a sensory perturbation, not a targeted optogenetic
    input.  ``stim_targets`` is therefore an explicit operational assumption;
    the default list is the chemosensory set ASH/AWC/ASE, and it can be
    overridden by the caller.

    The protocol alternates 50→25 mM and 25→50 mM steps every 30 s.  The
    loader encodes the two directions separately as signed, sustained
    commands on the measured ON/OFF sensory classes rather than collapsing
    both into one positive pulse.  The raw ratio columns are z-scored; they
    are multiplied by ``observation_scale`` (default 0.18, the Randi ΔF/F0
    scale) before being fed to the ΔF/F0-calibrated model/encoder.
    """

    # Measured 0–10 s means from the five audited recordings: ASER/ASH/AWC
    # prefer the 50→25 mM down-step, ASEL prefers the 25→50 mM up-step.
    # Values are (down-step sign, up-step sign).
    DEFAULT_POLARITY_SIGNS = {
        "ASER": (+1.0, -1.0),
        "ASEL": (-1.0, +1.0),
        "ASHR": (+1.0, -1.0),
        "ASHL": (+1.0, -1.0),
        "AWCR": (+1.0, -1.0),
        "AWCL": (+1.0, -1.0),
    }

    def __init__(
        self,
        path: str,
        aconnectome_path: str = DEFAULT_ACONNECTOME_IDS_PATH,
        target_fs: float = 2.0,
        stim_targets: Sequence[str] = ("ASH", "AWC", "ASE"),
        max_recordings: Optional[int] = None,
        observation_scale: float = 0.18,
        polarity_signs: Optional[dict] = None,
    ):
        import pandas as pd

        self.data_dir = path
        self.target_fs = float(target_fs)
        self.observation_scale = float(observation_scale)
        self.polarity_signs = dict(self.DEFAULT_POLARITY_SIGNS)
        if polarity_signs is not None:
            self.polarity_signs.update(polarity_signs)
        self.neuron_names, self.name_to_idx = load_canonical_neurons(aconnectome_path)
        self.n_neurons = len(self.neuron_names)
        target_indices = []
        for n in stim_targets:
            cname = canonicalize_neuron_label(n)
            if cname in self.name_to_idx:
                target_indices.append(self.name_to_idx[cname])
                continue
            # Toyoshima labels are often bilateral classes (ASH, AWC, ASE),
            # whereas the canonical connectome uses ASHL/ASHR etc.
            for suffix in ("L", "R"):
                split_name = cname + suffix
                if split_name in self.name_to_idx:
                    target_indices.append(self.name_to_idx[split_name])
        self.stim_targets = sorted(set(target_indices))
        if not self.stim_targets:
            raise ValueError("No Toyoshima stimulus target maps to the canonical 300-neuron space")

        timing_path = os.path.join(path, "stimulation_timing.xlsx")
        if not os.path.isfile(timing_path):
            raise FileNotFoundError(f"Toyoshima stimulation timing not found: {timing_path}")
        timing = pd.read_excel(timing_path, header=1)
        timing = timing.dropna(subset=["#sample"])
        self.timing = timing

        records = []
        for _, row in timing.iterrows():
            sample = int(row["#sample"])
            ratio_path = os.path.join(path, f"{sample}_ratio.csv")
            names_path = os.path.join(path, f"{sample}_uniqNames.csv")
            if not (os.path.isfile(ratio_path) and os.path.isfile(names_path)):
                continue
            rec = self._load_recording(sample, ratio_path, names_path, row)
            if rec is not None:
                records.append(rec)
            if max_recordings is not None and len(records) >= int(max_recordings):
                break
        if not records:
            raise FileNotFoundError(f"No complete Toyoshima ratio/name pairs found in {path}")
        self.recordings = records

    def _load_recording(self, sample, ratio_path, names_path, timing_row):
        import pandas as pd

        raw = pd.read_csv(ratio_path, header=None).to_numpy(dtype=np.float64)
        labels = pd.read_csv(names_path, header=None)[0].astype(str).tolist()
        if raw.ndim != 2 or raw.shape[1] != len(labels):
            return None
        fs = float(timing_row["frames/sec"])
        if not np.isfinite(fs) or fs <= 0:
            return None

        # Resample to the common trial clock.  Interpolation is applied per
        # column before mapping so unmapped neurons remain masked.
        # The export is already z-scored (std ≈ 1).  Put it on the Randi
        # ΔF/F0 scale before interpolation and noise estimation.
        raw = raw * self.observation_scale
        n_native = raw.shape[0]
        t_native = np.arange(n_native, dtype=np.float64) / fs
        t_grid = np.arange(0.0, t_native[-1] + 1e-9, 1.0 / self.target_fs)
        y = np.zeros((len(t_grid), self.n_neurons), dtype=np.float32)
        mask = np.zeros((len(t_grid), self.n_neurons), dtype=bool)
        noise = np.ones(self.n_neurons, dtype=np.float32)

        mapped = []
        for col, label in enumerate(labels):
            cname = canonicalize_neuron_label(label.strip())
            if cname not in self.name_to_idx:
                continue
            cidx = self.name_to_idx[cname]
            series = raw[:, col]
            finite = np.isfinite(series)
            if finite.sum() < 5:
                continue
            if not np.all(finite):
                series = np.interp(np.arange(n_native), np.where(finite)[0], series[finite])
            y[:, cidx] = np.interp(t_grid, t_native, series).astype(np.float32)
            mask[:, cidx] = True
            noise[cidx] = float(
                np.clip(np.std(np.diff(series)) / np.sqrt(2.0), 0.01, 5.0)
            )
            mapped.append(cidx)
        if len(mapped) < 5:
            return None

        first = float(timing_row["frame number of first stimuli (NaCl concenctration change from 50 mM to 25mM)"])
        interval = float(timing_row["every N frames NaCl concentration change"])
        cycles = int(round(float(timing_row["number of stimulation cycles"])))
        onsets = []
        for k in range(max(0, cycles)):
            t_sec = (first + k * interval) / fs
            idx = int(round(t_sec * self.target_fs))
            if 0 <= idx < len(t_grid):
                onsets.append(idx)
        if not onsets:
            return None
        return {
            "rec_id": int(sample),
            "name": str(timing_row.get("name", sample)),
            "y": y,
            "mask": mask,
            "noise_scale": noise,
            "dt": 1.0 / self.target_fs,
            "onsets": sorted(set(onsets)),
            "interval_s": interval / fs if fs > 0 else 30.0,
            "mapped_neurons": len(mapped),
        }

    def build_trial_dataset(
        self,
        pre_vols: int = 10,
        post_vols: int = 60,
        stim_duration_s: float = 30.0,
        min_observed_neurons: int = 5,
        rebaseline_on_pre: bool = True,
        max_pre_abs: float = 5.0,
        trial_clip_abs: Optional[float] = 10.0,
    ) -> RandiTrialDataset:
        """Cut NaCl-onset trials with signed, sustained ON/OFF commands.

        ``ev_idx`` is even for the 50→25 mM down-step and odd for the
        25→50 mM up-step.  The step duration is capped by the inter-onset
        interval, so a 30 s protocol does not leak into the next trial.
        """
        trial_len = int(pre_vols) + int(post_vols)
        dt = 1.0 / self.target_fs
        stim_list, y_list, mask_list, noise_list, dt_list, meta_list = [], [], [], [], [], []

        for rec in self.recordings:
            y_full = rec["y"]
            mask_full = rec["mask"]
            n_t = y_full.shape[0]
            for ev_idx, onset in enumerate(rec["onsets"]):
                v_start = int(onset) - int(pre_vols)
                v_end = int(onset) + int(post_vols)
                if v_start < 0 or v_end > n_t:
                    continue
                y_seg = y_full[v_start:v_end].copy()
                m_seg = mask_full[v_start:v_end].copy()
                stim_seg = np.zeros((trial_len, self.n_neurons), dtype=np.float32)
                rel = int(pre_vols)
                downstep = (ev_idx % 2 == 0)
                step_duration_s = min(
                    float(stim_duration_s), float(rec.get("interval_s", stim_duration_s))
                )
                pulse_steps = max(1, int(round(step_duration_s / dt)))
                pulse_steps = min(pulse_steps, trial_len - rel)
                for target in self.stim_targets:
                    target_name = self.neuron_names[target]
                    signs = self.polarity_signs.get(target_name, (1.0, 1.0))
                    sign = signs[0] if downstep else signs[1]
                    stim_seg[rel:rel + pulse_steps, target] = sign

                if rebaseline_on_pre and pre_vols >= 3:
                    for c in np.where(np.any(m_seg[:pre_vols], axis=0))[0]:
                        pre = y_seg[:pre_vols, c][m_seg[:pre_vols, c]]
                        if pre.size < 3:
                            continue
                        med = float(np.median(pre))
                        if abs(med) > max_pre_abs:
                            m_seg[:, c] = False
                            y_seg[:, c] = 0.0
                            continue
                        y_seg[:, c] -= med
                        if trial_clip_abs is not None:
                            y_seg[:, c] = np.clip(y_seg[:, c], -trial_clip_abs, trial_clip_abs)
                        y_seg[~m_seg[:, c], c] = 0.0

                if int(np.sum(np.any(m_seg, axis=0))) < int(min_observed_neurons):
                    continue

                stim_list.append(torch.from_numpy(stim_seg))
                y_list.append(torch.from_numpy(y_seg.astype(np.float32)))
                mask_list.append(torch.from_numpy(m_seg.astype(bool)))
                noise_list.append(torch.from_numpy(rec["noise_scale"].astype(np.float32)))
                dt_list.append(dt)
                meta_list.append({
                    "rec_id": int(rec["rec_id"]),
                    "event_idx": int(ev_idx),
                    "stim_neuron": "NaCl",
                    "stim_neuron_idx": int(self.stim_targets[0]),
                    "polarity": "down" if downstep else "up",
                    "polarity_value": 1 if downstep else -1,
                    "step_duration_s": float(step_duration_s),
                    "source": str(rec["name"]),
                })

        if not y_list:
            raise ValueError("No Toyoshima trials survived slicing/min-observation filters")
        return RandiTrialDataset(
            stim_inputs=torch.stack(stim_list),
            y=torch.stack(y_list),
            obs_mask=torch.stack(mask_list),
            noise_scale=torch.stack(noise_list),
            dts=torch.tensor(dt_list, dtype=torch.float32),
            meta=meta_list,
        )


def resolve_default_data_path() -> str:
    """Resolve Randi exported_data directory via env var or standard local/Colab fallbacks."""
    env_path = os.environ.get("RANDI_DATA_DIR")
    if env_path:
        return env_path
    candidates = [
        "/content/drive/MyDrive/exported_data",
        "./data/exported_data",
        "/Users/yanlu/Google Drive/マイドライブ/exported_data",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return "./data/exported_data"


def nested_grouped_splits(
    dataset: RandiTrialDataset,
    n_splits: int = 5,
    val_size: float = 0.2,
    random_state: int = 42,
) -> List[Dict[str, List[int]]]:
    """Recording-grouped outer CV with a grouped validation split inside each train fold."""
    from sklearn.model_selection import GroupKFold

    groups = np.array([int(dataset.meta[i]["rec_id"]) for i in range(len(dataset))], dtype=np.int64)
    n_splits = int(min(max(2, n_splits), len(np.unique(groups))))
    outer = GroupKFold(n_splits=n_splits)
    folds = []
    for tr_pos, te_pos in outer.split(np.arange(len(dataset)), groups=groups):
        core, val, _ = split_train_validation_indices(
            dataset, tr_pos.tolist(), val_size=val_size,
            random_state=random_state, group_by_recording=True,
        )
        folds.append({"train": core, "val": val, "test": te_pos.tolist()})
    return folds


def resolve_default_toyoshima_path() -> str:
    """Resolve the Toyoshima cleandata_smoothened2 directory."""
    env_path = os.environ.get("TOYOSHIMA_DATA_DIR")
    if env_path:
        return env_path
    candidates = [
        "/content/drive/MyDrive/cleandata_smoothened2",
        "./data/cleandata_smoothened2",
        "/Users/yanlu/Google Drive/マイドライブ/cleandata_smoothened2",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return "./data/cleandata_smoothened2"


def split_trial_dataset(
    dataset: RandiTrialDataset,
    test_size: float = 0.1,
    random_state: int = 42,
    group_by_recording: bool = True,
    max_trials: Optional[int] = None,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Split trials into train/test subsets grouped by recording ID to avoid worm leakage."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    n_total = len(dataset) if max_trials is None else min(len(dataset), int(max_trials))
    indices = np.arange(n_total)
    groups = np.array([int(dataset.meta[i]["rec_id"]) for i in range(n_total)], dtype=np.int64)
    unique_recs = np.unique(groups)

    if group_by_recording and len(unique_recs) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_pos, te_pos = next(splitter.split(indices, groups=groups))
        train_idx = indices[tr_pos].tolist()
        test_idx = indices[te_pos].tolist()
        info = {
            "grouped": 1,
            "train_recordings": int(len(np.unique(groups[tr_pos]))),
            "test_recordings": int(len(np.unique(groups[te_pos]))),
        }
        return train_idx, test_idx, info

    train_idx, test_idx = train_test_split(
        indices.tolist(), test_size=test_size, random_state=random_state
    )
    info = {
        "grouped": 0,
        "train_recordings": int(len(unique_recs)),
        "test_recordings": int(len(unique_recs)),
    }
    return train_idx, test_idx, info

def split_train_validation_indices(
    dataset: RandiTrialDataset,
    train_idx: Sequence[int],
    val_size: float = 0.15,
    random_state: int = 42,
    group_by_recording: bool = True,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Split an existing train index list into train-core/validation by recording.

    Validation trials are removed from training, so validation metrics are not
    in-sample.  Recording-level grouping prevents same-worm leakage.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    indices = np.asarray(list(train_idx), dtype=np.int64)
    if len(indices) == 0:
        return [], [], {"val_recordings": 0, "train_recordings": 0}
    groups = np.array([int(dataset.meta[int(i)]["rec_id"]) for i in indices], dtype=np.int64)
    unique_recs = np.unique(groups)

    if group_by_recording and len(unique_recs) >= 2 and 0.0 < val_size < 1.0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        tr_pos, va_pos = next(splitter.split(indices, groups=groups))
        core = indices[tr_pos].tolist()
        val = indices[va_pos].tolist()
        info = {
            "train_recordings": int(len(np.unique(groups[tr_pos]))),
            "val_recordings": int(len(np.unique(groups[va_pos]))),
        }
        return core, val, info

    core, val = train_test_split(
        indices.tolist(), test_size=val_size, random_state=random_state
    )
    info = {
        "train_recordings": int(len(unique_recs)),
        "val_recordings": int(len(unique_recs)),
    }
    return core, val, info
