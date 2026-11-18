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
    clip_abs: Optional[float] = 20.0,
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
    ):
        self.device = device
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
                    for c in np.where(np.any(m_seg[:pre_vols], axis=0))[0]:
                        pre_valid = y_seg[:pre_vols, c][m_seg[:pre_vols, c]]
                        if pre_valid.size >= 3:
                            y_seg[:, c] -= np.median(pre_valid)
                            y_seg[~m_seg[:, c], c] = 0.0

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
