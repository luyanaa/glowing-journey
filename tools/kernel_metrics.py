"""Time-resolved stimulus-pair response kernel metrics K_ij(t)."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    if a_f.size < 2 or b_f.size < 2:
        return 0.0
    da = a_f - a_f.mean()
    db = b_f - b_f.mean()
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(da * db) / denom)


def _trapz(y: np.ndarray, dx: float) -> float:
    if y.size < 2:
        return float(y.sum() * dx)
    return float(0.5 * dx * np.sum(y[:-1] + y[1:]))


def compute_pair_kernel_metrics(
    yt_ev: np.ndarray,
    yp_ev: np.ndarray,
    stim_target: np.ndarray,
    neu_sel: np.ndarray,
    valid_ev: np.ndarray,
    t0: int,
    t_ev_end: int,
    dt_obs: float = 0.5,
    min_trials: int = 3,
    min_common_steps: int = 4,
    strong_thresh: float = 0.08,
    prefix: str = "kernel_hop1",
) -> Dict[str, float]:
    """Evaluate trial-averaged time-resolved response kernels K_ij(t).

    For each stimulus signature i observed across >= min_trials trials and each
    selected target neuron j observed in >= min_trials trials within [t0:t_ev_end],
    builds K_ij^true(t) and K_ij^pred(t) over valid time steps and evaluates
    curve correlation/RMSE/R2, AUC, peak amplitude/sign, peak latency, and
    spectral power correlation.
    """
    n_trials = yt_ev.shape[0]
    t0 = int(t0)
    t_ev_end = int(t_ev_end)
    if t_ev_end <= t0:
        return {}

    stim_groups: Dict[Tuple[int, ...], List[int]] = {}
    for b in range(n_trials):
        sig = tuple(np.flatnonzero(stim_target[b]).tolist())
        if sig:
            stim_groups.setdefault(sig, []).append(b)

    true_curves: List[np.ndarray] = []
    pred_curves: List[np.ndarray] = []
    pair_curve_rs: List[float] = []
    aucs_t: List[float] = []
    aucs_p: List[float] = []
    peaks_t: List[float] = []
    peaks_p: List[float] = []
    lat_t: List[float] = []
    lat_p: List[float] = []
    fft_t: List[np.ndarray] = []
    fft_p: List[np.ndarray] = []

    for grp_idx in stim_groups.values():
        if len(grp_idx) < int(min_trials):
            continue
        m_grp = valid_ev[grp_idx, t0:t_ev_end, :] & neu_sel[grp_idx, None, :]
        obs_trials = m_grp.any(axis=1).sum(axis=0)
        keep_cols = np.flatnonzero(obs_trials >= int(min_trials))
        if keep_cols.size == 0:
            continue

        yt_sub = yt_ev[grp_idx, t0:t_ev_end, :][:, :, keep_cols]
        yp_sub = yp_ev[grp_idx, t0:t_ev_end, :][:, :, keep_cols]
        m_sub = m_grp[:, :, keep_cols]
        cnt_t = m_sub.sum(axis=0)  # (T_ev, K)

        for k_idx in range(keep_cols.size):
            step_ok = cnt_t[:, k_idx] >= 1
            if int(step_ok.sum()) < int(min_common_steps):
                continue
            denom = np.maximum(cnt_t[step_ok, k_idx], 1).astype(np.float64)
            k_true = (yt_sub[:, step_ok, k_idx] * m_sub[:, step_ok, k_idx]).sum(axis=0) / denom
            k_pred = (yp_sub[:, step_ok, k_idx] * m_sub[:, step_ok, k_idx]).sum(axis=0) / denom

            true_curves.append(k_true)
            pred_curves.append(k_pred)
            if np.std(k_true) > 1e-12 and np.std(k_pred) > 1e-12:
                pair_curve_rs.append(_safe_pearson_r(k_true, k_pred))

            aucs_t.append(_trapz(k_true, float(dt_obs)))
            aucs_p.append(_trapz(k_pred, float(dt_obs)))

            pk_idx_t = int(np.argmax(np.abs(k_true)))
            pk_idx_p = int(np.argmax(np.abs(k_pred)))
            pk_t = float(k_true[pk_idx_t])
            pk_p = float(k_pred[pk_idx_p])
            peaks_t.append(pk_t)
            peaks_p.append(pk_p)

            if abs(pk_t) > float(strong_thresh):
                lat_t.append(float(pk_idx_t) * float(dt_obs))
                lat_p.append(float(pk_idx_p) * float(dt_obs))

            if k_true.size >= 4:
                spec_t = np.abs(np.fft.rfft(k_true - k_true.mean()))[1:]
                spec_p = np.abs(np.fft.rfft(k_pred - k_pred.mean()))[1:]
                if spec_t.size >= 1:
                    fft_t.append(spec_t)
                    fft_p.append(spec_p)

    if not true_curves:
        return {}

    cat_t = np.concatenate(true_curves)
    cat_p = np.concatenate(pred_curves)
    pk_t_arr = np.asarray(peaks_t, dtype=np.float64)
    pk_p_arr = np.asarray(peaks_p, dtype=np.float64)
    auc_t_arr = np.asarray(aucs_t, dtype=np.float64)
    auc_p_arr = np.asarray(aucs_p, dtype=np.float64)

    out: Dict[str, float] = {
        f"{prefix}_n_pairs": int(len(true_curves)),
        f"{prefix}_n_points": int(cat_t.size),
        f"{prefix}_r": _safe_pearson_r(cat_t, cat_p),
        f"{prefix}_r2": float(r2_score(cat_t, cat_p)) if cat_t.size >= 2 else float("nan"),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(cat_t, cat_p))),
        f"{prefix}_pair_curve_r_mean": float(np.mean(pair_curve_rs)) if pair_curve_rs else 0.0,
        f"{prefix}_auc_r": _safe_pearson_r(auc_t_arr, auc_p_arr),
        f"{prefix}_auc_mae": float(mean_absolute_error(auc_t_arr, auc_p_arr)),
        f"{prefix}_peak_amp_r": _safe_pearson_r(pk_t_arr, pk_p_arr),
        f"{prefix}_peak_amp_mae": float(mean_absolute_error(pk_t_arr, pk_p_arr)),
    }
    pt_std = float(np.std(pk_t_arr))
    out[f"{prefix}_amp_ratio"] = float(np.std(pk_p_arr) / pt_std) if pt_std > 1e-12 else 0.0

    strong = np.abs(pk_t_arr) > float(strong_thresh)
    if np.any(strong):
        out[f"{prefix}_sign_acc"] = float(np.mean(np.sign(pk_t_arr[strong]) == np.sign(pk_p_arr[strong])))
    if lat_t:
        lat_t_arr = np.asarray(lat_t, dtype=np.float64)
        lat_p_arr = np.asarray(lat_p, dtype=np.float64)
        out[f"{prefix}_peak_latency_mae_s"] = float(mean_absolute_error(lat_t_arr, lat_p_arr))
        out[f"{prefix}_peak_latency_r"] = _safe_pearson_r(lat_t_arr, lat_p_arr)
    if fft_t:
        fft_t_arr = np.concatenate(fft_t)
        fft_p_arr = np.concatenate(fft_p)
        out[f"{prefix}_fft_r"] = _safe_pearson_r(fft_t_arr, fft_p_arr)
    return out
