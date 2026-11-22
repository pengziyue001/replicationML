"""Out-of-sample R², size-split R², and Diebold-Mariano tests."""

from __future__ import annotations

import numpy as np


def r2_oos(y: np.ndarray, yhat: np.ndarray) -> float:
    residual = y - yhat
    denom = float(np.dot(y, y))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.dot(residual, residual)) / denom


def size_masks(slices, me: np.ndarray, n: int, cutoff: int = 1000):
    top = np.zeros(n, dtype=bool)
    bottom = np.zeros(n, dtype=bool)
    for _, start, end in slices:
        values = me[start:end]
        valid = np.flatnonzero(np.isfinite(values))
        if valid.size == 0:
            continue
        order = valid[np.argsort(values[valid], kind="mergesort")]
        k = min(cutoff, order.size)
        bottom[start + order[:k]] = True
        top[start + order[-k:]] = True
    return top, bottom


def newey_west_t(series: np.ndarray) -> float:
    """t-statistic for the mean, with a Bartlett Newey-West variance."""
    series = np.asarray(series, dtype=np.float64)
    length = series.shape[0]
    mean = float(series.mean())
    demeaned = series - mean
    if length < 3:
        return float("nan")
    lags = int(np.floor(4.0 * (length / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(demeaned @ demeaned) / length
    variance = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(demeaned[lag:] @ demeaned[:-lag]) / length
        variance += 2.0 * weight * gamma
    variance = max(variance, 0.0)
    se = np.sqrt(variance / length)
    if se == 0:
        return float("nan")
    return mean / se


def monthly_loss_gap(y, yhat_row, yhat_col, yyyymm, finite) -> np.ndarray:
    """Mean squared-error difference, row model minus column model, by month.

    A positive value means the column model has the smaller squared error.
    """
    months = np.unique(yyyymm[finite])
    gaps = np.empty(months.shape[0], dtype=np.float64)
    for i, month in enumerate(months):
        mask = finite & (yyyymm == month)
        e_row = y[mask] - yhat_row[mask]
        e_col = y[mask] - yhat_col[mask]
        gaps[i] = float(np.mean(e_row**2 - e_col**2))
    return gaps
