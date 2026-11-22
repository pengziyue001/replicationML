"""Build one month of the Kronecker design without storing the full sample."""

from __future__ import annotations

import numpy as np


def month_design(chars: np.ndarray, sic: np.ndarray, x: np.ndarray, n_sic: int) -> np.ndarray:
    """Return z = (x ⊗ c, industry dummies) for one month, float32."""
    n_obs, n_char = chars.shape
    n_macro = x.shape[0]
    design = np.empty((n_obs, n_macro * n_char + n_sic), dtype=np.float32)
    chars32 = np.asarray(chars, dtype=np.float32)
    for k, value in enumerate(np.asarray(x, dtype=np.float32)):
        design[:, k * n_char : (k + 1) * n_char] = chars32 * value
    dummies = design[:, n_macro * n_char :]
    dummies.fill(0.0)
    valid = sic >= 0
    if n_sic and np.any(valid):
        rows = np.flatnonzero(valid)
        dummies[rows, sic[valid]] = 1.0
    return design
