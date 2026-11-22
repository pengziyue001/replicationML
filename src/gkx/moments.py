"""Cross-product moments of the Kronecker design, one month at a time.

For month t the macro vector x_t is common to every stock and

    z_{i,t} = (x_t ⊗ c_{i,t}, d_{i,t}).

Z'Z and Z'y then collapse to products of the characteristic cross-matrix with
x_t, plus industry counts. Optional weights produce Z' W Z and Z' W y, which
is the inner step of Huber IRLS. Weights are not squared.
"""

from __future__ import annotations

import numpy as np


def accumulate(
    chars: np.ndarray,
    y: np.ndarray,
    sic: np.ndarray,
    x_by_month: dict[int, np.ndarray],
    slices: list[tuple[int, int, int]],
    n_sic: int,
    weights: np.ndarray | None = None,
) -> dict:
    n_char = chars.shape[1]
    n_macro = next(iter(x_by_month.values())).shape[0]
    p_inter = n_macro * n_char
    p = p_inter + n_sic
    gram = np.zeros((p, p), dtype=np.float64)
    cross = np.zeros(p, dtype=np.float64)
    column_sum = np.zeros(p, dtype=np.float64)
    sum_y = 0.0
    sum_yy = 0.0
    sum_w = 0.0
    sum_wy = 0.0
    n_obs = 0

    for yyyymm, start, end in slices:
        x = np.asarray(x_by_month[int(yyyymm)], dtype=np.float64)
        c_mat = np.asarray(chars[start:end], dtype=np.float64)
        y_vec = np.asarray(y[start:end], dtype=np.float64)
        codes = np.asarray(sic[start:end])
        if weights is None:
            w = np.ones(end - start, dtype=np.float64)
        else:
            w = np.asarray(weights[start:end], dtype=np.float64)
        weighted_c = c_mat * w[:, None]
        ct_c = weighted_c.T @ c_mat
        ct_y = weighted_c.T @ y_vec
        ct_sum = weighted_c.sum(axis=0)

        gram[:p_inter, :p_inter] += np.kron(np.outer(x, x), ct_c)
        cross[:p_inter] += np.kron(x, ct_y)
        column_sum[:p_inter] += np.kron(x, ct_sum)

        valid = codes >= 0
        ct_d = np.zeros((n_char, n_sic), dtype=np.float64)
        counts = np.zeros(n_sic, dtype=np.float64)
        d_y = np.zeros(n_sic, dtype=np.float64)
        if np.any(valid):
            np.add.at(ct_d.T, codes[valid], weighted_c[valid])
            np.add.at(counts, codes[valid], w[valid])
            np.add.at(d_y, codes[valid], w[valid] * y_vec[valid])
        cross_block = (x[:, None, None] * ct_d[None, :, :]).reshape(p_inter, n_sic)
        gram[:p_inter, p_inter:] += cross_block
        gram[p_inter:, :p_inter] += cross_block.T
        gram[np.arange(p_inter, p), np.arange(p_inter, p)] += counts
        cross[p_inter:] += d_y
        column_sum[p_inter:] += counts

        sum_y += float(y_vec.sum())
        sum_yy += float(y_vec @ y_vec)
        sum_w += float(w.sum())
        sum_wy += float(w @ y_vec)
        n_obs += int(end - start)

    return {
        "n": n_obs,
        "gram": gram,
        "cross": cross,
        "column_sum": column_sum,
        "sum_y": sum_y,
        "sum_yy": sum_yy,
        "sum_w": sum_w,
        "sum_wy": sum_wy,
        "n_char": n_char,
        "n_macro": n_macro,
        "n_sic": n_sic,
    }


def standardize_moments(moments: dict, min_std: float = 1e-8) -> dict:
    """Map uncentered moments into the centered and column-scaled coordinates.

    Columns with training standard deviation below `min_std` are marked inactive.
    Their scaled column is left at zero and the coefficient is fixed at zero later.
    """
    n = moments["n"]
    gram = moments["gram"]
    cross = moments["cross"]
    column_sum = moments["column_sum"]
    mu = column_sum / n
    var = np.diag(gram) / n - mu**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    active = std >= min_std
    scale = np.ones_like(std)
    scale[active] = std[active]
    inv_scale = 1.0 / scale
    centered = gram - n * np.outer(mu, mu)
    gram_s = centered * inv_scale[:, None] * inv_scale[None, :]
    # Zero out inactive rows and columns so they cannot affect the fit.
    gram_s[~active, :] = 0.0
    gram_s[:, ~active] = 0.0
    cross_s = (cross - mu * moments["sum_y"]) * inv_scale
    cross_s[~active] = 0.0
    return {
        "mu": mu,
        "scale": scale,
        "active": active,
        "gram": gram_s,
        "cross": cross_s,
        "n": n,
        "sum_yy": moments["sum_yy"],
    }
