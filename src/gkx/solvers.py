"""OLS, Huber, elastic net, PCR and PLS from cross-product moments."""

from __future__ import annotations

import numpy as np

from gkx.moments import standardize_moments


def _solve_spd(gram: np.ndarray, cross: np.ndarray, active: np.ndarray) -> np.ndarray:
    theta = np.zeros(gram.shape[0], dtype=np.float64)
    if not np.any(active):
        return theta
    block = gram[np.ix_(active, active)]
    rhs = cross[active]
    coef, _, _, _ = np.linalg.lstsq(block, rhs, rcond=None)
    theta[active] = coef
    return theta


def fit_ols_raw(moments: dict) -> dict:
    """Uncentered OLS on the original columns. No intercept is added."""
    variance = np.diag(moments["gram"])
    active = variance > 1e-10
    theta = _solve_spd(moments["gram"], moments["cross"], active)
    if np.any(active):
        singular = np.linalg.svd(moments["gram"][np.ix_(active, active)], compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-15))
    else:
        condition = np.inf
    return {
        "theta": theta,
        "active": active,
        "mu": np.zeros(theta.shape[0]),
        "scale": np.ones(theta.shape[0]),
        "intercept": 0.0,
        "n": moments["n"],
        "sum_yy": moments["sum_yy"],
        "gram": moments["gram"],
        "cross": moments["cross"],
        "standardized": False,
        "rank": int(np.linalg.matrix_rank(moments["gram"][np.ix_(active, active)])) if np.any(active) else 0,
        "condition": condition,
    }


def _spectral_norm(gram: np.ndarray, active: np.ndarray) -> float:
    dim = int(active.sum())
    if dim == 0:
        return 1.0
    block = gram[np.ix_(active, active)]
    vector = np.random.default_rng(0).normal(size=dim)
    vector /= np.linalg.norm(vector)
    for _ in range(30):
        vector = block @ vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0
        vector /= norm
    return float(vector @ (block @ vector))


def _prox_enet(value: np.ndarray, step: float, lam: float, rho: float) -> np.ndarray:
    """Prox of step * (lam*(1-rho)*|.| + 0.5*lam*rho*||.||^2)."""
    threshold = step * lam * (1.0 - rho)
    shrink = 1.0 + step * lam * rho
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0) / shrink


def _enet_fista(
    gram: np.ndarray,
    cross: np.ndarray,
    active: np.ndarray,
    n: int,
    lam: float,
    rho: float,
    max_iter: int,
    tol: float,
    theta0: np.ndarray | None = None,
) -> np.ndarray:
    theta = np.zeros(gram.shape[0], dtype=np.float64) if theta0 is None else theta0.copy()
    anchor = theta.copy()
    momentum = 1.0
    lipschitz = max(_spectral_norm(gram, active) / n, 1e-12)
    step = 1.0 / lipschitz
    for _ in range(max_iter):
        grad = (gram @ anchor - cross) / n
        grad[~active] = 0.0
        updated = _prox_enet(anchor - step * grad, step, lam, rho)
        updated[~active] = 0.0
        momentum_next = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        anchor_next = updated + ((momentum - 1.0) / momentum_next) * (updated - theta)
        shift = np.linalg.norm(updated - theta)
        theta = updated
        anchor = anchor_next
        momentum = momentum_next
        if shift < tol * (1.0 + np.linalg.norm(theta)):
            break
    return theta


def fit_enet(
    moments: dict,
    lam: float,
    rho: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> dict:
    """Elastic net in standardized coordinates.

    Smooth loss is (1 / (2N)) ||y - Z θ||^2. The penalty follows the paper:
    lam * (1 - rho) * ||θ||_1 + 0.5 * lam * rho * ||θ||_2^2.
    """
    scaled = standardize_moments(moments)
    theta = _enet_fista(
        scaled["gram"], scaled["cross"], scaled["active"], scaled["n"], lam, rho, max_iter, tol
    )
    return {
        **scaled,
        "theta": theta,
        "lambda": lam,
        "nonzero": int(np.count_nonzero(np.abs(theta) > 1e-8)),
        "intercept": moments["sum_y"] / moments["n"],
        "standardized": True,
    }


def fit_wls(base: dict, weighted_moments: dict) -> np.ndarray:
    """One Huber IRLS step in the fixed standardized coordinates of `base`."""
    scaled = weighted_standardized(base, weighted_moments)
    return _solve_spd(scaled["gram"], scaled["cross"], base["active"])


def fit_enet_weighted(
    base: dict,
    weighted_moments: dict,
    lam: float,
    rho: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-7,
    theta0: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Weighted elastic net with an unpenalized intercept.

    The intercept is profiled out of the Huber quadratic step. With constant
    weights and centered columns it reduces to the training mean of y.
    """
    scaled = weighted_standardized(base, weighted_moments)
    n = float(base["n"])
    gram = scaled["gram"]
    cross = scaled["cross"]
    ones = scaled["ones"]
    active = base["active"]
    sum_w = float(scaled["sum_w"])
    sum_wy = float(scaled["sum_wy"])
    theta = np.zeros(gram.shape[0]) if theta0 is None else theta0.copy()
    anchor = theta.copy()
    momentum = 1.0
    lipschitz = max(_spectral_norm(gram, active) / n, 1e-12)
    step = 1.0 / lipschitz

    def intercept_at(coef: np.ndarray) -> float:
        if sum_w <= 0:
            return 0.0
        return (sum_wy - float(coef @ ones)) / sum_w

    for _ in range(max_iter):
        level = intercept_at(anchor)
        grad = (gram @ anchor - cross + level * ones) / n
        grad[~active] = 0.0
        updated = _prox_enet(anchor - step * grad, step, lam, rho)
        updated[~active] = 0.0
        momentum_next = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        anchor = updated + ((momentum - 1.0) / momentum_next) * (updated - theta)
        shift = np.linalg.norm(updated - theta)
        theta = updated
        momentum = momentum_next
        if shift < tol * (1.0 + np.linalg.norm(theta)):
            break
    return theta, intercept_at(theta)


def fit_pcr_path(moments: dict, ks: list[int]) -> dict:
    """Principal-component regression for every requested component count."""
    scaled = standardize_moments(moments)
    active = scaled["active"]
    gram = scaled["gram"]
    cross = scaled["cross"]
    idx = np.flatnonzero(active)
    evals, evecs = np.linalg.eigh(gram[np.ix_(idx, idx)])
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    loadings = {}
    thetas = {}
    max_k = min(max(ks), evecs.shape[1])
    for k in ks:
        k_use = min(k, max_k)
        weights = evecs[:, :k_use]
        small = weights.T @ gram[np.ix_(idx, idx)] @ weights
        rhs = weights.T @ cross[idx]
        beta = np.linalg.lstsq(small, rhs, rcond=None)[0]
        theta = np.zeros(gram.shape[0], dtype=np.float64)
        theta[idx] = weights @ beta
        thetas[k] = theta
        loadings[k] = k_use
    return {
        **scaled,
        "thetas": thetas,
        "components": loadings,
        "intercept": moments["sum_y"] / moments["n"],
        "standardized": True,
    }


def fit_pls_path(moments: dict, ks: list[int]) -> dict:
    """Univariate NIPALS PLS. Every requested component count shares one deflation."""
    result = _pls_path_explicit(standardize_moments(moments), ks)
    result["intercept"] = moments["sum_y"] / moments["n"]
    result["sum_yy"] = moments["sum_yy"]
    return result


def _pls_path_explicit(scaled: dict, ks: list[int]) -> dict:
    active = scaled["active"]
    idx = np.flatnonzero(active)
    gram0 = scaled["gram"][np.ix_(idx, idx)]
    cross0 = scaled["cross"][idx]
    gram = gram0.copy()
    cross = cross0.copy()
    mapping = np.eye(idx.size)
    theta_reduced = np.zeros(idx.size, dtype=np.float64)
    thetas = {}
    max_k = min(max(ks), idx.size)
    wanted = set(int(k) for k in ks)
    n_done = 0
    for k in range(1, max_k + 1):
        norm = np.linalg.norm(cross)
        if norm < 1e-12:
            break
        weight = cross / norm
        gram_weight = gram @ weight
        denominator = float(weight @ gram_weight)
        if denominator <= 1e-18:
            break
        loading = gram_weight / denominator
        score_coef = float(weight @ cross) / denominator
        original = mapping @ weight
        theta_reduced = theta_reduced + score_coef * original
        n_done = k
        if k in wanted:
            theta = np.zeros(scaled["gram"].shape[0], dtype=np.float64)
            theta[idx] = theta_reduced
            thetas[k] = theta
        # Deflate Z on the right and y by its projection on the score.
        # b_new = A' (b - c * G w), with A = I - w p' and G w = denom * p.
        cross_before = cross - score_coef * gram_weight
        step = np.eye(idx.size) - np.outer(weight, loading)
        mapping = mapping @ step
        gram = step.T @ gram @ step
        cross = step.T @ cross_before
    for k in ks:
        if k not in thetas and n_done:
            theta = np.zeros(scaled["gram"].shape[0], dtype=np.float64)
            theta[idx] = theta_reduced
            thetas[int(k)] = theta
    scaled = dict(scaled)
    scaled["thetas"] = thetas
    scaled["components"] = n_done
    scaled["standardized"] = True
    return scaled


def weighted_standardized(base: dict, weighted: dict) -> dict:
    """Z_s' W Z_s and Z_s' W y, using the unweighted centering and scaling in `base`."""
    mu = base["mu"]
    scale = base["scale"]
    active = base["active"]
    inv_scale = 1.0 / scale
    sum_w = weighted["sum_w"]
    column_w = weighted["column_sum"]
    centered = (
        weighted["gram"]
        - np.outer(column_w, mu)
        - np.outer(mu, column_w)
        + sum_w * np.outer(mu, mu)
    )
    gram = centered * inv_scale[:, None] * inv_scale[None, :]
    cross = (weighted["cross"] - mu * weighted["sum_wy"]) * inv_scale
    ones = (weighted["column_sum"] - mu * weighted["sum_w"]) * inv_scale
    gram[~active, :] = 0.0
    gram[:, ~active] = 0.0
    cross[~active] = 0.0
    ones[~active] = 0.0
    return {
        "gram": gram,
        "cross": cross,
        "ones": ones,
        "n": base["n"],
        "active": active,
        "sum_w": weighted["sum_w"],
        "sum_wy": weighted["sum_wy"],
    }


def huber_weights(residual: np.ndarray, xi: float) -> np.ndarray:
    magnitude = np.abs(residual)
    weights = np.ones_like(magnitude)
    tail = magnitude > xi
    weights[tail] = xi / magnitude[tail]
    return weights


def reconstruction_sse(theta: np.ndarray, scaled: dict) -> float:
    """||y - Z_s θ||^2 from moments. Z_s is centered and scaled."""
    residual_gram = theta @ scaled["gram"] @ theta
    residual_cross = theta @ scaled["cross"]
    return float(scaled["sum_yy"] - 2.0 * residual_cross + residual_gram)


def predict_raw(
    chars_month: np.ndarray,
    sic_month: np.ndarray,
    x: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    theta: np.ndarray,
    intercept: float = 0.0,
    standardized: bool = True,
) -> np.ndarray:
    """Predict one month.

    Standardized fits are stored in centered/scaled coordinates and the training
    mean of y is added back. Raw OLS uses the original columns and no intercept.
    """
    n_char = chars_month.shape[1]
    n_macro = x.shape[0]
    p_inter = n_macro * n_char
    if standardized:
        theta_use = theta / scale
        adjustment = intercept - float(np.dot(mu / scale, theta))
    else:
        theta_use = theta
        adjustment = intercept
    slopes = x @ theta_use[:p_inter].reshape(n_macro, n_char)
    yhat = chars_month.astype(np.float64) @ slopes + adjustment
    sic_coef = theta_use[p_inter:]
    codes = sic_month
    valid = codes >= 0
    if np.any(valid) and sic_coef.size:
        yhat[valid] += sic_coef[codes[valid]]
    return yhat.astype(np.float64)
