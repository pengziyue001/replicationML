"""Recursive spline GLM with group lasso.

Each of the 920 predictors is expanded as
    z, (z - c1)^2, (z - c2)^2, (z - c3)^2
with three knots at the training-sample quartiles. A group-lasso penalty keeps
or drops all four terms of a predictor together. One unpenalized intercept is
added separately. Huber refits the lambda chosen under squared loss.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gkx.config import load_config
from gkx.design import month_design
from gkx.panel import Panel
from gkx.paths import LOG_DIR, PREDICTION_DIR
from gkx.solvers import _spectral_norm, huber_weights

GROUP_SIZE = 4


def _basis(design: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """design (n, p), knots (n_knots, p) -> (n, p * 4), one group per predictor."""
    n_obs, n_pred = design.shape
    basis = np.empty((n_obs, n_pred, GROUP_SIZE), dtype=np.float32)
    basis[:, :, 0] = design
    for knot_id in range(knots.shape[0]):
        deviation = design - knots[knot_id]
        basis[:, :, knot_id + 1] = deviation * deviation
    return basis.reshape(n_obs, n_pred * GROUP_SIZE)


def _sample_knots(panel: Panel, slices, x_map, limit: int, quantiles) -> np.ndarray:
    blocks = []
    filled = 0
    for yyyymm, start, end in slices:
        take = min(end - start, limit - filled)
        if take <= 0:
            break
        design = month_design(panel.chars[start : start + take], panel.sic[start : start + take], x_map[yyyymm], panel.n_sic)
        blocks.append(design)
        filled += take
    sample = np.vstack(blocks)
    return np.quantile(sample, quantiles, axis=0).astype(np.float32)


def _accumulate(panel, slices, x_map, knots, theta=None, intercept=0.0, xi=None):
    n_pred = panel.chars.shape[1] * 9 + panel.n_sic
    width = n_pred * GROUP_SIZE
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros(width, dtype=np.float64)
    column_sum = np.zeros(width, dtype=np.float64)
    sum_y = 0.0
    sum_yy = 0.0
    sum_w = 0.0
    sum_wy = 0.0
    n_obs = 0
    for yyyymm, start, end in slices:
        raw = month_design(panel.chars[start:end], panel.sic[start:end], x_map[yyyymm], panel.n_sic)
        basis = _basis(raw, knots)
        target = np.asarray(panel.y[start:end], dtype=np.float64)
        if theta is None:
            weighted_basis = basis
            weight = None
        else:
            fitted = basis @ theta.astype(np.float32) + np.float32(intercept)
            residual = target - fitted.astype(np.float64)
            weight = huber_weights(residual, xi).astype(np.float32)
            sqrt_weight = np.sqrt(weight)
            weighted_basis = basis * sqrt_weight[:, None]
        response = target.astype(np.float32) if weight is None else (weight * target).astype(np.float32)
        gram += weighted_basis.T @ weighted_basis
        cross += basis.T @ response
        column_sum += basis.sum(axis=0) if weight is None else (basis * weight[:, None]).sum(axis=0)
        sum_y += float(target.sum())
        sum_yy += float(target @ target)
        if weight is None:
            sum_w += float(end - start)
            sum_wy += float(target.sum())
        else:
            sum_w += float(weight.sum())
            sum_wy += float(weight @ target)
        n_obs += end - start
    return {
        "gram": gram,
        "cross": cross,
        "column_sum": column_sum,
        "sum_y": sum_y,
        "sum_yy": sum_yy,
        "sum_w": sum_w,
        "sum_wy": sum_wy,
        "n": n_obs,
    }


def _prox_group(value: np.ndarray, step: float, lam: float) -> np.ndarray:
    grouped = value.reshape(-1, GROUP_SIZE)
    norms = np.linalg.norm(grouped, axis=1)
    scale = np.zeros_like(norms)
    thresh = step * lam
    keep = norms > thresh
    scale[keep] = 1.0 - thresh / norms[keep]
    return (grouped * scale[:, None]).ravel()


def _fit(moments: dict, lam: float, max_iter: int, tol: float, theta0=None):
    gram = moments["gram"]
    cross = moments["cross"]
    column_sum = moments["column_sum"]
    n_obs = float(moments["n"])
    sum_w = float(moments["sum_w"])
    sum_wy = float(moments["sum_wy"])
    theta = np.zeros(gram.shape[0]) if theta0 is None else theta0.copy()
    anchor = theta.copy()
    momentum = 1.0
    active = np.ones(gram.shape[0], dtype=bool)
    lipschitz = max(_spectral_norm(gram, active) / n_obs, 1e-12)
    step = 1.0 / lipschitz

    def intercept(coef: np.ndarray) -> float:
        if sum_w <= 0:
            return 0.0
        return (sum_wy - float(column_sum @ coef)) / sum_w

    for _ in range(max_iter):
        level = intercept(anchor)
        grad = (gram @ anchor - cross + level * column_sum) / n_obs
        updated = _prox_group(anchor - step * grad, step, lam)
        momentum_next = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        anchor_next = updated + ((momentum - 1.0) / momentum_next) * (updated - theta)
        shift = np.linalg.norm(updated - theta)
        theta = updated
        anchor = anchor_next
        momentum = momentum_next
        if shift < tol * (1.0 + np.linalg.norm(theta)):
            break
    level = intercept(theta)
    grouped = theta.reshape(-1, GROUP_SIZE)
    n_groups = int(np.count_nonzero(np.linalg.norm(grouped, axis=1) > 1e-8))
    return theta, level, n_groups


def _score(panel, slices, x_map, knots, theta, intercept, destination=None) -> float:
    sse = 0.0
    count = 0
    theta32 = theta.astype(np.float32)
    for yyyymm, start, end in slices:
        raw = month_design(panel.chars[start:end], panel.sic[start:end], x_map[yyyymm], panel.n_sic)
        pred = _basis(raw, knots) @ theta32 + np.float32(intercept)
        if destination is not None:
            destination[start:end] = pred
        gap = np.asarray(panel.y[start:end], dtype=np.float64) - pred
        sse += float(gap @ gap)
        count += end - start
    return sse / max(count, 1)


def _characteristic_drops(moments, theta, intercept, n_char: int) -> np.ndarray:
    """Unweighted R² drop from removing every spline group tied to one characteristic."""
    gram = moments["gram"]
    cross = moments["cross"]
    column_sum = moments["column_sum"]
    sum_y = moments["sum_y"]
    sum_yy = moments["sum_yy"]
    n_obs = moments["n"]
    sum_w = moments["sum_w"]

    def sse(coef):
        level = (sum_y - float(column_sum @ coef)) / n_obs if n_obs else 0.0
        # Use the model's intercept only as a check; the drop uses the optimal
        # intercept after the groups are removed.
        del level
        fitted = (moments["sum_wy"] - float(column_sum @ coef)) / sum_w
        residual = (
            sum_yy
            - 2.0 * fitted * sum_y
            + n_obs * fitted**2
            - 2.0 * float(coef @ cross)
            + 2.0 * fitted * float(coef @ column_sum)
            + float(coef @ gram @ coef)
        )
        return residual

    # The unweighted moments are the ones passed in. sum_wy/sum_w equal sum_y/n.
    full = sse(theta)
    drops = np.zeros(n_char)
    n_macro = 9
    for characteristic in range(n_char):
        reduced = theta.copy()
        for macro in range(n_macro):
            group = macro * n_char + characteristic
            reduced[group * GROUP_SIZE : (group + 1) * GROUP_SIZE] = 0.0
        drops[characteristic] = (sse(reduced) - full) / sum_yy
    return drops


def run(years: list[int]) -> None:
    cfg = load_config()
    glm_cfg = cfg["glm"]
    panel = Panel()
    path = LOG_DIR / "glm_params.json"
    params = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"glm": {}, "glm_huber": {}}
    arrays = {}
    for name in ("glm", "glm_huber"):
        file_path = PREDICTION_DIR / f"{name}.npy"
        arrays[name] = np.load(file_path) if file_path.exists() else np.full(panel.n, np.nan, dtype=np.float32)
    drops = {name: np.zeros(panel.chars.shape[1]) for name in ("glm", "glm_huber")}
    drop_counts = {name: 0 for name in ("glm", "glm_huber")}
    quantiles = [float(q) for q in glm_cfg["quantiles"]]
    lambdas = [float(v) for v in glm_cfg["lambdas"]]
    print(f"glm groups={panel.chars.shape[1] * 9 + panel.n_sic} basis={GROUP_SIZE}", flush=True)
    for year in years:
        if str(year) in params["glm"] and str(year) in params["glm_huber"]:
            print(f"{year} already done", flush=True)
            continue
        started = time.perf_counter()
        train_years, val_years = panel.train_val_years(year)
        train_slices = panel.slices_for_years(train_years)
        val_slices = panel.slices_for_years(val_years)
        test_slices = panel.slices_for_years([year])
        mu, std = panel.macro_mean_std(train_slices)
        x_map = panel.x_by_month(train_slices + val_slices + test_slices, mu, std)
        knots = _sample_knots(panel, train_slices, x_map, int(glm_cfg["knot_sample"]), quantiles)
        moments = _accumulate(panel, train_slices, x_map, knots)
        xi = float(
            np.quantile(
                np.concatenate([np.abs(panel.y[start:end]) for _, start, end in train_slices]),
                float(cfg["huber"]["quantile"]),
            )
        )
        print(f"=== GLM {year} gram {time.perf_counter() - started:.1f}s xi={xi:.4f} ===", flush=True)
        best = None
        for lam in lambdas:
            theta, intercept, n_groups = _fit(moments, lam, int(glm_cfg["max_iter"]), float(glm_cfg["tol"]))
            mse = _score(panel, val_slices, x_map, knots, theta, intercept)
            print(f"  lambda={lam:.3g} val_mse={mse:.6f} groups={n_groups}", flush=True)
            if best is None or mse < best[0]:
                best = (mse, lam, theta, intercept, n_groups)
        _, lam, theta, intercept, n_groups = best
        if "glm" not in params or str(year) not in params["glm"]:
            destination = arrays["glm"]
            test_mse = _score(panel, test_slices, x_map, knots, theta, intercept, destination)
            np.save(PREDICTION_DIR / "glm.npy", destination)
            params["glm"][str(year)] = {"lambda": lam, "groups": n_groups, "test_mse": test_mse, "val_mse": best[0]}
            drops["glm"] += _characteristic_drops(moments, theta, intercept, panel.chars.shape[1])
            drop_counts["glm"] += 1
            print(f"glm {year} test_mse={test_mse:.6f} lambda={lam:.3g} groups={n_groups}", flush=True)
        huber_theta, huber_intercept = theta, intercept
        for _ in range(int(glm_cfg["huber_rounds"])):
            weighted = _accumulate(
                panel, train_slices, x_map, knots, theta=huber_theta, intercept=huber_intercept, xi=xi
            )
            huber_theta, huber_intercept, huber_groups = _fit(
                weighted, lam, int(glm_cfg["max_iter"]), float(glm_cfg["tol"]), theta0=huber_theta
            )
        destination = arrays["glm_huber"]
        test_mse = _score(panel, test_slices, x_map, knots, huber_theta, huber_intercept, destination)
        np.save(PREDICTION_DIR / "glm_huber.npy", destination)
        params["glm_huber"][str(year)] = {
            "lambda": lam,
            "groups": int(huber_groups),
            "test_mse": test_mse,
            "xi": xi,
        }
        drops["glm_huber"] += _characteristic_drops(moments, huber_theta, intercept, panel.chars.shape[1])
        drop_counts["glm_huber"] += 1
        path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        print(
            f"glm_huber {year} test_mse={test_mse:.6f} groups={huber_groups} "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    importance = {
        name: {
            characteristic: float(drops[name][index] / max(drop_counts[name], 1))
            for index, characteristic in enumerate(panel.char_names)
        }
        for name in drops
        if drop_counts[name]
    }
    if importance:
        (LOG_DIR / "glm_importance.json").write_text(json.dumps(importance), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    sample = load_config()["sample"]
    years = [args.year] if args.year else list(range(int(sample["first_test_year"]), int(sample["last_test_year"]) + 1))
    run(years)


if __name__ == "__main__":
    main()
