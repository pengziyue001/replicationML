"""Recursive annual evaluation of the linear model group.

Training grows by one year at each refit. Validation is the preceding 12
years. Hyperparameters minimize validation MSE, and the tested year is not
used for estimation or tuning.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gkx.config import load_config
from gkx.moments import accumulate
from gkx.panel import Panel
from gkx.paths import LOG_DIR, PREDICTION_DIR
from gkx.solvers import (
    fit_enet,
    fit_enet_weighted,
    fit_ols_raw,
    fit_pcr_path,
    fit_pls_path,
    huber_weights,
    predict_raw,
)

FIT_DIR = PREDICTION_DIR / "fits"
FIT_DIR.mkdir(parents=True, exist_ok=True)
MODELS = ["ols", "ols_huber", "ols3", "ols3_huber", "enet", "enet_huber", "pls", "pcr"]


def _load_params() -> dict:
    path = LOG_DIR / "linear_params.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {name: {} for name in MODELS}


def _save_params(params: dict) -> None:
    (LOG_DIR / "linear_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def _prediction_array(panel: Panel, name: str) -> np.ndarray:
    path = PREDICTION_DIR / f"{name}.npy"
    if path.exists():
        array = np.load(path)
        if array.shape[0] == panel.n:
            return array
    return np.full(panel.n, np.nan, dtype=np.float32)


def _store_fit(name: str, year: int, fit: dict, theta: np.ndarray) -> None:
    np.savez(
        FIT_DIR / f"{name}_{year}.npz",
        theta=theta,
        mu=np.asarray(fit["mu"], dtype=np.float64),
        scale=np.asarray(fit["scale"], dtype=np.float64),
        intercept=np.float64(fit.get("intercept", 0.0)),
        standardized=np.int8(1 if fit.get("standardized", False) else 0),
    )


def _predict_month(panel: Panel, start: int, end: int, x: np.ndarray, fit: dict, theta: np.ndarray):
    return predict_raw(
        panel.chars[start:end],
        panel.sic[start:end],
        x,
        fit["mu"],
        fit["scale"],
        theta,
        intercept=float(fit.get("intercept", 0.0)),
        standardized=bool(fit.get("standardized", False)),
    )


def _score(panel, slices, x_map, fit, theta, destination=None) -> float:
    sse = 0.0
    count = 0
    y = panel.y
    for yyyymm, start, end in slices:
        pred = _predict_month(panel, start, end, x_map[yyyymm], fit, theta)
        if destination is not None:
            destination[start:end] = pred.astype(np.float32)
        sse += float(np.sum((y[start:end] - pred) ** 2))
        count += end - start
    return sse / max(count, 1)


def _xi(panel: Panel, slices, quantile: float) -> float:
    pieces = [np.abs(panel.y[start:end]) for _, start, end in slices]
    return float(np.quantile(np.concatenate(pieces), quantile))


def _huber_reweight(panel, slices, x_map, fit, theta, xi: float) -> np.ndarray:
    weights = np.ones(panel.n, dtype=np.float64)
    y = panel.y
    for yyyymm, start, end in slices:
        pred = _predict_month(panel, start, end, x_map[yyyymm], fit, theta)
        weights[start:end] = huber_weights(y[start:end] - pred, xi)
    return weights


def _fit_huber_ols(panel, train_slices, x_map, moments, xi, rounds, tol):
    fit = fit_ols_raw(moments)
    theta = fit["theta"]
    for _ in range(rounds):
        weights = _huber_reweight(panel, train_slices, x_map, fit, theta, xi)
        weighted = accumulate(
            panel.chars, panel.y, panel.sic, x_map, train_slices, panel.n_sic, weights
        )
        active = np.diag(weighted["gram"]) > 1e-10
        updated = np.zeros_like(theta)
        block = active
        coef = np.linalg.lstsq(weighted["gram"][np.ix_(block, block)], weighted["cross"][block], rcond=None)[0]
        updated[block] = coef
        if np.linalg.norm(updated - theta) < tol * (1.0 + np.linalg.norm(updated)):
            theta = updated
            break
        theta = updated
    fit["theta"] = theta
    fit["xi"] = xi
    return fit, theta


def _ols3_design(panel: Panel, start: int, end: int, columns: list[int]) -> np.ndarray:
    chars = np.asarray(panel.chars[start:end][:, columns], dtype=np.float64)
    return np.column_stack([np.ones(end - start), chars])


def _fit_ols3(panel, slices, columns, weights=None) -> np.ndarray:
    dim = len(columns) + 1
    gram = np.zeros((dim, dim))
    cross = np.zeros(dim)
    for _, start, end in slices:
        design = _ols3_design(panel, start, end, columns)
        target = np.asarray(panel.y[start:end], dtype=np.float64)
        if weights is None:
            gram += design.T @ design
            cross += design.T @ target
        else:
            w = weights[start:end]
            gram += (design * w[:, None]).T @ design
            cross += (design * w[:, None]).T @ target
    return np.linalg.lstsq(gram, cross, rcond=None)[0]


def _predict_ols3(panel, start, end, columns, theta) -> np.ndarray:
    return _ols3_design(panel, start, end, columns) @ theta


def _score_ols3(panel, slices, columns, theta, destination=None) -> float:
    sse = 0.0
    count = 0
    for _, start, end in slices:
        pred = _predict_ols3(panel, start, end, columns, theta)
        if destination is not None:
            destination[start:end] = pred.astype(np.float32)
        sse += float(np.sum((panel.y[start:end] - pred) ** 2))
        count += end - start
    return sse / max(count, 1)


def _fit_ols3_huber(panel, slices, columns, xi, rounds, tol) -> np.ndarray:
    theta = _fit_ols3(panel, slices, columns)
    y = panel.y
    for _ in range(rounds):
        weights = np.ones(panel.n, dtype=np.float64)
        for _, start, end in slices:
            pred = _predict_ols3(panel, start, end, columns, theta)
            weights[start:end] = huber_weights(y[start:end] - pred, xi)
        updated = _fit_ols3(panel, slices, columns, weights)
        if np.linalg.norm(updated - theta) < tol * (1.0 + np.linalg.norm(updated)):
            return updated
        theta = updated
    return theta


def _select_lambda(panel, moments, train_slices, val_slices, x_map, cfg) -> tuple[dict, float]:
    lambdas = [float(v) for v in cfg["enet"]["lambdas"]]
    rho = float(cfg["enet"]["rho"])
    best = None
    for lam in lambdas:
        fit = fit_enet(moments, lam, rho, int(cfg["enet"]["max_iter"]), float(cfg["enet"]["tol"]))
        mse = _score(panel, val_slices, x_map, fit, fit["theta"])
        if best is None or mse < best[0]:
            best = (mse, lam, fit)
    # Widen once if validation prefers a boundary, so the grid is not the result.
    edge = best[1]
    extra = None
    if edge == lambdas[0]:
        extra = edge / 10.0
    elif edge == lambdas[-1]:
        extra = edge * 10.0
    if extra is not None:
        fit = fit_enet(moments, extra, rho, int(cfg["enet"]["max_iter"]), float(cfg["enet"]["tol"]))
        mse = _score(panel, val_slices, x_map, fit, fit["theta"])
        if mse < best[0]:
            best = (mse, extra, fit)
    return best[2], best[1]


def _refit_enet_huber(panel, base_fit, train_slices, x_map, lam, rho, xi, rounds, tol):
    theta = base_fit["theta"].copy()
    intercept = float(base_fit["intercept"])
    fit = dict(base_fit)
    for _ in range(rounds):
        fit["theta"] = theta
        fit["intercept"] = intercept
        weights = _huber_reweight(panel, train_slices, x_map, fit, theta, xi)
        weighted = accumulate(
            panel.chars, panel.y, panel.sic, x_map, train_slices, panel.n_sic, weights
        )
        updated, intercept_new = fit_enet_weighted(
            base_fit, weighted, lam, rho, max_iter=400, tol=tol, theta0=theta
        )
        if np.linalg.norm(updated - theta) < tol * (1.0 + np.linalg.norm(updated)):
            theta, intercept = updated, intercept_new
            break
        theta, intercept = updated, intercept_new
    fit["theta"] = theta
    fit["intercept"] = intercept
    fit["standardized"] = True
    fit["xi"] = xi
    fit["nonzero"] = int(np.count_nonzero(np.abs(theta) > 1e-8))
    return fit, theta


def _select_components(panel, path, val_slices, x_map, ks) -> tuple[np.ndarray, int, float]:
    best_mse = None
    best_k = None
    best_theta = None
    for k in ks:
        if k not in path["thetas"]:
            continue
        mse = _score(panel, val_slices, x_map, path, path["thetas"][k])
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_k = int(k)
            best_theta = path["thetas"][k]
    return best_theta, best_k, best_mse


def run(years: list[int]) -> None:
    cfg = load_config()
    panel = Panel()
    params = _load_params()
    columns = [panel.char_names.index(name) for name in cfg["ols3"]["features"]]
    huber_cfg = cfg["huber"]
    print(
        f"panel rows={panel.n:,} characteristics={len(panel.char_names)} industries={panel.n_sic}",
        flush=True,
    )
    for year in years:
        train_years, val_years = panel.train_val_years(year)
        train_slices = panel.slices_for_years(train_years)
        val_slices = panel.slices_for_years(val_years)
        test_slices = panel.slices_for_years([year])
        mu, std = panel.macro_mean_std(train_slices)
        all_slices = train_slices + val_slices + test_slices
        x_map = panel.x_by_month(all_slices, mu, std)
        print(
            f"\n=== {year} train {train_years[0]}-{train_years[-1]} "
            f"val {val_years[0]}-{val_years[-1]} ===",
            flush=True,
        )
        started = time.perf_counter()
        moments = accumulate(panel.chars, panel.y, panel.sic, x_map, train_slices, panel.n_sic)
        xi = _xi(panel, train_slices, float(huber_cfg["quantile"]))
        print(f"moments {time.perf_counter() - started:.1f}s  huber xi={xi:.4f}", flush=True)

        pending = [name for name in MODELS if str(year) not in params[name]]
        if not pending:
            print("already done", flush=True)
            continue

        if "ols" in pending or "ols_huber" in pending:
            fit = fit_ols_raw(moments)
            if "ols" in pending:
                destination = _prediction_array(panel, "ols")
                test_mse = _score(panel, test_slices, x_map, fit, fit["theta"], destination)
                np.save(PREDICTION_DIR / "ols.npy", destination)
                _store_fit("ols", year, fit, fit["theta"])
                params["ols"][str(year)] = {
                    "test_mse": test_mse,
                    "rank": fit["rank"],
                    "condition": fit["condition"],
                }
                print(f"ols test_mse={test_mse:.6f} rank={fit['rank']}", flush=True)
            if "ols_huber" in pending:
                hfit, htheta = _fit_huber_ols(
                    panel,
                    train_slices,
                    x_map,
                    moments,
                    xi,
                    int(huber_cfg["max_iter"]),
                    float(huber_cfg["tol"]),
                )
                destination = _prediction_array(panel, "ols_huber")
                test_mse = _score(panel, test_slices, x_map, hfit, htheta, destination)
                np.save(PREDICTION_DIR / "ols_huber.npy", destination)
                _store_fit("ols_huber", year, hfit, htheta)
                params["ols_huber"][str(year)] = {"test_mse": test_mse, "xi": xi}
                print(f"ols_huber test_mse={test_mse:.6f}", flush=True)

        if "ols3" in pending or "ols3_huber" in pending:
            theta3 = _fit_ols3(panel, train_slices, columns)
            if "ols3" in pending:
                destination = _prediction_array(panel, "ols3")
                test_mse = _score_ols3(panel, test_slices, columns, theta3, destination)
                np.save(PREDICTION_DIR / "ols3.npy", destination)
                params["ols3"][str(year)] = {"test_mse": test_mse}
                print(f"ols3 test_mse={test_mse:.6f}", flush=True)
            if "ols3_huber" in pending:
                theta3h = _fit_ols3_huber(
                    panel,
                    train_slices,
                    columns,
                    xi,
                    int(huber_cfg["max_iter"]),
                    float(huber_cfg["tol"]),
                )
                destination = _prediction_array(panel, "ols3_huber")
                test_mse = _score_ols3(panel, test_slices, columns, theta3h, destination)
                np.save(PREDICTION_DIR / "ols3_huber.npy", destination)
                params["ols3_huber"][str(year)] = {"test_mse": test_mse, "xi": xi}
                print(f"ols3_huber test_mse={test_mse:.6f}", flush=True)

        if "enet" in pending or "enet_huber" in pending:
            enet_fit, lam = _select_lambda(panel, moments, train_slices, val_slices, x_map, cfg)
            if "enet" in pending:
                destination = _prediction_array(panel, "enet")
                test_mse = _score(panel, test_slices, x_map, enet_fit, enet_fit["theta"], destination)
                np.save(PREDICTION_DIR / "enet.npy", destination)
                _store_fit("enet", year, enet_fit, enet_fit["theta"])
                params["enet"][str(year)] = {
                    "test_mse": test_mse,
                    "lambda": lam,
                    "nonzero": enet_fit["nonzero"],
                }
                print(f"enet test_mse={test_mse:.6f} lambda={lam:.3g} nonzero={enet_fit['nonzero']}", flush=True)
            if "enet_huber" in pending:
                hfit, htheta = _refit_enet_huber(
                    panel,
                    enet_fit,
                    train_slices,
                    x_map,
                    lam,
                    float(cfg["enet"]["rho"]),
                    xi,
                    rounds=8,
                    tol=float(huber_cfg["tol"]),
                )
                destination = _prediction_array(panel, "enet_huber")
                test_mse = _score(panel, test_slices, x_map, hfit, htheta, destination)
                np.save(PREDICTION_DIR / "enet_huber.npy", destination)
                _store_fit("enet_huber", year, hfit, htheta)
                params["enet_huber"][str(year)] = {
                    "test_mse": test_mse,
                    "lambda": lam,
                    "nonzero": hfit["nonzero"],
                    "xi": xi,
                }
                print(f"enet_huber test_mse={test_mse:.6f} nonzero={hfit['nonzero']}", flush=True)

        if "pls" in pending:
            path = fit_pls_path(moments, [int(k) for k in cfg["pls"]["ks"]])
            theta, k, val_mse = _select_components(panel, path, val_slices, x_map, cfg["pls"]["ks"])
            destination = _prediction_array(panel, "pls")
            test_mse = _score(panel, test_slices, x_map, path, theta, destination)
            np.save(PREDICTION_DIR / "pls.npy", destination)
            _store_fit("pls", year, path, theta)
            params["pls"][str(year)] = {"test_mse": test_mse, "k": k, "val_mse": val_mse}
            print(f"pls test_mse={test_mse:.6f} k={k}", flush=True)

        if "pcr" in pending:
            path = fit_pcr_path(moments, [int(k) for k in cfg["pcr"]["ks"]])
            theta, k, val_mse = _select_components(panel, path, val_slices, x_map, cfg["pcr"]["ks"])
            destination = _prediction_array(panel, "pcr")
            test_mse = _score(panel, test_slices, x_map, path, theta, destination)
            np.save(PREDICTION_DIR / "pcr.npy", destination)
            _store_fit("pcr", year, path, theta)
            params["pcr"][str(year)] = {"test_mse": test_mse, "k": k, "val_mse": val_mse}
            print(f"pcr test_mse={test_mse:.6f} k={k}", flush=True)

        _save_params(params)
        print(f"year {year} finished in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None, help="Run a single test year, for a timing check.")
    args = parser.parse_args()
    cfg = load_config()["sample"]
    if args.year is not None:
        years = [args.year]
    else:
        years = list(range(int(cfg["first_test_year"]), int(cfg["last_test_year"]) + 1))
    run(years)


if __name__ == "__main__":
    main()
