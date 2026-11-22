"""One-off check: does PLS sign depend on component count or scaling?

Does not write prediction files used by the GLM run.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from gkx.moments import accumulate, standardize_moments
from gkx.panel import Panel
from gkx.paths import LOG_DIR
from gkx.solvers import _pls_path_explicit, predict_raw

KS = [1, 2, 3, 4, 5, 6, 8, 10]


def _r2(panel, slices, x_map, fit, theta) -> float:
    sse = 0.0
    sst = 0.0
    for yyyymm, start, end in slices:
        pred = predict_raw(
            panel.chars[start:end],
            panel.sic[start:end],
            x_map[yyyymm],
            fit["mu"],
            fit["scale"],
            theta,
            intercept=float(fit["intercept"]),
            standardized=bool(fit["standardized"]),
        )
        y = np.asarray(panel.y[start:end], dtype=np.float64)
        gap = y - pred
        sse += float(gap @ gap)
        sst += float(y @ y)
    return 100.0 * (1.0 - sse / sst)


def _val_mse(panel, slices, x_map, fit, theta) -> float:
    sse = 0.0
    count = 0
    for yyyymm, start, end in slices:
        pred = predict_raw(
            panel.chars[start:end],
            panel.sic[start:end],
            x_map[yyyymm],
            fit["mu"],
            fit["scale"],
            theta,
            intercept=float(fit["intercept"]),
            standardized=bool(fit["standardized"]),
        )
        y = np.asarray(panel.y[start:end], dtype=np.float64)
        gap = y - pred
        sse += float(gap @ gap)
        count += end - start
    return sse / count


def _variants(moments):
    scaled = standardize_moments(moments)
    scale = scaled["scale"]
    centered = {
        "mu": scaled["mu"],
        "scale": np.ones_like(scale),
        "active": scaled["active"],
        "gram": scaled["gram"] * scale[:, None] * scale[None, :],
        "cross": scaled["cross"] * scale,
        "n": scaled["n"],
        "sum_yy": scaled["sum_yy"],
        "intercept": moments["sum_y"] / moments["n"],
        "standardized": True,
    }
    active = np.diag(moments["gram"]) > 1e-10
    raw_gram = moments["gram"].copy()
    raw_cross = moments["cross"].copy()
    raw_gram[~active, :] = 0.0
    raw_gram[:, ~active] = 0.0
    raw_cross[~active] = 0.0
    raw = {
        "mu": np.zeros(raw_gram.shape[0]),
        "scale": np.ones(raw_gram.shape[0]),
        "active": active,
        "gram": raw_gram,
        "cross": raw_cross,
        "n": moments["n"],
        "sum_yy": moments["sum_yy"],
        "intercept": 0.0,
        "standardized": False,
    }
    scaled = dict(scaled)
    scaled["intercept"] = moments["sum_y"] / moments["n"]
    scaled["standardized"] = True
    return {"standardized": scaled, "centered": centered, "raw": raw}


def main():
    params = json.loads((LOG_DIR / "linear_params.json").read_text(encoding="utf-8"))["pls"]
    counts = {}
    for row in params.values():
        counts[row["k"]] = counts.get(row["k"], 0) + 1
    print("chosen_k_counts", dict(sorted(counts.items())))
    panel = Panel()
    for year in (2000, 2010):
        train_years, val_years = panel.train_val_years(year)
        train_slices = panel.slices_for_years(train_years)
        val_slices = panel.slices_for_years(val_years)
        test_slices = panel.slices_for_years([year])
        mu, std = panel.macro_mean_std(train_slices)
        x_map = panel.x_by_month(train_slices + val_slices + test_slices, mu, std)
        moments = accumulate(panel.chars, panel.y, panel.sic, x_map, train_slices, panel.n_sic)
        print(f"\n=== {year} logged_k={params[str(year)]['k']} ===")
        for name, spec in _variants(moments).items():
            path = _pls_path_explicit(spec, KS)
            path["mu"] = spec["mu"]
            path["scale"] = spec["scale"]
            path["intercept"] = spec["intercept"]
            path["standardized"] = spec["standardized"]
            best_k = None
            best_mse = None
            for k in KS:
                if k not in path["thetas"]:
                    continue
                mse = _val_mse(panel, val_slices, x_map, path, path["thetas"][k])
                r2 = _r2(panel, test_slices, x_map, path, path["thetas"][k])
                print(f"{name:14} k={k:2d} val_mse={mse:.6f} test_r2={r2:7.3f}%")
                if best_mse is None or mse < best_mse:
                    best_mse = mse
                    best_k = k
            print(f"{name:14} validation picks k={best_k}")


if __name__ == "__main__":
    main()
