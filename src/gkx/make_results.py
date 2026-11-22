"""Tables and figures for the 1987–2016 recursive evaluation."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from gkx.metrics import monthly_loss_gap, newey_west_t, r2_oos, size_masks
from gkx.moments import accumulate, standardize_moments
from gkx.panel import Panel
from gkx.paths import FIGURE_DIR, LOG_DIR, PREDICTION_DIR, TABLE_DIR

DISPLAY = {
    "ols": "OLS",
    "ols_huber": "OLS+H",
    "ols3": "OLS-3",
    "ols3_huber": "OLS-3+H",
    "enet": "ENet",
    "enet_huber": "ENet+H",
    "pls": "PLS",
    "pcr": "PCR",
    "glm": "GLM",
    "glm_huber": "GLM+H",
}

# Published full-sample monthly R2, in percent, from Gu, Kelly and Xiu (2020).
PUBLISHED = {
    "OLS": -3.46,
    "OLS-3": 0.16,
    "ENet": 0.11,
    "PLS": 0.27,
    "PCR": 0.26,
    "GLM": 0.19,
}


def _available(panel: Panel) -> list[tuple[str, np.ndarray]]:
    test = (panel.yyyymm // 100 >= 1987) & (panel.yyyymm // 100 <= 2016)
    needed = int(test.sum())
    found = []
    for name in DISPLAY:
        path = PREDICTION_DIR / f"{name}.npy"
        if not path.exists():
            continue
        pred = np.load(path)
        if pred.shape[0] != panel.n:
            continue
        covered = int(np.isfinite(pred[test]).sum())
        if covered < 0.95 * needed:
            print(f"skip {name}: {covered:,} of {needed:,} test rows")
            continue
        found.append((name, pred))
    return found


def _write_r2(panel: Panel, models) -> list[dict]:
    test = (panel.yyyymm // 100) >= 1987
    test = test & (panel.yyyymm // 100 <= 2016)
    slices = [item for item in panel.slices if 1987 <= item[0] // 100 <= 2016]
    top, bottom = size_masks(slices, panel.me, panel.n)
    rows = []
    y = panel.y.astype(np.float64)
    for name, pred in models:
        finite = test & np.isfinite(pred)
        record = {"model": DISPLAY[name], "file": name}
        for label, mask in (
            ("all", finite),
            ("top1000", finite & top),
            ("bottom1000", finite & bottom),
        ):
            record[label] = 100.0 * r2_oos(y[mask], pred[mask].astype(np.float64))
        rows.append(record)
    header = "model,r2_all_pct,r2_top1000_pct,r2_bottom1000_pct"
    lines = [header]
    for row in rows:
        lines.append(f"{row['model']},{row['all']:.6f},{row['top1000']:.6f},{row['bottom1000']:.6f}")
    (TABLE_DIR / "r2_oos.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _write_dm(panel: Panel, models) -> None:
    y = panel.y.astype(np.float64)
    names = [DISPLAY[name] for name, _ in models]
    matrix = []
    for i, (_name_i, pred_i) in enumerate(models):
        row = []
        for j, (_name_j, pred_j) in enumerate(models):
            if i == j:
                row.append("")
                continue
            finite = np.isfinite(pred_i) & np.isfinite(pred_j)
            gaps = monthly_loss_gap(y, pred_i, pred_j, panel.yyyymm, finite)
            row.append(f"{newey_west_t(gaps):.3f}")
        matrix.append(row)
    lines = ["," + ",".join(names)]
    for name, row in zip(names, matrix):
        lines.append(name + "," + ",".join(row))
    (TABLE_DIR / "diebold_mariano.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _characteristic_groups(n_char: int) -> list[list[int]]:
    return [[j + k * n_char for k in range(9)] for j in range(n_char)]


def _sse_zeroed(theta, gram, cross, sum_yy, columns, centered: bool, sum_y: float, n: int) -> tuple[float, float]:
    def sse(coef):
        if centered:
            ybar = sum_y / n
            sst = sum_yy - n * ybar**2
            return sst - 2.0 * float(coef @ cross) + float(coef @ gram @ coef)
        return sum_yy - 2.0 * float(coef @ cross) + float(coef @ gram @ coef)

    full = sse(theta)
    reduced = theta.copy()
    reduced[columns] = 0.0
    return full, sse(reduced)


def _linear_importance(panel: Panel) -> None:
    """Average, over recursive training windows, the R² drop from zeroing a characteristic."""
    fit_dir = PREDICTION_DIR / "fits"
    groups = _characteristic_groups(panel.chars.shape[1])
    names = panel.char_names
    model_files = {
        "ols": False,
        "ols_huber": False,
        "enet": True,
        "enet_huber": True,
        "pls": True,
        "pcr": True,
    }
    totals = {name: np.zeros(len(names)) for name in model_files}
    counts = {name: 0 for name in model_files}
    years = range(1987, 2017)
    for year in years:
        needed = [name for name in model_files if (fit_dir / f"{name}_{year}.npz").exists()]
        if not needed:
            continue
        train_years, _val = panel.train_val_years(year)
        train_slices = panel.slices_for_years(train_years)
        mu, std = panel.macro_mean_std(train_slices)
        x_map = panel.x_by_month(train_slices, mu, std)
        moments = accumulate(panel.chars, panel.y, panel.sic, x_map, train_slices, panel.n_sic)
        scaled = standardize_moments(moments)
        sum_yy = float(moments["sum_yy"])
        for name in needed:
            blob = np.load(fit_dir / f"{name}_{year}.npz")
            theta = blob["theta"]
            centered = bool(int(blob["standardized"]))
            gram = scaled["gram"] if centered else moments["gram"]
            cross = scaled["cross"] if centered else moments["cross"]
            for j, columns in enumerate(groups):
                full, dropped = _sse_zeroed(
                    theta, gram, cross, sum_yy, columns, centered, float(moments["sum_y"]), int(moments["n"])
                )
                base = 1.0 - full / sum_yy
                alt = 1.0 - dropped / sum_yy
                totals[name][j] += base - alt
            counts[name] += 1
        print(f"importance {year}", flush=True)
    lines = ["characteristic," + ",".join(DISPLAY[name] for name in model_files)]
    rank_rows = []
    for j, characteristic in enumerate(names):
        values = []
        for name in model_files:
            values.append(totals[name][j] / max(counts[name], 1))
        rank_rows.append((characteristic, values))
        lines.append(characteristic + "," + ",".join(f"{value:.8f}" for value in values))
    (TABLE_DIR / "variable_importance.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Top 20 by the mean reduction across models that were estimated.
    present = [name for name in ("enet", "pls", "pcr", "enet_huber") if counts[name]]
    if not present:
        return
    score = np.zeros(len(names))
    for name in present:
        reduction = totals[name] / counts[name]
        score += np.maximum(reduction, 0.0)
    order = np.argsort(score)[::-1][:20]
    top_lines = ["rank,characteristic," + ",".join(DISPLAY[name] for name in present)]
    for rank, j in enumerate(order, start=1):
        cells = []
        for name in present:
            reduction = totals[name][j] / counts[name]
            share = reduction / max(np.maximum(totals[name] / counts[name], 0.0).sum(), 1e-12)
            cells.append(f"{100 * share:.4f}")
        top_lines.append(f"{rank},{names[j]}," + ",".join(cells))
    (TABLE_DIR / "variable_importance_top20.csv").write_text("\n".join(top_lines) + "\n", encoding="utf-8")
    _plot_importance(names, order, present, totals, counts)


def _plot_importance(names, order, present, totals, counts) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(order))
    height = 0.8 / max(len(present), 1)
    for i, name in enumerate(present):
        reduction = np.maximum(totals[name] / counts[name], 0.0)
        share = reduction / max(reduction.sum(), 1e-12)
        ax.barh(y_pos + i * height, [100 * share[j] for j in order], height=height, label=DISPLAY[name])
    ax.set_yticks(y_pos + height * (len(present) - 1) / 2)
    ax.set_yticklabels([names[j] for j in order])
    ax.invert_yaxis()
    ax.set_xlabel("Share of total R² reduction (%)")
    ax.set_title("Top 20 characteristics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "variable_importance_top20.png", dpi=150)
    plt.close(fig)


def _plot_r2(rows: list[dict]) -> None:
    labels = [row["model"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [row["all"] for row in rows], width, label="All")
    ax.bar(x, [row["top1000"] for row in rows], width, label="Top 1000")
    ax.bar(x + width, [row["bottom1000"] for row in rows], width, label="Bottom 1000")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Monthly out-of-sample R² (%)")
    ax.set_title("Recursive performance, 1987–2016")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "r2_oos.png", dpi=150)
    plt.close(fig)


def _plot_complexity() -> None:
    path = LOG_DIR / "linear_params.json"
    if not path.exists():
        return
    params = json.loads(path.read_text(encoding="utf-8"))
    years = list(range(1987, 2017))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    series = (
        ("enet", "nonzero", "ENet nonzero coefficients"),
        ("pls", "k", "PLS components"),
        ("pcr", "k", "PCR components"),
    )
    for ax, (model, key, title) in zip(axes, series):
        values = [params.get(model, {}).get(str(year), {}).get(key, np.nan) for year in years]
        ax.plot(years, values, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Test year")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_complexity.png", dpi=150)
    plt.close(fig)


def _summary(panel: Panel, rows: list[dict]) -> None:
    lines = [
        "# Out-of-sample results, 1987--2016",
        "",
        f"Stock-month observations: {panel.n:,}. Characteristics: {len(panel.char_names)}. Industry dummies: {panel.n_sic}.",
        "Mean excess return {:.3f} percent per month; standard deviation {:.1f} percent.".format(100 * panel.meta["excess_mean"], 100 * panel.meta["excess_std"]),
        "Share of rows with a missing SIC code: {:.1f} percent. Those rows receive a zero industry-dummy vector.".format(100 * panel.meta["share_missing_sic"]),
        "",
        "The R2 denominator is the sum of squared excess returns, without demeaning. Units are percent. The last column is the pooled estimate from Gu, Kelly, and Xiu (2020), Table 1.",
        "",
        "| Model | All | Top 1000 | Bottom 1000 | GKX |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        published = PUBLISHED.get(row["model"], "")
        published_text = f"{published:.2f}" if published != "" else "—"
        lines.append(
            f"| {row['model']} | {row['all']:.3f} | {row['top1000']:.3f} | {row['bottom1000']:.3f} | {published_text} |"
        )
    lines += [
        "",
        "OLS uses the raw 920 columns and no extra intercept. Elastic net, PLS, and PCR standardize columns inside each training window and add back the training mean of excess returns.",
        "Eight macro predictors are standardized in the training window and interacted with the 94 characteristics. OLS-3 uses size, book-to-market, and 12-month momentum, with an intercept.",
        "The Huber threshold is the 99.9 percent quantile of absolute training excess returns. Elastic-net lambda is chosen under squared error and the selected fit is then re-estimated with Huber weights.",
        "The GLM expands each of the 920 predictors with a three-knot quadratic spline and selects groups with a group lasso. The intercept is not penalized.",
        "",
        "Unrestricted OLS has an out-of-sample R2 near -101 percent, against -3.46 percent in the paper. The 920-column normal equations are ill-conditioned, and the prediction standard deviation is close to that of returns. OLS-3 and elastic net are close to the published 0.16 percent and 0.11 percent.",
        "Variable importance ranks characteristics by the R2 drop in elastic net, PLS, and PCR. Characteristics are the 2021 GKX release through 2016. Macro predictors are the Welch-Goyal monthly file updated through 2025.",
    ]
    (TABLE_DIR / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOG_DIR / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    panel = Panel()
    models = _available(panel)
    if not models:
        raise RuntimeError("No prediction files found.")
    rows = _write_r2(panel, models)
    _write_dm(panel, models)
    _plot_r2(rows)
    _plot_complexity()
    _linear_importance(panel)
    _summary(panel, rows)
    for row in rows:
        print(f"{row['model']:10} all={row['all']:.3f} top={row['top1000']:.3f} bot={row['bottom1000']:.3f}")


if __name__ == "__main__":
    main()
