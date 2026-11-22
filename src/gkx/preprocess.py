"""Filter the GKX panel, rank characteristics, and store contiguous arrays.

Rows are stock-months from March 1957 through December 2016. Characteristics
on a row are already lagged in the source file, so RET on that row is the
forecast target. The risk-free rate is the same month's Treasury-bill return.
The eight macro predictors are not merged here; they are lagged at estimation
time so each training window can be standardized on its own.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from gkx.paths import AUDIT_DIR, GKX_CSV, GOYAL_PARQUET, PROCESSED_DIR

NON_FEATURES = {"permno", "DATE", "RET", "prc", "SHROUT", "mve0", "sic2"}


def characteristic_names() -> list[str]:
    schema = pl.scan_csv(GKX_CSV, n_rows=0).collect_schema().names()
    names = [name for name in schema if name not in NON_FEATURES]
    if len(names) != 94:
        raise RuntimeError(f"Expected 94 characteristics, found {len(names)}: {names}")
    for required in ("mvel1", "bm", "mom12m"):
        if required not in names:
            raise RuntimeError(f"Missing OLS-3 characteristic {required}")
    return names


def _rank_frame(frame: pl.DataFrame, chars: list[str]) -> pl.DataFrame:
    cleaned = [
        pl.when(pl.col(name).is_infinite()).then(None).otherwise(pl.col(name)).alias(name)
        for name in chars
    ]
    frame = frame.with_columns(cleaned)
    frame = frame.with_columns(
        [
            pl.col(name).fill_null(pl.col(name).median().over("yyyymm")).fill_null(0.0).alias(name)
            for name in chars
        ]
    )
    count = pl.len().over("yyyymm")
    frame = frame.with_columns(
        [
            pl.when(count <= 1)
            .then(0.0)
            .otherwise((pl.col(name).rank(method="average").over("yyyymm") - 1) / (count - 1) * 2 - 1)
            .alias(name)
            for name in chars
        ]
    )
    return frame


def build_panel(start_yyyymm: int = 195703, end_yyyymm: int = 201612) -> dict:
    chars = characteristic_names()
    start_date = start_yyyymm * 100 + 1
    end_date = end_yyyymm * 100 + 31
    frame = (
        pl.scan_csv(GKX_CSV, infer_schema_length=10000)
        .select(["permno", "DATE", "RET", "sic2", *chars])
        .filter((pl.col("DATE") >= start_date) & (pl.col("DATE") <= end_date))
        .with_columns(
            [
                (pl.col("DATE") // 100).alias("yyyymm"),
                pl.col("sic2").cast(pl.Float64, strict=False).cast(pl.Int32, strict=False),
                *[pl.col(name).cast(pl.Float64, strict=False) for name in chars],
            ]
        )
        .filter(pl.col("RET").is_not_null() & pl.col("RET").is_finite())
        .with_columns(pl.col("mvel1").alias("me"))
        .collect(engine="streaming")
    )
    n_before = frame.height
    macro = pl.read_parquet(GOYAL_PARQUET).select(["yyyymm", "rfree"])
    frame = frame.join(macro, on="yyyymm", how="left")
    missing_rf = int(frame["rfree"].null_count())
    frame = frame.filter(pl.col("rfree").is_not_null())
    frame = frame.with_columns((pl.col("RET") - pl.col("rfree")).alias("excess"))
    frame = _rank_frame(frame, chars)
    frame = frame.sort(["yyyymm", "permno"])

    sic_values = (
        frame.filter(pl.col("sic2").is_not_null())
        .select(pl.col("sic2").cast(pl.Int32))
        .unique()
        .sort("sic2")["sic2"]
        .to_list()
    )
    sic_map = {int(code): i for i, code in enumerate(sic_values)}
    sic = np.full(frame.height, -1, dtype=np.int16)
    raw_sic = frame["sic2"].to_numpy()
    for code, index in sic_map.items():
        sic[raw_sic == code] = index

    yyyymm = frame["yyyymm"].to_numpy().astype(np.int32)
    payload = {
        "permno": frame["permno"].to_numpy().astype(np.int32),
        "yyyymm": yyyymm,
        "y": frame["excess"].to_numpy().astype(np.float32),
        "me": frame["me"].to_numpy().astype(np.float32),
        "sic": sic,
        "chars": frame.select(chars).to_numpy().astype(np.float32),
    }
    for key, value in payload.items():
        np.save(PROCESSED_DIR / f"{key}.npy", value)
    meta = {
        "characteristics": chars,
        "sic_levels": [int(code) for code in sic_values],
        "n_rows": int(frame.height),
        "n_dropped_missing_rfree": missing_rf,
        "n_rows_before_rfree_filter": n_before,
        "start_yyyymm": int(yyyymm[0]),
        "end_yyyymm": int(yyyymm[-1]),
        "excess_mean": float(payload["y"].mean()),
        "excess_std": float(payload["y"].std()),
        "share_missing_sic": float(np.mean(sic < 0)),
        "rank_min": float(np.nanmin(payload["chars"])),
        "rank_max": float(np.nanmax(payload["chars"])),
        "rank_mean_abs": float(np.mean(np.abs(payload["chars"].mean(axis=0)))),
    }
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (AUDIT_DIR / "panel.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    meta = build_panel()
    print(json.dumps(meta, indent=2))
    if not 0.0 < meta["excess_mean"] < 0.03:
        raise RuntimeError(f"Excess-return mean looks wrong: {meta['excess_mean']}")
    if meta["rank_min"] < -1.01 or meta["rank_max"] > 1.01:
        raise RuntimeError("Ranked characteristics left [-1, 1].")


if __name__ == "__main__":
    main()
