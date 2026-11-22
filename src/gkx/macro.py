"""Build the eight Welch–Goyal macroeconomic predictors.

The official monthly workbook contains the raw inputs (not the transformed
predictors). Definitions follow Welch and Goyal (2008) and the replication
note that reconstructs the same series:

    dp  = log(D12) - log(Index)
    ep  = log(E12) - log(Index)
    bm  = b/m
    ntis, tbl, svar are taken from the file
    tms = lty - tbl
    dfy = BAA - AAA

`Rfree` is the monthly Treasury-bill return and is kept separate so it can be
subtracted from the same month's stock return. The eight predictors are lagged
one month at merge time: the end-of-month-t value forecasts the month-t+1 return.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from gkx.paths import GOYAL_PARQUET, GOYAL_XLSX

MACRO_COLUMNS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "nan", "none"}:
            return None
        return float(value)
    return float(value)


def build_monthly_predictors(xlsx_path=GOYAL_XLSX, out_path=GOYAL_PARQUET) -> pl.DataFrame:
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook["Monthly"]
    rows = sheet.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    index = {name: i for i, name in enumerate(header)}
    records = []
    for row in rows:
        yyyymm = row[index["yyyymm"]]
        if yyyymm is None:
            continue
        records.append(
            {
                "yyyymm": int(yyyymm),
                "index_level": _to_float(row[index["Index"]]),
                "d12": _to_float(row[index["D12"]]),
                "e12": _to_float(row[index["E12"]]),
                "bm": _to_float(row[index["b/m"]]),
                "tbl": _to_float(row[index["tbl"]]),
                "aaa": _to_float(row[index["AAA"]]),
                "baa": _to_float(row[index["BAA"]]),
                "lty": _to_float(row[index["lty"]]),
                "ntis": _to_float(row[index["ntis"]]),
                "rfree": _to_float(row[index["Rfree"]]),
                "svar": _to_float(row[index["svar"]]),
                "spvw": _to_float(row[index["CRSP_SPvw"]]),
            }
        )
    workbook.close()
    frame = pl.DataFrame(
        records,
        schema={
            "yyyymm": pl.Int32,
            "index_level": pl.Float64,
            "d12": pl.Float64,
            "e12": pl.Float64,
            "bm": pl.Float64,
            "tbl": pl.Float64,
            "aaa": pl.Float64,
            "baa": pl.Float64,
            "lty": pl.Float64,
            "ntis": pl.Float64,
            "rfree": pl.Float64,
            "svar": pl.Float64,
            "spvw": pl.Float64,
        },
    ).sort("yyyymm")
    frame = frame.with_columns(
        [
            pl.when((pl.col("d12") > 0) & (pl.col("index_level") > 0))
            .then(pl.col("d12").log() - pl.col("index_level").log())
            .otherwise(None)
            .alias("dp"),
            pl.when((pl.col("e12") > 0) & (pl.col("index_level") > 0))
            .then(pl.col("e12").log() - pl.col("index_level").log())
            .otherwise(None)
            .alias("ep"),
            (pl.col("lty") - pl.col("tbl")).alias("tms"),
            (pl.col("baa") - pl.col("aaa")).alias("dfy"),
        ]
    )
    # Dividend yield uses the lagged price index. Used only as an audit check.
    frame = frame.with_columns(
        pl.when((pl.col("d12") > 0) & (pl.col("index_level").shift(1) > 0))
        .then(pl.col("d12").log() - pl.col("index_level").shift(1).log())
        .otherwise(None)
        .alias("dy")
    )
    keep = ["yyyymm", "rfree", "dy", "spvw", *MACRO_COLUMNS]
    out = frame.select(keep)
    out.write_parquet(out_path)
    return out


def audit_predictors(frame: pl.DataFrame) -> dict:
    """Check a published numerical anchor and the scale of Rfree."""
    row = frame.filter(pl.col("yyyymm") == 195001)
    if row.height != 1:
        raise RuntimeError("Goyal monthly file has no January 1950 row.")
    tms = float(row["tms"][0])
    dy = float(row["dy"][0])
    rfree = frame.filter((pl.col("yyyymm") >= 195703) & (pl.col("yyyymm") <= 201612))["rfree"]
    rfree_med = float(rfree.drop_nulls().median())
    report = {
        "tms_195001": tms,
        "dy_195001": dy,
        "rfree_median_195703_201612": rfree_med,
        "tms_anchor_ok": abs(tms - 0.0108) < 2e-4,
        "dy_anchor_ok": abs(dy - (-2.68)) < 0.05,
        "rfree_is_monthly_decimal": 0 < rfree_med < 0.02,
    }
    if not report["tms_anchor_ok"] or not report["rfree_is_monthly_decimal"]:
        raise RuntimeError(f"Goyal predictor audit failed: {report}")
    return report


def main():
    frame = build_monthly_predictors()
    report = audit_predictors(frame)
    sample = frame.filter((pl.col("yyyymm") >= 195702) & (pl.col("yyyymm") <= 201612))
    nulls = {name: int(sample[name].null_count()) for name in ["rfree", *MACRO_COLUMNS]}
    print("wrote", GOYAL_PARQUET)
    print("audit", report)
    print("nulls_195702_201612", nulls)
    print(sample.select(["yyyymm", *MACRO_COLUMNS, "rfree"]).describe())


if __name__ == "__main__":
    main()
