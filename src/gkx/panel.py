"""Load the processed panel and the monthly macro predictors."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from gkx.config import load_config
from gkx.paths import GOYAL_PARQUET, PROCESSED_DIR


def previous_month(yyyymm: int) -> int:
    year, month = divmod(int(yyyymm), 100)
    if month == 1:
        return (year - 1) * 100 + 12
    return int(yyyymm) - 1


class Panel:
    def __init__(self):
        self.permno = np.load(PROCESSED_DIR / "permno.npy")
        self.yyyymm = np.load(PROCESSED_DIR / "yyyymm.npy")
        self.y = np.load(PROCESSED_DIR / "y.npy")
        self.me = np.load(PROCESSED_DIR / "me.npy")
        self.sic = np.load(PROCESSED_DIR / "sic.npy")
        self.chars = np.load(PROCESSED_DIR / "chars.npy", mmap_mode="r")
        self.meta = json.loads((PROCESSED_DIR / "meta.json").read_text(encoding="utf-8"))
        self.char_names = self.meta["characteristics"]
        self.n_sic = len(self.meta["sic_levels"])
        self.n = int(self.yyyymm.shape[0])
        self.slices = self._month_slices()
        self.by_year: dict[int, list[tuple[int, int, int]]] = {}
        for yyyymm, start, end in self.slices:
            self.by_year.setdefault(int(yyyymm) // 100, []).append((int(yyyymm), start, end))
        macro = pl.read_parquet(GOYAL_PARQUET).sort("yyyymm")
        names = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
        self.macro_months = macro["yyyymm"].to_numpy().astype(np.int32)
        self.macro = macro.select(names).to_numpy().astype(np.float64)
        self.macro_index = {int(month): i for i, month in enumerate(self.macro_months)}

    def _month_slices(self) -> list[tuple[int, int, int]]:
        months = self.yyyymm
        cuts = np.flatnonzero(months[1:] != months[:-1]) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, months.shape[0]]
        return [(int(months[start]), int(start), int(end)) for start, end in zip(starts, ends)]

    def slices_for_years(self, years) -> list[tuple[int, int, int]]:
        out = []
        for year in years:
            out.extend(self.by_year.get(int(year), []))
        return out

    def train_val_years(self, test_year: int) -> tuple[list[int], list[int]]:
        cfg = load_config()["sample"]
        val_len = int(cfg["val_years"])
        train_start = int(cfg["train_start_year"])
        val_years = list(range(test_year - val_len, test_year))
        train_years = list(range(train_start, test_year - val_len))
        return train_years, val_years

    def macro_mean_std(self, slices) -> tuple[np.ndarray, np.ndarray]:
        months = []
        seen = set()
        for yyyymm, _, _ in slices:
            lagged = previous_month(yyyymm)
            if lagged not in seen:
                seen.add(lagged)
                months.append(self.macro[self.macro_index[lagged]])
        matrix = np.vstack(months)
        mu = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mu, std

    def x_by_month(self, slices, mu: np.ndarray, std: np.ndarray) -> dict[int, np.ndarray]:
        out = {}
        for yyyymm, _, _ in slices:
            if yyyymm in out:
                continue
            lagged = self.macro[self.macro_index[previous_month(yyyymm)]]
            out[yyyymm] = np.concatenate(([1.0], (lagged - mu) / std))
        return out
