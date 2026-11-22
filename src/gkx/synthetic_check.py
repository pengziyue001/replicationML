"""Check the moment algebra and the linear solvers against dense implementations."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression

from gkx.moments import accumulate
from gkx.solvers import fit_enet, fit_ols_raw, fit_pcr_path, fit_pls_path, predict_raw


def _dense_design(chars, sic, x_by_row, n_sic):
    n, n_char = chars.shape
    n_macro = x_by_row.shape[1]
    z = np.zeros((n, n_macro * n_char + n_sic))
    for i in range(n):
        z[i, : n_macro * n_char] = np.kron(x_by_row[i], chars[i])
        if sic[i] >= 0:
            z[i, n_macro * n_char + sic[i]] = 1.0
    return z


def _panel():
    rng = np.random.default_rng(7)
    n_char = 4
    n_sic = 3
    months = np.array([202001, 202002, 202003, 202004])
    counts = [30, 25, 40, 20]
    yyyymm = np.repeat(months, counts)
    chars = rng.normal(size=(yyyymm.size, n_char))
    sic = rng.integers(0, n_sic, size=yyyymm.size)
    sic[:3] = -1
    macro = rng.normal(size=(len(months), 2))
    x_month = {int(m): np.concatenate([[1.0], macro[i]]) for i, m in enumerate(months)}
    x_row = np.vstack([np.repeat(x_month[int(m)][None, :], c, axis=0) for m, c in zip(months, counts)])
    y = chars @ rng.normal(size=n_char) + x_row[:, 1] * 0.3 + rng.normal(scale=0.1, size=yyyymm.size)
    slices = []
    start = 0
    for month, count in zip(months, counts):
        slices.append((int(month), start, start + count))
        start += count
    return chars, y, sic, x_month, slices, n_sic, x_row


def main():
    chars, y, sic, x_month, slices, n_sic, x_row = _panel()
    moments = accumulate(chars, y, sic, x_month, slices, n_sic)
    dense = _dense_design(chars, sic, x_row, n_sic)
    assert np.allclose(moments["gram"], dense.T @ dense, atol=1e-8)
    assert np.allclose(moments["cross"], dense.T @ y, atol=1e-8)
    assert np.allclose(moments["column_sum"], dense.sum(axis=0), atol=1e-8)
    assert moments["n"] == len(y)

    ols = fit_ols_raw(moments)
    yhat = np.concatenate(
        [
            predict_raw(
                chars[s:e],
                sic[s:e],
                x_month[m],
                ols["mu"],
                ols["scale"],
                ols["theta"],
                intercept=0.0,
                standardized=False,
            )
            for m, s, e in slices
        ]
    )
    reference = LinearRegression(fit_intercept=False).fit(dense, y).predict(dense)
    # Standardization with an intercept-equivalent column (the constant macro
    # times a characteristic) spans the same column space, so fitted values match.
    ols_gap = np.max(np.abs(yhat - reference))
    print("ols_gap", ols_gap)
    assert ols_gap < 1e-6

    # PLS. sklearn centers; our moments are centered by standardization.
    # Compare on already-standardized columns so both solve the same problem.
    scale = dense.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    dense_s = (dense - dense.mean(axis=0)) / scale
    pls = fit_pls_path(moments, [1, 2, 3])
    for k in (1, 2, 3):
        pred = np.concatenate(
            [
                predict_raw(
                    chars[s:e],
                    sic[s:e],
                    x_month[m],
                    pls["mu"],
                    pls["scale"],
                    pls["thetas"][k],
                    intercept=pls["intercept"],
                )
                for m, s, e in slices
            ]
        )
        sk = PLSRegression(n_components=k, scale=False)
        sk.fit(dense_s, y - y.mean())
        sk_pred = sk.predict(dense_s).ravel() + y.mean()
        gap = np.max(np.abs(pred - sk_pred))
        print("pls_gap", k, gap)
        assert gap < 1e-5

    pcr = fit_pcr_path(moments, [1, 2, 4])
    for k in (1, 2, 4):
        pred = np.concatenate(
            [
                predict_raw(
                    chars[s:e],
                    sic[s:e],
                    x_month[m],
                    pcr["mu"],
                    pcr["scale"],
                    pcr["thetas"][k],
                    intercept=pcr["intercept"],
                )
                for m, s, e in slices
            ]
        )
        pca = PCA(n_components=k)
        scores = pca.fit_transform(dense_s)
        sk_pred = LinearRegression().fit(scores, y).predict(scores)
        gap = np.max(np.abs(pred - sk_pred))
        print("pcr_gap", k, gap)
        assert gap < 1e-5

    # sklearn minimizes (1/2n) SSE + alpha * l1_ratio * L1 + 0.5 * alpha * (1-l1_ratio) * L2.
    # That is our objective at the same lambda, with rho = 1 - l1_ratio.
    for lam in (1e-3, 1e-2, 1e-1):
        en = fit_enet(moments, lam=lam, rho=0.5, max_iter=4000, tol=1e-10)
        pred = np.concatenate(
            [
                predict_raw(
                    chars[s:e],
                    sic[s:e],
                    x_month[m],
                    en["mu"],
                    en["scale"],
                    en["theta"],
                    intercept=en["intercept"],
                )
                for m, s, e in slices
            ]
        )
        sk = ElasticNet(alpha=lam, l1_ratio=0.5, fit_intercept=False, max_iter=20000, tol=1e-12)
        sk.fit(dense_s, y - y.mean())
        sk_pred = sk.predict(dense_s) + y.mean()
        gap = np.max(np.abs(pred - sk_pred))
        print("enet_gap", lam, gap, "nonzero", en["nonzero"])
        assert gap < 1e-4

    print("synthetic checks passed")


if __name__ == "__main__":
    main()
