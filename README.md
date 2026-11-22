# Replication of Gu, Kelly, and Xiu (2020)

Monthly stock-level excess-return forecasts for March 1957--December 2016. The out-of-sample window is 1987--2016. Each year the training sample grows by one year from 1957, and the validation window is the preceding 12 years. The test year is not used for estimation or tuning.

## Data

Place the inputs under `dataset/` before running:

- `dataset/GKX_20201231.csv`: 94 lagged firm characteristics. The return on a row is the forecast target.
- `dataset/goyal/PredictorData2025.xlsx`: updated monthly Welch and Goyal (2008) series. `macro.py` builds `dp`, `ep`, `bm`, `ntis`, `tbl`, `tms`, `dfy`, and `svar`, and checks the January 1950 term spread and dividend yield.

The excess return is `RET` minus the same month's Treasury-bill return. Macro predictors are lagged one month and standardized inside each training window. Characteristics are filled with the monthly cross-sectional median and ranked to \([-1,1]\). They are interacted with a constant and the eight macro predictors. Seventy-four SIC industry dummies are appended and are not interacted, giving 920 columns. A stock with a missing industry code gets a zero dummy vector.

## Run

From this directory:

```powershell
$env:PYTHONPATH = "src"
python -m gkx.macro
python -m gkx.preprocess
python -m gkx.synthetic_check
python -m gkx.run_linear
python -m gkx.run_glm
python -m gkx.make_results
```

Add `--year 1987` to fit a single test year. Completed years are skipped.

`python -m gkx.pls_probe` refits PLS for 2000 and 2010 under three scalings. It does not overwrite the stored predictions.

## Models

- OLS and OLS with Huber loss: all 920 columns, no extra intercept.
- OLS-3 and OLS-3 with Huber loss: `mvel1`, `bm`, and `mom12m`, with an intercept.
- Elastic net: \(\rho=0.5\). \(\lambda\) minimizes validation squared error. The Huber fit re-estimates that \(\lambda\) with Huber weights.
- PLS and PCR: the number of components minimizes validation squared error. Columns are standardized in the training window, and predictions add back the training mean of excess returns.
- GLM and GLM with Huber loss: each predictor is expanded as \(z,(z-c_1)^2,(z-c_2)^2,(z-c_3)^2\), with knots at the training-sample 25th, 50th, and 75th percentiles. A group lasso keeps or drops the four terms together. The intercept is not penalized. \(\lambda\) is chosen under squared error; the Huber fit uses that \(\lambda\).

## Results

`python -m gkx.make_results` writes:

- `outputs/tables/r2_oos.csv`: monthly out-of-sample \(R^2\) (percent) for all stocks, the top 1,000, and the bottom 1,000 by market equity
- `outputs/tables/diebold_mariano.csv`
- `outputs/tables/variable_importance_top20.csv`
- `outputs/figures/r2_oos.png` and `variable_importance_top20.png`
- `outputs/tables/results_summary.md`

The \(R^2\) denominator is the sum of squared excess returns, without subtracting the mean.
