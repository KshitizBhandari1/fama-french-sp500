# Point-in-Time Fama-French 3-Factor Proxy Engine (S&P 500 Universe)

An empirical replication study of the Fama-French 3-Factor model, focusing on constructing cross-sectional proxy factors entirely from scratch within a Point-in-Time (PIT) S&P 500 equity universe.

The project evaluates how factor premiums shift and tracking errors behave when constructing empirical risk factors using a high-quality, large-cap index constraint rather than the unconstrained institutional market universe (CRSP). The underlying regression parameters are resolved using an explicitly built closed-form Ordinary Least Squares (OLS) matrix solver.

---

## Motivation and Universe Alignment

Standard factor models are typically validated against the official Ken French Data Library, which covers the entire CRSP database (including micro-caps and highly illiquid small-cap stocks).

In practice, institutional mandates are frequently restricted to large, liquid investment horizons like the S&P 500. Attempting to run a classic Fama-French replication within this bounded universe creates a structural asset pricing mismatch:

* **The Universe Bounding Problem:** The official SMB benchmark isolates the size premium using an unconstrained CRSP universe split at the NYSE median market capitalization, capturing true micro-caps. Because this proxy is bound to the S&P 500 matrix, our median split classifies mid-caps as our "Small" portfolio slice, creating a structural factor definition mismatch.
* **The Quality Filter:** S&P 500 eligibility mandates consecutive quarters of positive net income, introducing an automatic financial quality screen that is absent in an unconstrained market sort.

The goal of this project is to evaluate how much factor variance is captured by an index-constrained proxy framework and pinpoint where specific market regimes cause these empirical factors to structurally decouple from the official benchmarks.

---

## Methodology and Engine Design
* **Universe:** Point-in-Time S&P 500 constituents (strictly mapping index entry/exit records to eliminate look-ahead and survivorship bias).
* **Rebalancing Frequency:** Monthly cross-sectional portfolio sorts.
* **Factor Sorting Mechanics:**
  * Size Factor Proxy (`smb`): Median market capitalization split.
  * Value Factor Proxy (`hml`): Top 30% / Bottom 30% sort based on Book-to-Market ratios sourced programmatically from SEC EDGAR filings.
* **Statistical Calibration:** Once the empirical monthly factor series are fully constructed, alpha, beta, and tracking standard errors are extracted using direct linear algebra:

$$\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T Y$$

---

## Core Factor Performance Diagnostics (132 Months)

The empirical results over a 132-month synchronized tracking horizon highlight the exact tracking efficiency of our S&P 500 proxy factor engine:

| Diagnostic Metric | Market Excess Return (`mkt-rf`) | Size Premium (`smb`) | Value Premium (`hml`) |
| :--- | :---: | :---: | :---: |
| **Pearson Correlation** | 0.9826 | 0.5107 | 0.8503 |
| **Coefficient of Determination ($R^2$)** | 96.55% | 26.08% | 72.30% |
| **Factor Beta ($\beta \pm SE$)** | 0.943 $\pm$ 0.016 | 0.308 $\pm$ 0.046 | 0.725 $\pm$ 0.039 |
| **Idiosyncratic Alpha ($\alpha \pm SE$)** | -0.015\% $\pm$ 0.072\% | -0.154\% $\pm$ 0.125\% | 0.028\% $\pm$ 0.146\% |
| **Residual Standard Error (RSE)** | 0.80% | 1.43% | 1.68% |

---

## Portfolio Attribution and Methodological Insights

### 1. Market Excess Return Alignment
The high tracking profile ($R^2 = 96.55\%$) verifies that our independently constructed capitalization-weighted asset processing mirrors the market benchmark. The minor under-unity beta ($0.943$) is a structural footprint of the large-cap environment: the proxy misses the high-beta micro-cap tail that frequently amplifies official benchmark returns during highly speculative expansions.

### 2. The SMB Structural Disconnect
The low correlation ($0.5107$) and weak $R^2$ ($26.08\%$) represent an **economically accurate boundary result**. While the official benchmark uses a strict NYSE median market capitalization to prevent small-cap sorting distortion from NASDAQ listings, our engine is restricted to the S&P 500. Consequently, our "smb" portfolio is fundamentally a "Large Mid-Cap minus Mega-Cap" factor. This underscores that hard capitalization bounds fundamentally disrupt the baseline definition of size premiums.

### 3. High-Fidelity Value Capture
The engine successfully reproduces the value premium ($R^2 = 72.30\%$). This validates the fundamental accounting data pipeline, indicating that balance sheet timing loops effectively synchronize with cross-sectional asset pricing cycles, even when restricted to the top tier of US equity capitalizations.

---

## Analysis of Visual Trends and Empirical Hypotheses

### 1. Cumulative Growth Trajectories
* **`mkt-rf` Divergence (Late-2021):** The proxy factor exhibits visible underperformance during the late-2021 liquidity surge. A plausible structural hypothesis is that the proxy engine filters out the highly speculative, non-index micro-cap assets that drove broader market returns during that window. Conversely, it provides some relative capital insulation during the subsequent 2022 normalization.

* **`smb` Regime Shielding (2022):** Official SMB experiences a severe drawdown in 2022, while our proxy factor maintains a horizontal trajectory. This divergence highlights a structural vulnerability mismatch: the official SMB factor was exposed to highly leveraged micro-caps vulnerable to rising borrowing costs, whereas our S&P 500 proxy isolated larger, institutional-grade mid-caps characterized by robust interest coverage.

* **`hml` 2017 Decoupling:** Official HML collapsed during the 2017 mega-cap tech expansion while the proxy held steady. This indicates that the proxy's large-cap value definitions captured structural energy and financial operators that benefited distinctively from corporate tax cuts.

### 2. OLS Regression Scatter Matrix
* **Market Excess:** Extremely tight, low-variance cluster confirming high-fidelity beta capture.
* **Size Premium:** Highly diffuse scatter plot ($RSE = 1.43\%$) reflecting that the size factor behaves differently when stripped of micro-cap stocks.
* **Value Premium:** Highly concentrated clustering with a small positive alpha intercept ($0.028\%$). Because the S&P 500 eligibility rules screens out highly distressed firms, this engine may structurally filter out classic "value traps," capturing an implicit high-quality profitability premium.

### 3. Rolling 12-Month Correlations
* **`mkt-rf` (Mean 0.98):** Singular drop to $\sim 0.91$ at the start of 2018. This suggests a period where institutions temporarily treated large-caps as hyper-liquid sources of cash amid Fed's October 2017 QT program.
* **`smb` (Regime Disruptions):** Wild fluctuations dropping to zero on two major occasions, demonstrating that during periods of macro stress, mid-cap returns distributions become completely orthogonal to that of micro-caps.
* **`hml` 2014-2016 Commodity Pivot:** Rolling 12-month correlation entered a severe downtrend, bottoming out near zero in early 2016. This historical decoupling reflects a major structural test: while the official unconstrained benchmark became swamped by highly leveraged, distressed, micro-cap energy operators, our engine's S&P 500 quality threshold effectively insulated the value portfolio, preserving alignment with stable, cash-generating operators. 

---

## Operational Constraints and Data Limitations
* **Survivorship Defenses:** Portfolios are dynamically remapped monthly based on exact historical index entry/exit records across a total cohort of **754 tickers**.
* **Data Provisioning Constraints:** Due to open-source pricing API limits on delisted tickers, **139 tickers** were structurally dropped during pricing extraction from `yfinance`. Historical accounting metric collection via SEC EDGAR successfully recovered **538 stocks**, with **77 tickers** skipped during the fundamental statement processing stage.
* **Engine Portability:** Proxy factor replication engine logic is entirely decoupled from the data provider. Connecting this engine to an institutional database cluster (CRSP/Compustat) using the exact same PIT mapping dictionary interface will function completely bias-free with **zero code modifications**.
