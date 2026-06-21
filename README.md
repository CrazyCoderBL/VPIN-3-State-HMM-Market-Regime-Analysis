# VPIN + 3-State HMM Market Regime Analysis

This project uses a **3-state Hidden Markov Model (HMM)** and **VPIN (Volume-Synchronized Probability of Informed Trading)** to analyse market behaviour and identify whether the market is in a relatively calm, uncertain, or high-risk condition.

The main purpose is to study whether taking a cautious approach during periods of elevated volatility and market toxicity could help retain returns compared with remaining fully exposed throughout all market conditions.

The notebook is demonstrated using daily market data for AAPL.

---

## Objective

Financial markets do not behave the same way at all times. There are periods where price movement is stable and trends are clearer, periods where the market is uncertain and volatile, and periods where risk increases sharply.

This notebook attempts to:

* Detect hidden market regimes using returns and volatility.
* Measure possible market toxicity using VPIN.
* Classify market conditions into:

  * **Calm / Relatively Bullish**
  * **Choppy / Uncertain**
  * **Crisis / High Risk**
* Analyse whether reducing exposure during unstable periods could help preserve returns.
* Compare regime-aware returns with a fully invested baseline.

---

## Core Idea

The project combines two signals:

### 1. 3-State Hidden Markov Model

A Gaussian Hidden Markov Model is used to learn hidden market states from:

* Daily log returns
* Rolling volatility

The three states are then ordered according to their average volatility:

| State              | Interpretation                   |
| ------------------ | -------------------------------- |
| Lowest volatility  | Calm / relatively bullish market |
| Medium volatility  | Choppy / uncertain market        |
| Highest volatility | Crisis / high-risk market        |

These labels are analytical interpretations based mainly on volatility. A calm state does not always mean the market will move upward, and a crisis state does not always mean that returns will be negative.

### 2. VPIN Market-Toxicity Measure

VPIN estimates the imbalance between inferred buying and selling volume. A high VPIN value may indicate that market activity is becoming more one-sided or uncertain.

VPIN is used as an additional caution signal alongside the HMM regime classification rather than as a standalone prediction model.

---

## Regime-Aware Exposure Analysis

Instead of treating every market day equally, the notebook studies whether reducing exposure during riskier conditions could help protect capital.

| Detected Condition        | Exposure Used for Analysis |
| ------------------------- | -------------------------: |
| Calm / relatively bullish |                       100% |
| Calm with elevated VPIN   |                        50% |
| Choppy / uncertain        |                        50% |
| Choppy with elevated VPIN |                         0% |
| Crisis / high risk        |                         0% |

The final exposure is shifted by one day before calculating returns. This avoids using information from the same day to make a decision for that day.

The analysis is intended to answer the following question:

> If exposure was reduced when the market appeared unstable or toxic, how much return could have been retained compared with staying fully invested?

---

## Methodology

### Data Collection

Historical daily OHLCV data is downloaded using `yfinance`.

```python
df = yf.download("AAPL", start="2010-01-01", progress=False)
```

### Feature Engineering

The notebook calculates:

* Log returns
* 21-day rolling volatility
* Return z-scores
* Estimated buy volume
* Estimated sell volume
* Volume imbalance
* VPIN score

Buy and sell volume are estimated using the normalized return distribution:

```python
df["Buy_Factor"] = norm.cdf(df["Z_Score"])
df["V_Buy"] = df["Volume"] * df["Buy_Factor"]
df["V_Sell"] = df["Volume"] * (1 - df["Buy_Factor"])
```

VPIN is calculated using a rolling 21-day window:

```python
df["VPIN"] = (
    df["Volume_Imbalance"].rolling(window=21).sum()
    / df["Volume"].rolling(window=21).sum()
)
```

### HMM Training and Evaluation

The 3-state Gaussian HMM is trained using daily returns and rolling volatility as input features.

The model is trained on historical data from **2010 to 2019**. The year **2020** is excluded from the analysis because of the unusually extreme market behaviour during the COVID period. The model is then evaluated on data from **2021 onward**.

The hidden states identified by the HMM are sorted based on their average volatility and interpreted as calm, choppy, and crisis market conditions.

The evaluation compares two cases:

1. **Full Exposure**
   Remaining invested throughout the evaluation period.

2. **Regime-Aware Exposure**
   Reducing or removing exposure during periods classified as uncertain or high risk, especially when VPIN is elevated.
