# quant-perf-measures
A comprehensive list of quantitative finance portfolio/strategy performance measures.

# Quantitative Finance Performance Measures with Explanation and Implementation

## 1. Total Return

The total return measures the overall performance of an investment over a specific period from t=0 to t=T.

$$\text{Total Return} = \prod_{t=1}^{T} (1 + r_t) - 1$$

Where:
- $r_t$ is the return at time $t$
- $T$ is the total number of periods

```python
import numpy as np

def total_return(returns):
    return np.prod(1 + returns) - 1
```

## 2. Annualized Return

The annualized return normalizes the total return to a one-year period, allowing for comparison of investments held for different lengths of time.

$$\text{Annualized Return} = (1 + \text{Total Return})^{\frac{252}{T}} - 1$$

Where:
- 252 is the typical number of trading days in a year (for daily returns)
- $T$ is the total number of periods

```python
def annualized_return(returns, periods_per_year):
    total_return = np.prod(1 + returns) - 1
    return (1 + total_return) ** (periods_per_year / len(returns)) - 1
```

## 3. Sharpe Ratio

The Sharpe ratio measures the excess return (or risk premium) per unit of deviation in an investment asset or a trading strategy. Keep in mind that Sharpe Ratio is **always annulized**.

$$\text{Sharpe Ratio} = \frac{E[R_p - R_f]}{\sigma_p} = \frac{\overline{R_p - R_f}}{\sqrt{\text{Var}[R_p - R_f]}}$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate
- $E[R_p - R_f]$ is the expected value of the excess return
- $\sigma_p$ is the standard deviation of the portfolio's excess return

```python
def sharpe_ratio(returns, risk_free_rate, periods_per_year):
    excess_returns = returns - risk_free_rate
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
```

## 4. Sortino Ratio

The Sortino ratio is a variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility by using the asset's standard deviation of negative portfolio returns.

$$\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d}$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate
- $\sigma_d$ is the standard deviation of negative portfolio returns

```python
def sortino_ratio(returns, risk_free_rate, periods_per_year):
    excess_returns = returns - risk_free_rate
    downside_returns = np.minimum(excess_returns, 0)
    return np.sqrt(periods_per_year) * excess_returns.mean() / np.std(downside_returns)
```

## 5. Maximum Drawdown

Maximum Drawdown (MDD) measures the largest peak-to-trough decline in the cumulative returns of a portfolio.

$$\text{MDD} = \min_{t \in (0,T)} \left( \frac{\text{Peak Value}_t - \text{Trough Value}_t}{\text{Peak Value}_t} \right)$$

Where:
- $T$ is the total number of periods
- Peak Value is the highest cumulative return achieved
- Trough Value is the lowest cumulative return after the peak

```python
def maximum_drawdown(returns):
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return np.min(drawdown)
```

## 6. Calmar Ratio

The Calmar ratio is a risk-adjusted performance measure that relates the average annual compounded rate of return to the maximum drawdown.

$$\text{Calmar Ratio} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$

```python
def calmar_ratio(returns, periods_per_year):
    return annualized_return(returns, periods_per_year) / abs(maximum_drawdown(returns))
```

## 7. Information Ratio

The Information Ratio measures a portfolio manager's ability to generate excess returns relative to a benchmark, but also attempts to identify the consistency of the investor.

$$\text{Information Ratio} = \frac{R_p - R_b}{\sigma_{R_p - R_b}}$$

Where:
- $R_p$ is the return of the portfolio
- $R_b$ is the return of the benchmark
- $\sigma_{R_p - R_b}$ is the standard deviation of the excess return

```python
def information_ratio(returns, benchmark_returns, periods_per_year):
    active_returns = returns - benchmark_returns
    return np.sqrt(periods_per_year) * active_returns.mean() / active_returns.std()
```

## 8. Alpha (CAPM)

Alpha represents the active return on an investment, gauging the performance of an investment against a market index or benchmark that is considered to represent the market's movement as a whole.

$$\alpha = R_p - [R_f + \beta(R_m - R_f)]$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate
- $\beta$ is the beta of the portfolio
- $R_m$ is the return of the market

```python
def alpha(returns, benchmark_returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    excess_benchmark_returns = benchmark_returns - risk_free_rate
    beta = np.cov(excess_returns, excess_benchmark_returns)[0, 1] / np.var(excess_benchmark_returns)
    return excess_returns.mean() - beta * excess_benchmark_returns.mean()
```

## 9. Beta

Beta is a measure of the volatility, or systematic risk, of a security or portfolio in comparison to the market as a whole.

$$\beta = \frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}$$

Where:
- $R_p$ is the return of the portfolio
- $R_m$ is the return of the market

```python
def beta(returns, benchmark_returns):
    return np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
```

## 10. Treynor Ratio

The Treynor ratio, also known as the reward-to-volatility ratio, measures the returns earned in excess of that which could have been earned on a riskless investment per unit of market risk assumed.

$$\text{Treynor Ratio} = \frac{R_p - R_f}{\beta}$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate
- $\beta$ is the beta of the portfolio

```python
def treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year):
    excess_returns = returns - risk_free_rate
    beta_val = beta(returns, benchmark_returns)
    return np.sqrt(periods_per_year) * excess_returns.mean() / beta_val
```

## 11. Value at Risk (VaR)

VaR estimates how much a set of investments might lose, given normal market conditions, in a set time period such as a day.

$$\text{VaR}_\alpha = -\inf\{l : P(L > l) \leq \alpha\}$$

Where:
- $\alpha$ is the confidence level
- $L$ is the loss of the portfolio

```python
def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))
```

## 12. Conditional Value at Risk (CVaR)

CVaR, also known as Expected Shortfall, measures the expected loss given that the loss is greater than the VaR.

$$\text{CVaR}_\alpha = E[L|L \geq \text{VaR}_\alpha]$$

Where:
- $\alpha$ is the confidence level
- $L$ is the loss of the portfolio

```python
def conditional_value_at_risk(returns, confidence_level=0.95):
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()
```

## 13. Omega Ratio

The Omega ratio is a risk-return performance measure of an investment asset, portfolio, or strategy.

$$\text{Omega}(r) = \frac{\int_r^\infty (1-F(x))dx}{\int_{-\infty}^r F(x)dx}$$

Where:
- $r$ is the threshold return
- $F(x)$ is the cumulative distribution function of the returns

```python
def omega_ratio(returns, threshold):
    return np.mean(np.maximum(returns - threshold, 0)) / np.mean(np.maximum(threshold - returns, 0))
```

## 14. Kappa Ratio

The Kappa ratio is a generalization of the Sortino ratio for higher-order moments.

$$\text{Kappa}_n(r) = \frac{R_p - r}{LPM_n(r)}$$

Where:
- $R_p$ is the return of the portfolio
- $r$ is the threshold return
- $LPM_n(r)$ is the Lower Partial Moment of order $n$

```python
def kappa_ratio(returns, threshold, n):
    lower_partial_moment = np.mean(np.maximum(threshold - returns, 0) ** n)
    return (np.mean(returns) - threshold) / (lower_partial_moment ** (1 / n))
```

## 15. Upside Potential Ratio

The Upside Potential Ratio measures upside performance relative to downside risk.

$$\text{UPR} = \frac{E[\max(R-r, 0)]}{\sqrt{LPM_2(r)}}$$

Where:
- $R$ is the return
- $r$ is the threshold return
- $LPM_2(r)$ is the Lower Partial Moment of order 2

```python
def upside_potential_ratio(returns, threshold):
    upside = np.maximum(returns - threshold, 0)
    downside = np.maximum(threshold - returns, 0)
    return np.mean(upside) / np.sqrt(np.mean(downside ** 2))
```

## 16. Gain-Loss Ratio

The Gain-Loss Ratio is the ratio between the average gain and the average loss.

$$\text{Gain-Loss Ratio} = \frac{E[R|R>0]}{|E[R|R<0]|}$$

Where:
- $R$ represents the returns

```python
def gain_loss_ratio(returns):
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    return np.mean(gains) / abs(np.mean(losses))
```

## 17. Ulcer Index

The Ulcer Index measures downside risk in terms of both depth and duration of price declines.

$$\text{UI} = \sqrt{\frac{\sum_{i=1}^n R_i^2}{n}}$$

Where:
- $R_i$ is the percentage drawdown from previous peak
- $n$ is the number of periods

```python
def ulcer_index(returns):
    cum_returns = np.cumprod(1 + returns)
    drawdowns = 1 - cum_returns / np.maximum.accumulate(cum_returns)
    return np.sqrt(np.mean(drawdowns ** 2))
```

## 18. Pain Index

The Pain Index is the average percentage drawdown over the period.

$$\text{Pain Index} = \frac{\sum_{i=1}^n |D_i|}{n}$$

Where:
- $D_i$ is the drawdown at time $i$
- $n$ is the number of periods

```python
def pain_index(returns):
    cum_returns = np.cumprod(1 + returns)
    drawdowns = 1 - cum_returns / np.maximum.accumulate(cum_returns)
    return np.mean(drawdowns)
```

## 19. Pain Ratio

The Pain Ratio relates the excess return over the risk-free rate to the Pain Index.

$$\text{Pain Ratio} = \frac{R_p - R_f}{\text{Pain Index}}$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate

```python
def pain_ratio(returns, risk_free_rate, periods_per_year):
    return (annualized_return(returns, periods_per_year) - risk_free_rate) / pain_index(returns)
```

## 20. Martin Ratio

The Martin Ratio is similar to the Pain Ratio but uses the Ulcer Index instead of the Pain Index.

$$\text{Martin Ratio} = \frac{R_p - R_f}{\text{Ulcer Index}}$$

Where:
- $R_p$ is the return of the portfolio
- $R_f$ is the risk-free rate

```python
def martin_ratio(returns, risk_free_rate, periods_per_year):
    return (annualized_return(returns, periods_per_year) - risk_free_rate) / ulcer_index(returns)
```

## 21. Skewness

Skewness measures the asymmetry of the probability distribution of returns about its mean.

$$\text{Skewness} = \frac{E[(R-\mu)^3]}{\sigma^3}$$

Where:
- $R$ is the return
- $\mu$ is the mean of the returns
- $\sigma$ is the standard deviation of the returns

```python
from scipy import stats

def skewness(returns):
    return stats.skew(returns)
```

## 22. Kurtosis

Kurtosis measures the "tailedness" of the probability distribution of returns.

$$\text{Kurtosis} = \frac{E[(R-\mu)^4]}{\sigma^4}$$

Where:
- $R$ is the return
- $\mu$ is the mean of the returns
- $\sigma$ is the standard deviation of the returns

```python
def kurtosis(returns):
    return stats.kurtosis(returns)
```

## 23. Jarque-Bera Test

The Jarque-Bera test is a statistical test of the hypothesis that sample data have the skewness and kurtosis matching a normal distribution.

$$JB = \frac{n}{6}\left(S^2 + \frac{1}{4}(K-3)^2\right)$$

Where:
- $n$ is the number of observations
- $S$ is the sample skewness
- $K$ is the sample kurtosis

```python
def jarque_bera_test(returns):
    return stats.jarque_bera(returns)
```

## 24. Hurst Exponent

The Hurst exponent measures the long-term memory of a time series. It relates to the autocorrelations of the time series and the rate at which these decrease as the lag between pairs of values increases.

$$H = \frac{\log(R/S)}{\log(T)}$$

Where:
- $R/S$ is the rescaled range
- $T$ is the duration of the sample of data
- $H$ is the Hurst exponent

```python
def hurst_exponent(returns, lags=range(2, 100)):
    tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
```

## 25. Autocorrelation

Autocorrelation measures the correlation between a time series and a lagged version of itself.

$$\rho_k = \frac{E[(R_t - \mu)(R_{t+k} - \mu)]}{\sigma^2}$$

Where:
- $R_t$ is the return at time $t$
- $\mu$ is the mean of the returns
- $\sigma^2$ is the variance of the returns
- $k$ is the lag

```python
import pandas as pd

def autocorrelation(returns, lag=1):
    return pd.Series(returns).autocorr(lag)
```

## 26. Maximum Consecutive Wins/Losses

This metric measures the longest streak of consecutive positive returns (wins) and negative returns (losses).

```python
def max_consecutive_wins_losses(returns):
    streaks = np.diff(np.where(np.diff(returns >= 0))[0])
    return max(streaks), min(streaks)
```

## 27. Win Rate

The Win Rate is the proportion of positive returns to total returns.

$$\text{Win Rate} = \frac{\text{Number of Positive Returns}}{\text{Total Number of Returns}}$$

```python
def win_rate(returns):
    return np.sum(returns > 0) / len(returns)
```

## 28. Profit Factor

The Profit Factor is the ratio of the sum of all profits over the sum of all losses.

$$\text{Profit Factor} = \frac{\sum \text{Profits}}{\sum |\text{Losses}|}$$

```python
def profit_factor(returns):
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    return np.sum(gains) / abs(np.sum(losses))
```

## 29. Tail Ratio

The Tail Ratio measures the ratio of right tail returns to left tail returns.

$$\text{Tail Ratio} = \frac{|\text{95th percentile return}|}{|\text{5th percentile return}|}$$

```python
def tail_ratio(returns, percentile=5):
    return abs(np.percentile(returns, 100 - percentile)) / abs(np.percentile(returns, percentile))
```

## 30. Downside Deviation

Downside Deviation is similar to standard deviation but only for returns below a threshold.

$$\text{Downside Deviation} = \sqrt{\frac{\sum_{i=1}^n \min(R_i - T, 0)^2}{n}}$$

Where:
- $R_i$ is the return at time $i$
- $T$ is the target return (often 0 or the risk-free rate)
- $n$ is the number of returns

```python
def downside_deviation(returns, threshold=0):
    downside_returns = np.minimum(returns - threshold, 0)
    return np.sqrt(np.mean(downside_returns ** 2))
```

## Usage Example

Here's an example of how you might use these functions:

```python
import numpy as np

# Generate some random returns
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000)
benchmark_returns = np.random.normal(0.0005, 0.01, 1000)
risk_free_rate = 0.02 / 252  # Assuming daily returns and 2% annual risk-free rate
periods_per_year = 252  # Assuming daily returns

# Calculate various metrics
print(f"Total Return: {total_return(returns):.4f}")
print(f"Annualized Return: {annualized_return(returns, periods_per_year):.4f}")
print(f"Sharpe Ratio: {sharpe_ratio(returns, risk_free_rate, periods_per_year):.4f}")
print(f"Maximum Drawdown: {maximum_drawdown(returns):.4f}")
print(f"Calmar Ratio: {calmar_ratio(returns, periods_per_year):.4f}")
print(f"Information Ratio: {information_ratio(returns, benchmark_returns, periods_per_year):.4f}")
print(f"Sortino Ratio: {sortino_ratio(returns, risk_free_rate, periods_per_year):.4f}")
print(f"Omega Ratio: {omega_ratio(returns, risk_free_rate):.4f}")
print(f"Kappa Ratio: {kappa_ratio(returns, risk_free_rate, 3):.4f}")
print(f"Ulcer Index: {ulcer_index(returns):.4f}")
print(f"Hurst Exponent: {hurst_exponent(returns):.4f}")
print(f"Autocorrelation: {autocorrelation(returns):.4f}")
print(f"Win Rate: {win_rate(returns):.4f}")
print(f"Profit Factor: {profit_factor(returns):.4f}")
print(f"Tail Ratio: {tail_ratio(returns):.4f}")
print(f"Downside Deviation: {downside_deviation(returns):.4f}")
```

This comprehensive set of performance measures provides a thorough analysis of quantitative finance strategies and portfolios. Each measure offers unique insights into different aspects of performance, risk, and return characteristics. By using these measures in combination, analysts can gain a well-rounded understanding of an investment strategy's behavior and effectiveness.
