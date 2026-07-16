# Phoenix Autocall Pricer

Monte Carlo pricing engine for autocallable structured products, built in Python.  
Underlying asset: **Eurostoxx 50** (live market data via Yahoo Finance).  
Model: **Black-Scholes** under risk-neutral probability measure.

---

## Features

- **100,000 GBM trajectories** simulated under risk-neutral measure Q
- Full **Phoenix Autocall mechanics**: early recall, memory coupon, capital protection at maturity
- **Variance reduction** via antithetic variates
- **95% confidence interval** on the price (Central Limit Theorem)
- **Greeks** computed via bump-and-reval with common random numbers (Delta, Vega, Theta)
- **Equilibrium coupon search** via bisection: replicates the real structuring process (par issuance)
- **2 visualizations**: payoff distribution by scenario, price/volatility sensitivity curve

---

## Installation

```bash
pip install numpy pandas yfinance matplotlib
```

---

## Usage

```bash
python autocall_pricer.py
```

Market data (spot price, historical volatility) are downloaded automatically.  
Charts are saved in the current directory (`distribution_payoffs.png`, `vega_curve.png`).

---

## Parameters

All parameters are configurable at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `TICKER` | `^STOXX50E` | Underlying asset (Yahoo Finance ticker) |
| `NOMINAL` | 1,000 EUR | Notional amount |
| `COUPON_RATE` | 7% | Coupon per observation period |
| `RECALL_BARRIER` | 100% | Early recall barrier |
| `COUPON_BARRIER` | 70% | Coupon payment barrier |
| `PROTECTION_BARRIER` | 60% | Capital protection barrier at maturity |
| `OBSERVATION_DATES` | [1, 2, 3, 4, 5] | Observation dates (in years) |
| `COUPON_MEMORY` | True | Coupon memory mechanism (Phoenix) |
| `RISK_FREE_RATE` | 3% | EUR OIS rate (update as needed) |
| `N_SIMULATIONS` | 100,000 | Monte Carlo trajectories |

---

## Results (Eurostoxx 50, July 2026)

```
Spot (S0)              : 6,204.91
Historical volatility  :   15.48%  (252 trading days)
Risk-free rate         :    3.00%  (EUR OIS)

Monte Carlo price      : 1,059.09 EUR  (105.91% of notional)
95% CI                 : [1,058.45, 1,059.73]
Statistical error      :    ±0.64 EUR

Early recalls          :   81.4%
Capital protected      :   15.3%
Capital loss           :    3.3%

Delta                  :   0.0000  (scale-invariant at issuance)
Vega                   : -501.04   EUR/unit sigma  (-5.01 EUR per +1pp vol)
Theta                  :  +0.1666  EUR/trading day

Equilibrium coupon     :    3.97%  (par issuance, 15 iterations)
```

---

## Phoenix Autocall Mechanics

At each observation date:

| Condition | Event |
|---|---|
| S(t) ≥ S₀ × 100% | **Early recall**: notional + coupons (including memory) |
| S₀ × 70% ≤ S(t) < S₀ × 100% | **Coupon paid** (including accumulated memory) |
| S(t) < S₀ × 70% | **No coupon**: stored in memory (Phoenix) |

At maturity (if never recalled):

| Condition | Payoff |
|---|---|
| S(T) ≥ S₀ × 60% | Full notional repayment |
| S(T) < S₀ × 60% | Notional × (S(T) / S₀) — proportional capital loss |

---

## Theoretical Framework

**Risk-neutral measure**: simulation runs under Q (risk-neutral probability), not P (historical probability). Under Q, all assets earn the risk-free rate r, ensuring no-arbitrage pricing. The price equals the discounted expected payoff under Q.

**Exact GBM discretization**:
```
S(t + dt) = S(t) × exp( (r - σ²/2) × dt + σ × √dt × Z ),   Z ~ N(0,1)
```
The `(r - σ²/2)` term is the Itô correction (Jensen's inequality on the exponential function).

**Greeks (bump-and-reval)**: analytical derivatives do not exist for discontinuous barrier payoffs. Each Greek is approximated via central finite differences (O(h²) error), with common random numbers to cancel stochastic noise.

**Equilibrium coupon**: the price/coupon relationship is strictly increasing. Bisection inverts this relationship to find the maximum coupon compatible with par issuance. Convergence in O(log₂(n)), approximately 15 iterations.

---

## Model Limitations

- **Constant volatility**: Black-Scholes ignores the volatility smile and skew, which are particularly impactful for autocall downside barriers
- **Historical volatility** used as a proxy for implied volatility (backward-looking vs forward-looking)
- **Dividends** not modeled (adjustment: replace r with r - q in the drift)
- **Issuer credit risk** not captured
- **Risk-free rate** set manually (update according to current market conditions)

---

## Tech Stack

`Python` `NumPy` `pandas` `yfinance` `matplotlib`

---

## Code Structure

```
autocall_pricer.py
│
├── fetch_market_data()          # Spot and historical volatility (Yahoo Finance)
├── simulate_gbm()               # GBM trajectories under risk-neutral measure Q
├── compute_payoffs()            # Phoenix Autocall payoff + scenario classification
├── price_product()              # Monte Carlo price + 95% confidence interval
├── compute_greeks()             # Delta, Vega, Theta (bump-and-reval)
├── find_equilibrium_coupon()    # Equilibrium coupon via bisection
├── plot_payoff_distribution()   # Payoff distribution by scenario
├── plot_vega_curve()            # Price/volatility sensitivity curve
└── main()                       # Entry point
```
