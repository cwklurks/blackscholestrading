# Advanced Black-Scholes Option Pricing & Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blackscholestrading.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3572A5?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Issues](https://img.shields.io/github/issues/cwklurks/blackscholestrading)](https://github.com/cwklurks/blackscholestrading/issues)
[![PRs](https://img.shields.io/github/issues-pr/cwklurks/blackscholestrading)](https://github.com/cwklurks/blackscholestrading/pulls)
[![Last commit](https://img.shields.io/github/last-commit/cwklurks/blackscholestrading)](https://github.com/cwklurks/blackscholestrading/commits/main)

Black-Scholes options analysis with real market data, Greeks, IV surfaces, Monte Carlo, and multi-leg strategy payoffs. Designed for quick pricing checks and portfolio-level risk views.

## Live demo
<https://blackscholestrading.streamlit.app/>

## Features

- **Pricing and Greeks**
  - Closed-form Black-Scholes call/put pricing
  - Delta, Gamma, Theta, Vega, Rho with short explanations
  - Sensitivity charts across spot, volatility, and time

- **Market data**
  - Spot, option chains, OI, volume via `yfinance`
  - Historical volatility and IV smile/term structure
  - Bid/ask, moneyness filters, ITM/OTM highlighting

- **Visualization**
  - Heatmaps for price and Greek surfaces
  - 3D pricing/IV surfaces
  - Monte Carlo GBM path views
  - Strategy payoff diagrams

- **Options chain and portfolio**
  - Browse chains, click-to-load parameters
  - Build multi-leg strategies (spreads, straddles, condors)
  - Aggregate portfolio Greeks, max P/L, breakevens
  - Probability of profit estimates

## Quick start

```bash
git clone https://github.com/cwklurks/blackscholestrading.git
cd blackscholestrading
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
# open http://localhost:8501
