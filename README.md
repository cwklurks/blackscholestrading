# Black-Scholes Pricing Engine

Full-stack options pricing workstation implementing five stochastic models. FastAPI backend with a Next.js frontend covering pricing, Greeks, volatility surfaces, strategy payoffs, and historical backtesting.

[![CI](https://github.com/cwklurks/blackscholestrading/actions/workflows/ci.yml/badge.svg)](https://github.com/cwklurks/blackscholestrading/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Tech Stack

### Backend

- **Python 3.12** with FastAPI
- **NumPy** and **SciPy** for numerical computing
- **Numba** (JIT compilation) for Monte Carlo acceleration
- **yfinance** for real-time market data

### Frontend

- **Next.js 16** with the App Router
- **TypeScript** and **React 19**
- **shadcn/ui** component library with **Tailwind CSS**
- **Plotly.js** for interactive charts and surfaces

---

## Model Comparison

| Feature | Black-Scholes | Binomial Tree | Heston Model | GARCH Model | Bates Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Volatility** | Constant | Constant | Stochastic | Stochastic (Discrete) | Stochastic |
| **Exercise Style** | European | **American** | European | European | European |
| **Market Jumps** | No | No | No | No | Yes |
| **Skew/Smile** | No | No | Yes | Yes | Yes |
| **Computational Cost** | Low (Closed Form) | Medium (Iterative) | High (Monte Carlo) | High (Monte Carlo) | High (Monte Carlo) |

---

## Features

- 5 pricing models (Black-Scholes, Binomial, Heston, GARCH, Bates)
- Analytical and numerical Greeks (delta, gamma, theta, vega, rho)
- Implied volatility surface from live options chains
- Multi-leg strategy payoff diagrams
- Historical backtesting with P&L, max drawdown, Sharpe ratio
- Real-time market data via Yahoo Finance

---

## Quick Start

### Docker Compose

```bash
git clone https://github.com/cwklurks/blackscholestrading.git
cd blackscholestrading
docker-compose up
```

API at http://localhost:8000 -- Frontend at http://localhost:3000

### Manual Setup

```bash
# Backend
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd web
npm install
npm run dev
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/models` | List available pricing models |
| POST | `/api/price` | Price an option |
| POST | `/api/heatmap` | Spot x volatility heatmap |
| POST | `/api/monte-carlo` | Monte Carlo simulation |
| POST | `/api/volatility-surface` | IV surface from live chain |
| GET | `/api/market/{ticker}` | Stock price and history |
| GET | `/api/chain/{ticker}` | Options chain data |
| POST | `/api/strategy/payoff` | Multi-leg payoff diagram |
| POST | `/api/backtest` | Historical backtest |

---

## Architecture

```
blackscholestrader/
├── api/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py       # Application entrypoint
│   │   ├── routers/      # API route handlers
│   │   ├── schemas/      # Pydantic request/response models
│   │   └── services/     # Business logic layer
│   ├── models/           # Pricing models (BS, Binomial, MC)
│   ├── strategies/       # Option strategy and backtest logic
│   ├── data/             # Market data providers
│   ├── utils/            # Constants and helpers
│   └── tests/            # Test suite
├── web/                  # Next.js frontend
│   └── src/
│       ├── app/          # App Router pages
│       └── components/   # React components
└── docker-compose.yml    # Local development setup
```

---

## License

Distributed under the MIT License.
