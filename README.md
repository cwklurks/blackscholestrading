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
```

4. Open your browser to `http://localhost:8501`

### Usage

1. **Start with Options Chain**: Enable "Use Real Market Data" and enter a ticker (e.g., AAPL)
2. **Select an Option**: Use the dropdown menus to pick strike, type, and expiration
3. **Load & Analyze**: Click "Load This Option" to automatically populate the analysis
4. **Compare Prices**: View theoretical vs. market prices in the "Pricing & Greeks" tab
5. **Build Strategies**: Use the Portfolio Builder to create complex option positions

## 📊 Key Tabs

- **📈 Pricing & Greeks**: Core option valuation and risk analysis
- **🔥 Heatmaps**: Visual price sensitivity across scenarios  
- **🎲 Monte Carlo**: Statistical simulation validation
- **📊 Market Analysis**: Real market data and volatility analysis
- **💼 Portfolio**: Multi-position strategy builder
- **⛓️ Options Chain**: Live market data with click-to-load functionality

## 🎯 Professional Features

### Click-to-Load Options
Revolutionary feature that allows you to:
- Browse real options chains
- Click any option to automatically load its parameters
- Instantly analyze with theoretical models
- Compare market vs. fair value pricing

### Advanced Analytics
- **Greeks Sensitivity**: Black-themed charts showing risk exposure
- **IV Smile Analysis**: Identify volatility arbitrage opportunities  
- **Portfolio Risk**: Aggregate position analysis with hedging ratios
- **Probability Calculations**: Statistical profit/loss projections

## 🔧 Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical functions and optimization
- `plotly`: Interactive visualizations
- `yfinance`: Real-time market data
- `matplotlib`: Additional plotting capabilities
- `seaborn`: Statistical data visualization

## 📈 Use Cases

### For Traders
- **Identify Mispriced Options**: Compare theoretical vs. market prices
- **Risk Management**: Monitor Greeks exposure across positions
- **Strategy Development**: Test complex multi-leg strategies before trading

### For Educators
- **Teaching Tool**: Visual demonstrations of option pricing concepts
- **Interactive Learning**: Students can experiment with different parameters
- **Real Data**: Connect classroom theory with live market examples

### For Analysts
- **Research Platform**: Analyze volatility patterns and option flow
- **Backtesting**: Historical analysis with Monte Carlo validation
- **Report Generation**: Professional visualizations for presentations

## 🎨 Visual Design

- **Professional UI**: Clean, modern interface with intuitive navigation
- **Black Charts**: Elegant terminal and Greeks charts for professional look
- **Color Coding**: Green for calls, red for puts, consistent theming
- **Responsive Design**: Works on desktop and mobile devices

## 📝 Technical Details

### Black-Scholes Implementation
The core pricing engine implements the full Black-Scholes formula:
- **Call Price**: `C = S×N(d1) - K×e^(-rT)×N(d2)`
- **Put Price**: `P = K×e^(-rT)×N(-d2) - S×N(-d1)`

### Greeks Calculations
All Greeks are calculated analytically:
- **Delta**: First derivative with respect to underlying price
- **Gamma**: Second derivative (delta sensitivity)
- **Theta**: Time decay rate
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Monte Carlo Validation
Simulations use geometric Brownian motion:
- Thousands of price paths generated
- Statistical convergence analysis
- Comparison with analytical solutions

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🎯 Future Enhancements

- [ ] American option pricing using binomial trees
- [ ] Exotic options (barrier, Asian, lookback)
- [ ] Real-time options flow analysis
- [ ] Historical backtesting framework
- [ ] Advanced volatility models (GARCH, stochastic)
- [ ] Risk scenario analysis and stress testing
- [ ] Export functionality for reports and data
- [ ] Mobile app version
- [ ] API integration for institutional data feeds
