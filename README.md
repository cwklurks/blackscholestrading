# Advanced Black-Scholes Option Pricing & Analysis Platform

This repository provides a comprehensive Black-Scholes options analysis platform that combines theoretical pricing models with real-time market data. The application helps traders identify mispriced options, analyze risk through Greeks, and build complex options strategies with interactive visualizations.

**Live Demo:** [(https://blackscholestrading.streamlit.app/)]

## Features:

### 1. **Real-Time Options Pricing & Greeks**

- Calculate theoretical option values using the Black-Scholes model
- Display all Greeks (Delta, Gamma, Theta, Vega, Rho) with explanations
- Compare theoretical prices with actual market prices to spot opportunities
- Interactive charts showing how Greeks change across different price levels

### 2. **Live Market Data Integration**

- Fetch real-time stock prices and options chains from Yahoo Finance
- Calculate and display historical volatility
- Load actual bid/ask spreads, volume, and open interest
- Analyze implied volatility smile and term structure

### 3. **Enhanced Visualizations**

- Clean, spaced-out heatmaps showing option prices across spot price and volatility ranges
- 3D surface plots for visual option pricing analysis
- Monte Carlo simulation with price path visualization
- Portfolio payoff diagrams for strategy analysis

### 4. **Options Chain Analysis**

- Browse live options chains with advanced filtering (expiration, type, moneyness)
- Color-coded ITM/OTM options for quick identification
- Implied volatility analysis with smile visualization
- Open interest distribution charts

### 5. **Portfolio Builder**

- Build multi-leg options strategies (straddles, spreads, condors, etc.)
- Calculate aggregate portfolio Greeks
- Risk metrics including max profit/loss and breakeven points
- Probability of profit calculations

### 6. **User-Friendly UI**

- Date picker for expiration selection (no more confusing year decimals)
- Toggle between percentage and decimal inputs for volatility and rates
- Reference guides for typical volatility ranges and current treasury rates
- Tabbed interface for organized feature access

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cwklurks/blackscholestrading.git
cd blackscholestrading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
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

### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy with one click
4. Share your live app URL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## 📞 Support

For questions, suggestions, or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]

---

**Built with ❤️ for the options trading and financial analysis community** 
