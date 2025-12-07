import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings

from analytics import (
    BlackScholesModel,
    implied_volatility,
    monte_carlo_option_price,
    calculate_historical_volatility,
    iv_hv_stats,
    risk_reversal_and_fly,
    generate_gbm_replay,
    backtest_option_strategy,
    option_metrics,
    atm_iv_from_chain,
    price_with_model,
)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Page Config & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Black-Scholes Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minor tweaks that Streamlit config can't handle
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Subtle card styling */
    .metric-card {
        background-color: #252526;
        border: 1px solid #3e3e42;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card h3 {
        margin-top: 0;
        font-size: 1rem;
        color: #aaaaaa;
        font-weight: 500;
    }
    .metric-card h2 {
        margin: 5px 0 0 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Adjust sidebar spacing */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Clean up dataframe styling */
    .stDataFrame {
        border: 1px solid #3e3e42;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Fetching
# -----------------------------------------------------------------------------
def get_icon(name, size=20, color="currentColor", mode="html"):
    if mode == "url":
        # Use a specific color for the CDN URL (default to white for dark theme)
        url_color = "white" if color == "currentColor" else color.replace("#", "%23")
        return f"![{name}](https://api.iconify.design/lucide/{name}.svg?color={url_color}&height={size})"

    # Lucide Icons (v0.344.0)
    icons = {
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        "file-text": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line>',
        "calculator": '<rect width="16" height="20" x="4" y="2" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="16" y1="14" x2="16" y2="14"></line><line x1="16" y1="18" x2="16" y2="18"></line><line x1="12" y1="14" x2="12" y2="14"></line><line x1="12" y1="18" x2="12" y2="18"></line><line x1="8" y1="14" x2="8" y2="14"></line><line x1="8" y1="18" x2="8" y2="18"></line>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>',
        "wifi": '<path d="M5 12.55a11 11 0 0 1 14.08 0"></path><path d="M1.42 9a16 16 0 0 1 21.16 0"></path><path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path><line x1="12" y1="20" x2="12.01" y2="20"></line>'
    }
    
    svg_content = icons.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">{svg_content}</svg>'

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        fetched_at = datetime.utcnow()
        if hist is None or hist.empty:
            return None, None, fetched_at
        return hist, info, fetched_at
    except Exception as exc:
        return None, {"error": str(exc)}, datetime.utcnow()


@st.cache_data(show_spinner=False)
def fetch_options_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        fetched_at = datetime.utcnow()

        if not expirations:
            return None, None, fetched_at

        all_options = []

        for exp in expirations[:5]:
            try:
                opt = stock.option_chain(exp)
                calls = opt.calls
                puts = opt.puts

                calls['expiration'] = exp
                calls['type'] = 'Call'
                puts['expiration'] = exp
                puts['type'] = 'Put'

                all_options.append(calls)
                all_options.append(puts)
            except Exception:
                continue

        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            return options_df, expirations, fetched_at
        return None, None, fetched_at
    except Exception:
        return None, None, datetime.utcnow()


def parse_uploaded_prices(upload):
    if upload is None:
        return None
    try:
        df = pd.read_csv(upload)
        date_col = [c for c in df.columns if 'date' in c.lower()]
        price_col = [c for c in df.columns if c.lower() in {'close', 'price'}]
        if not date_col or not price_col:
            return None
        series = pd.Series(df[price_col[0]].values, index=pd.to_datetime(df[date_col[0]]))
        series = series.sort_index()
        return series
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    
    # 1. Market Data Configuration
    st.markdown(f"### {get_icon('activity')} Market Data", unsafe_allow_html=True)
    use_market_data = st.toggle("Live Connection", value=False, help="Pull real-time quotes via Yahoo Finance.")
    
    if use_market_data:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                    border-radius: 8px; padding: 12px; margin: 8px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {get_icon("wifi", color="#4ade80")}
                <span style="color: #e0e0e0; font-weight: 500;">Connected to Live Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #3e3e42; border-radius: 8px; padding: 12px; margin: 8px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #888;">●</span>
                <span style="color: #888;">Disconnected - Manual Mode</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col_tick, col_ref = st.columns([0.7, 0.3])
    with col_tick:
        ticker = st.text_input("Ticker", value="AAPL" if use_market_data else "BTC-USD", placeholder="AAPL", disabled=not use_market_data)
    with col_ref:
        st.write("") # Spacer
        st.write("")
        if st.button("↻", disabled=not use_market_data):
            fetch_stock_data.clear()
            fetch_options_chain.clear()
            st.session_state.last_refresh = datetime.utcnow()
    
    hist = None
    hist_vol = 0.5
    current_price = 100.0
    fetched_at = None
    
    if use_market_data and ticker:
        hist, info, fetched_at = fetch_stock_data(ticker)
        if hist is not None and not hist.empty:
            current_price = float(hist['Close'].iloc[-1])
            hist_vol = calculate_historical_volatility(hist['Close']) or 0.2
            st.caption(f"Price: ${current_price:.2f} | HV: {hist_vol:.1%}")
            if fetched_at:
                st.caption(f"Last updated: {fetched_at.strftime('%H:%M:%S')} UTC")
        else:
            st.warning("No data found.")

    with st.expander(f"{get_icon('database', mode='url')} Data Inputs", expanded=not use_market_data):
        price_upload = st.file_uploader("Upload History (CSV)", type=["csv"])
        uploaded_series = parse_uploaded_prices(price_upload)
        
        if uploaded_series is not None:
            hist = uploaded_series.to_frame(name="Close")
            current_price = float(hist['Close'].iloc[-1])
            hist_vol = calculate_historical_volatility(hist['Close']) or hist_vol
            st.success("CSV Loaded")

        manual_price = st.number_input("Spot Price", value=float(current_price), step=0.5)
        if manual_price != current_price:
            current_price = manual_price
            
        base_rate = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25) / 100
        dividend_yield = st.number_input("Div Yield (%)", value=0.0, step=0.25) / 100
        borrow_cost = st.number_input("Borrow Cost (%)", value=0.0, step=0.25) / 100

    # 2. Option Parameters
    with st.expander(f"{get_icon('file-text', mode='url')} Option Contract", expanded=True):
        # Handle selected option from chain
        if st.session_state.selected_option:
            opt = st.session_state.selected_option
            st.info(f"Analying: {opt['option_type']} ${opt['strike']} (Exp: {opt['expiration']})")
            if st.button("Clear Selection"):
                st.session_state.selected_option = None
                st.rerun()
            
            strike_default = opt['strike']
            vol_default = opt['implied_vol']
            selected_exp = opt['expiration']
            if hasattr(selected_exp, 'date'):
                exp_default = selected_exp.date()
            else:
                exp_default = datetime.now().date() + timedelta(days=30)
        else:
            strike_default = current_price
            vol_default = hist_vol
            exp_default = datetime.now().date() + timedelta(days=30)

        strike = st.number_input("Strike Price", value=float(strike_default), step=1.0)
        expiry_date = st.date_input("Expiration", value=exp_default, min_value=datetime.now().date())
        
        days_to_expiry = max((expiry_date - datetime.now().date()).days, 0)
        time_to_maturity = max(days_to_expiry / 365.0, 1e-6)
        st.caption(f"Time to Expiry: {days_to_expiry} days ({time_to_maturity:.3f}y)")

        volatility = st.slider("Implied Volatility (%)", min_value=1.0, max_value=200.0, value=vol_default * 100, step=0.5) / 100
    
    # 3. Model Configuration
    with st.expander(f"{get_icon('calculator', mode='url')} Pricing Model", expanded=False):
        pricing_model = st.selectbox("Model", ["Black-Scholes", "Binomial (American)", "Heston MC", "GARCH MC", "Bates Jump-Diffusion"])
        
        # Model specific params
        binomial_steps = 180
        mc_paths = 4000
        mc_steps = 120
        
        if pricing_model != "Black-Scholes":
            st.markdown("#### Advanced Model Params")
            if pricing_model == "Binomial (American)":
                binomial_steps = st.slider("Steps", 50, 500, 180)
            else:
                mc_paths = st.slider("Paths", 1000, 10000, 4000)
                mc_steps = st.slider("Steps", 50, 252, 120)

    # Dictionary for model params
    model_params = {
        "binomial_steps": binomial_steps,
        "mc_paths": mc_paths,
        "mc_steps": mc_steps,
        # Default advanced params (kept simple for UI)
        "heston_kappa": 1.5, "heston_theta": 0.04, "heston_rho": -0.7, "heston_vol_of_vol": 0.3,
        "heston_v0": volatility ** 2,
        "jump_lambda": 0.1, "jump_mu": -0.05, "jump_delta": 0.2,
        "garch_alpha0": 2e-6, "garch_alpha1": 0.08, "garch_beta1": 0.9
    }

# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------

# Helper for Plotly Dark Theme
def get_chart_layout(title="", height=400):
    return dict(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="sans-serif", size=12, color="#e0e0e0")
    )

st.title("Black-Scholes Trader")
st.markdown(f"Analysis for **{ticker.upper()}** @ ${current_price:.2f}")

# Calculate Metrics
interest_rate = base_rate # Use the session var directly
call_metrics = option_metrics(pricing_model, current_price, strike, time_to_maturity, interest_rate, dividend_yield, volatility, borrow_cost, model_params, 'call')
put_metrics = option_metrics(pricing_model, current_price, strike, time_to_maturity, interest_rate, dividend_yield, volatility, borrow_cost, model_params, 'put')

# 1. Hero Pricing Section
col_call, col_put = st.columns(2)

with col_call:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #4CAF50;">
        <h3>CALL OPTION</h3>
        <h2>${call_metrics['price']:.2f}</h2>
        <div style="display: flex; gap: 15px; margin-top: 10px; font-size: 0.9rem; color: #888;">
            <span>Δ {call_metrics['delta']:.3f}</span>
            <span>Θ {call_metrics['theta']:.3f}</span>
            <span>Γ {call_metrics['gamma']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_put:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #F44336;">
        <h3>PUT OPTION</h3>
        <h2>${put_metrics['price']:.2f}</h2>
        <div style="display: flex; gap: 15px; margin-top: 10px; font-size: 0.9rem; color: #888;">
            <span>Δ {put_metrics['delta']:.3f}</span>
            <span>Θ {put_metrics['theta']:.3f}</span>
            <span>Γ {call_metrics['gamma']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. Main Tabs
tabs = st.tabs(["Analysis", "Heatmaps", "Monte Carlo", "Market Data", "Portfolio", "Chain", "Backtest"])

with tabs[0]: # Analysis
    col_greeks, col_chart = st.columns([1, 2])
    
    with col_greeks:
        st.subheader("Greeks")
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
            "Call": [call_metrics['delta'], call_metrics['gamma'], call_metrics['theta'], call_metrics['vega'], call_metrics['rho']],
            "Put": [put_metrics['delta'], call_metrics['gamma'], put_metrics['theta'], call_metrics['vega'], put_metrics['rho']]
        })
        st.dataframe(greeks_df.style.format("{:.4f}", subset=["Call", "Put"]), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Scenario Analysis")
        shock_pct = st.slider("Spot Shock (%)", 1, 20, 5)
        vol_shock = st.slider("Vol Shock (+/- %)", 1, 20, 5)
        
    with col_chart:
        st.subheader("Sensitivity Analysis")
        # Greeks Chart
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        deltas_c = [BlackScholesModel(p, strike, time_to_maturity, interest_rate, volatility, dividend_yield, borrow_cost).delta_call() for p in price_range]
        deltas_p = [BlackScholesModel(p, strike, time_to_maturity, interest_rate, volatility, dividend_yield, borrow_cost).delta_put() for p in price_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=deltas_c, name="Call Delta", line=dict(color="#4CAF50")))
        fig.add_trace(go.Scatter(x=price_range, y=deltas_p, name="Put Delta", line=dict(color="#F44336")))
        fig.add_vline(x=current_price, line_dash="dash", line_color="#888", annotation_text="Spot")
        fig.update_layout(**get_chart_layout("Delta vs Spot Price"))
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]: # Heatmaps
    col_hm_ctrl, col_hm_plot = st.columns([1, 3])
    with col_hm_ctrl:
        spot_min = st.number_input('Min Spot', value=current_price*0.8)
        spot_max = st.number_input('Max Spot', value=current_price*1.2)
        vol_min = st.slider('Min Vol', 0.1, 1.0, max(0.1, volatility*0.5))
        vol_max = st.slider('Max Vol', 0.1, 2.0, min(2.0, volatility*1.5))
        
    with col_hm_plot:
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        
        for i, v in enumerate(vol_range):
            for j, s in enumerate(spot_range):
                model = BlackScholesModel(s, strike, time_to_maturity, interest_rate, v, dividend_yield, borrow_cost)
                call_prices[i, j] = model.call_price()
                
        fig_hm = go.Figure(data=go.Heatmap(
            z=call_prices,
            x=np.round(spot_range, 2),
            y=np.round(vol_range * 100, 1),
            colorscale='Viridis',
            colorbar=dict(title="Price")
        ))
        fig_hm.update_layout(**get_chart_layout("Call Price Heatmap (Spot vs Vol)"))
        fig_hm.update_xaxes(title="Spot Price")
        fig_hm.update_yaxes(title="Volatility (%)")
        st.plotly_chart(fig_hm, use_container_width=True)

with tabs[2]: # Monte Carlo
    if st.button("Run Simulation", key="btn_mc"):
        with st.spinner("Simulating..."):
            mc_price, mc_se, terminal_prices = monte_carlo_option_price(
                current_price, strike, time_to_maturity, interest_rate, volatility, 
                mc_paths, 'call', q=dividend_yield, borrow_cost=borrow_cost
            )
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MC Call Price", f"${mc_price:.2f}")
            c2.metric("Standard Error", f"${mc_se:.4f}")
            c3.metric("BS Diff", f"${mc_price - call_metrics['price']:.4f}")
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=terminal_prices, nbinsx=50, name='Terminal Prices', marker_color='#5E81AC'))
            fig_dist.add_vline(x=strike, line_dash="dash", line_color="#F44336", annotation_text="Strike")
            fig_dist.update_layout(**get_chart_layout("Terminal Price Distribution"))
            st.plotly_chart(fig_dist, use_container_width=True)

with tabs[3]: # Market Data
    if hist is not None:
        fig_candle = go.Figure(data=[go.Candlestick(x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'])])
        fig_candle.update_layout(**get_chart_layout(f"{ticker} Price History"))
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volatility Analysis
        returns = hist['Close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='21d Hist Vol', line=dict(color='#E5E9F0')))
        fig_vol.add_hline(y=volatility, line_dash="dash", line_color="#88C0D0", annotation_text="Current IV")
        fig_vol.update_layout(**get_chart_layout("Historical Volatility"))
        st.plotly_chart(fig_vol, use_container_width=True)
        
    else:
        st.info("Load market data or upload a CSV to view analysis.")

with tabs[4]: # Portfolio
    col_add, col_list = st.columns([1, 2])
    with col_add:
        st.markdown("#### Add Position")
        p_type = st.selectbox("Type", ["Long Call", "Short Call", "Long Put", "Short Put"])
        p_strike = st.number_input("Strike", value=strike)
        p_qty = st.number_input("Qty", 1)
        if st.button("Add to Portfolio"):
            st.session_state.portfolio.append({"type": p_type, "strike": p_strike, "qty": p_qty})
    
    with col_list:
        st.markdown("#### Current Holdings")
        if st.session_state.portfolio:
            pf_df = pd.DataFrame(st.session_state.portfolio)
            st.dataframe(pf_df, use_container_width=True, hide_index=True)
            if st.button("Clear Portfolio"):
                st.session_state.portfolio = []
                st.rerun()
            
            # Simple Payoff
            if not pf_df.empty:
                prices = np.linspace(current_price*0.5, current_price*1.5, 100)
                pnl = np.zeros_like(prices)
                
                for pos in st.session_state.portfolio:
                    is_call = "Call" in pos['type']
                    is_long = "Long" in pos['type']
                    k = pos['strike']
                    q = pos['qty']
                    
                    # Intrinsic value at expiry
                    val = np.maximum(prices - k, 0) if is_call else np.maximum(k - prices, 0)
                    
                    # Approx entry price (using current model)
                    entry_model = BlackScholesModel(current_price, k, time_to_maturity, interest_rate, volatility, dividend_yield, borrow_cost)
                    entry = entry_model.call_price() if is_call else entry_model.put_price()
                    
                    leg_pnl = (val - entry) if is_long else (entry - val)
                    pnl += leg_pnl * q
                
                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(x=prices, y=pnl, fill='tozeroy', name='P&L', line=dict(color='#81A1C1')))
                fig_payoff.add_vline(x=current_price, line_dash="dash", line_color="white", annotation_text="Current")
                fig_payoff.add_hline(y=0, line_color="#888")
                fig_payoff.update_layout(**get_chart_layout("Portfolio Payoff (at Expiry)"))
                st.plotly_chart(fig_payoff, use_container_width=True)

with tabs[5]: # Chain
    if use_market_data and ticker:
        chain_df, expirations, _ = fetch_options_chain(ticker)
        if chain_df is not None:
            sel_exp = st.selectbox("Expiry", expirations[:5] if expirations else [])
            if sel_exp:
                subset = chain_df[chain_df['expiration'] == sel_exp].copy()
                
                # Filter near money
                center_strike = current_price
                subset = subset[(subset['strike'] > center_strike * 0.8) & (subset['strike'] < center_strike * 1.2)]
                
                # Pivot for standard view
                calls = subset[subset['type'] == 'Call'][['strike', 'lastPrice', 'impliedVolatility', 'volume']]
                puts = subset[subset['type'] == 'Put'][['strike', 'lastPrice', 'impliedVolatility', 'volume']]
                
                chain_view = pd.merge(calls, puts, on='strike', suffixes=('_call', '_put'))
                chain_view = chain_view.rename(columns={
                    'lastPrice_call': 'Call Last', 'impliedVolatility_call': 'Call IV', 'volume_call': 'Call Vol',
                    'lastPrice_put': 'Put Last', 'impliedVolatility_put': 'Put IV', 'volume_put': 'Put Vol'
                })
                
                st.dataframe(chain_view.style.format({
                    'Call IV': '{:.1%}', 'Put IV': '{:.1%}', 
                    'Call Last': '${:.2f}', 'Put Last': '${:.2f}'
                }), use_container_width=True, hide_index=True)
                
    else:
        st.info("Market data required for options chain.")

with tabs[6]: # Backtest
    st.subheader("Strategy Backtest")
    bt_strategy = st.selectbox("Strategy Type", ["Long Call", "Short Call", "Long Put", "Short Put"])
    bt_qty = st.number_input("Backtest Qty", 1)
    
    if hist is not None and not hist.empty:
        if st.button("Run Backtest"):
            # Simple backtest logic
            side = "long" if "Long" in bt_strategy else "short"
            otype = "call" if "Call" in bt_strategy else "put"
            
            bt_res = backtest_option_strategy(
                hist['Close'], strike, expiry_date, interest_rate, volatility, 
                option_type=otype, quantity=bt_qty, side=side
            )
            
            if not bt_res.empty:
                fig_bt = px.line(bt_res, y='pnl', title="Cumulative P&L")
                fig_bt.update_layout(**get_chart_layout("Backtest P&L"))
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.metric("Total P&L", f"${bt_res['pnl'].iloc[-1]:.2f}")
    else:
        st.info("Load market data to backtest.")

