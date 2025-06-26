import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
import plotly.express as px
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Advanced Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

/* Fix metric styling - ensure readable text with proper contrast */
.stMetric {
    background-color: #ffffff !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid #e9ecef !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stMetric > div {
    color: #000000 !important;
    background-color: #ffffff !important;
}

.stMetric [data-testid="metric-container"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stMetric [data-testid="metric-container"] > div {
    color: #000000 !important;
}

.stMetric label {
    color: #495057 !important;
    font-weight: 600 !important;
}

.stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #000000 !important;
    font-size: 1.5rem !important;
    font-weight: bold !important;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Tab styling - reduce crowding with proper contrast */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
    background-color: #f8f9fa !important;
    border-radius: 15px !important;
    padding: 8px !important;
    margin-bottom: 20px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 60px !important;
    background-color: transparent !important;
    border-radius: 12px !important;
    color: #495057 !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    min-width: 150px !important;
}

.stTabs [aria-selected="true"] {
    background-color: white !important;
    color: #007bff !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

/* Black sidebar styling */
.css-1d391kg {
    background-color: #262730 !important;
}

/* Sidebar text - white on black */
.css-1d391kg .stMarkdown {
    color: #ffffff !important;
}

.css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
    color: #ffffff !important;
}

/* Fix sidebar date input */
.stDateInput > div > div {
    background-color: white !important;
    color: black !important;
}

.stDateInput input {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
}

/* Fix sidebar widgets - white text on black background */
.css-1d391kg .stSelectbox label,
.css-1d391kg .stNumberInput label,
.css-1d391kg .stSlider label,
.css-1d391kg .stRadio label,
.css-1d391kg .stCheckbox label {
    color: #ffffff !important;
}

/* Sidebar widget styling */
.css-1d391kg .stSelectbox > div > div,
.css-1d391kg .stNumberInput > div > div,
.css-1d391kg .stTextInput > div > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Main content area text */
.stMarkdown {
    color: var(--text-color) !important;
}

/* Plotly chart containers */
.js-plotly-plot {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Fix metric container text with white background */
div[data-testid="metric-container"] {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

div[data-testid="metric-container"] * {
    color: black !important;
}

/* Ensure proper contrast for all text elements */
.stApp {
    color: var(--text-color);
}

/* Black sidebar */
[data-testid="stSidebar"] {
    background-color: #262730 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Override any forced black text in inappropriate places */
.stApp [data-testid="stMarkdownContainer"] {
    color: var(--text-color) !important;
}

/* Info and success message styling */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 8px !important;
}

/* Terminal data styling - black text */
.terminal-data {
    background-color: #f8f9fa !important;
    color: #000000 !important;
    font-family: 'Courier New', monospace !important;
    padding: 10px !important;
    border-radius: 5px !important;
    border: 1px solid #dee2e6 !important;
}
</style>
""", unsafe_allow_html=True)

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def d1(self):
        return (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * sqrt(self.T)
    
    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * exp(-self.r * self.T) * norm.cdf(self.d2())
    
    def put_price(self):
        return self.K * exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())
    
    def delta_call(self):
        return norm.cdf(self.d1())
    
    def delta_put(self):
        return -norm.cdf(-self.d1())
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * sqrt(self.T))
    
    def theta_call(self):
        term1 = -self.S * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        term2 = -self.r * self.K * exp(-self.r * self.T) * norm.cdf(self.d2())
        return (term1 + term2) / 365
    
    def theta_put(self):
        term1 = -self.S * norm.pdf(self.d1()) * self.sigma / (2 * sqrt(self.T))
        term2 = self.r * self.K * exp(-self.r * self.T) * norm.cdf(-self.d2())
        return (term1 + term2) / 365
    
    def vega(self):
        return self.S * norm.pdf(self.d1()) * sqrt(self.T) / 100
    
    def rho_call(self):
        return self.K * self.T * exp(-self.r * self.T) * norm.cdf(self.d2()) / 100
    
    def rho_put(self):
        return -self.K * self.T * exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    def objective(sigma):
        model = BlackScholesModel(S, K, T, r, sigma)
        if option_type == 'call':
            return abs(model.call_price() - option_price)
        else:
            return abs(model.put_price() - option_price)
    
    result = minimize_scalar(objective, bounds=(0.001, 5), method='bounded')
    return result.x

def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000, option_type='call'):
    np.random.seed(42)
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(num_simulations)
    
    return option_price, std_error, ST

@st.cache_data
def fetch_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except:
        return None, None

# Fetch options chain
@st.cache_data
def fetch_options_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations:
            return None, None
        
        # Get options chain for all expirations
        all_options = []
        
        for exp in expirations[:5]:  # Limit to first 5 expirations for performance
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
            except:
                continue
        
        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            return options_df, expirations
        else:
            return None, None
    except:
        return None, None

# Calculate historical volatility
def calculate_historical_volatility(prices, periods=252):
    returns = np.log(prices / prices.shift(1))
    return returns.std() * np.sqrt(periods)

# Session state for option selection
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None

# Sidebar
with st.sidebar:
    st.header("📊 Option Parameters")
    
    # Market Data Section
    use_market_data = st.checkbox("Use Real Market Data", value=False)
    
    if use_market_data:
        ticker = st.text_input("Stock Ticker", value="AAPL")
        hist, info = fetch_stock_data(ticker)
        
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            hist_vol = calculate_historical_volatility(hist['Close'])
            st.success(f"Loaded {ticker} data")
            st.info(f"Current Price: ${current_price:.2f}")
            st.info(f"Historical Volatility: {hist_vol:.2%}")
        else:
            st.error("Failed to fetch market data")
            current_price = 100.0
            hist_vol = 0.2
    else:
        current_price = st.number_input("Current Asset Price ($)", value=100.0, step=1.0)
        hist_vol = 0.2
    
    # Override with selected option data if available
    if st.session_state.selected_option:
        st.info("🎯 **Using data from selected option!**")
        
        # Show selected option details
        opt = st.session_state.selected_option
        st.markdown(f"""
        **Selected:** {opt['option_type']} ${opt['strike']:.2f} Strike  
        **Expiry:** {opt['expiration'].strftime('%Y-%m-%d') if hasattr(opt['expiration'], 'strftime') else str(opt['expiration'])}  
        **Market IV:** {opt['implied_vol']:.1%}
        """)
        
        if st.button("🗑️ Clear Selection", type="secondary"):
            st.session_state.selected_option = None
            st.rerun()
        
        strike_default = st.session_state.selected_option['strike']
        vol_default = st.session_state.selected_option['implied_vol']
        
        # Add expiration date from selected option
        selected_exp = st.session_state.selected_option['expiration']
        if hasattr(selected_exp, 'date'):
            exp_default = selected_exp.date()
        else:
            exp_default = datetime.now().date() + timedelta(days=30)
    else:
        strike_default = 100.0
        vol_default = hist_vol
        exp_default = datetime.now().date() + timedelta(days=30)
    
    strike = st.number_input("Strike Price ($)", value=strike_default, step=1.0)
    
    # Improved date selection
    st.subheader("Expiration Date")
    expiry_date = st.date_input("Select Expiry", 
                               value=exp_default,
                               min_value=datetime.now().date())
    
    days_to_expiry = (expiry_date - datetime.now().date()).days
    st.info(f"📅 **Days to Expiry:** {days_to_expiry} days")
    
    time_to_maturity = days_to_expiry / 365.0
    
    # Improved volatility input with helper
    st.subheader("Volatility Settings")
    vol_input_type = st.radio("Input Type", ["Percentage", "Decimal"], horizontal=True)
    if vol_input_type == "Percentage":
        volatility_pct = st.number_input("Implied Volatility (%)", value=vol_default*100, step=1.0, format="%.1f")
        volatility = volatility_pct / 100
    else:
        volatility = st.number_input("Implied Volatility", value=vol_default, step=0.01, format="%.4f")
    
    # Common volatility references
    st.caption("📊 Typical ranges: Low (10-20%), Medium (20-40%), High (40%+)")
    
    # Improved interest rate with current market rates
    st.subheader("Risk-Free Rate")
    rate_input_type = st.radio("Rate Input", ["Percentage", "Decimal"], horizontal=True)
    if rate_input_type == "Percentage":
        interest_rate_pct = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25, format="%.2f")
        interest_rate = interest_rate_pct / 100
    else:
        interest_rate = st.number_input("Risk-Free Rate", value=0.05, step=0.0025, format="%.4f")
    
    # Reference rates
    st.caption("📈 Current US Treasury: 3M (~5.5%), 1Y (~5.0%), 10Y (~4.5%)")
    
    st.markdown("---")
    st.header("🔥 Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=max(0.01, volatility*0.5), step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=min(1.0, volatility*1.5), step=0.01)
    
    st.markdown("---")
    st.header("🎲 Monte Carlo Settings")
    num_simulations = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=100000, step=1000)

# Main content
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
    <h1 style="color: white; margin: 0; font-size: 3em; font-weight: 700;">🚀 Advanced Black-Scholes</h1>
    <h2 style="color: #f0f0f0; margin: 10px 0 0 0; font-weight: 300;">Option Pricing & Risk Analysis Platform</h2>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Pricing & Greeks", "🔥 Heatmaps", "🎲 Monte Carlo", "📊 Market Analysis", "💼 Portfolio", "⛓️ Options Chain"])

# Tab 1: Pricing and Greeks
with tab1:
    # Initialize model
    bs_model = BlackScholesModel(current_price, strike, time_to_maturity, interest_rate, volatility)
    
    # Calculate prices
    call_price = bs_model.call_price()
    put_price = bs_model.put_price()
    
    # Hero Section - Clean Pricing Display
    st.markdown("### 💰 Option Pricing")
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 3])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #90ee90, #32cd32); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <h3 style="margin: 0; color: #000; font-weight: bold;">CALL OPTION</h3>
            <h1 style="margin: 10px 0; color: #000; font-size: 2.5em;">${:.2f}</h1>
        </div>
        """.format(call_price), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffcccb, #ff6b6b); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <h3 style="margin: 0; color: #000; font-weight: bold;">PUT OPTION</h3>
            <h1 style="margin: 10px 0; color: #000; font-size: 2.5em;">${:.2f}</h1>
        </div>
        """.format(put_price), unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='width: 20px;'></div>", unsafe_allow_html=True)  # Spacer
    
    with col4:
        st.markdown("#### 📋 Contract Details")
        st.markdown(f"""
        **Underlying Price:** ${current_price:.2f}  
        **Strike Price:** ${strike:.2f}  
        **Expiration:** {expiry_date.strftime('%Y-%m-%d')} ({days_to_expiry} days)  
        **Implied Volatility:** {volatility:.1%}  
        **Risk-Free Rate:** {interest_rate:.2%}  
        """)
    
    # Key Metrics Row
    st.markdown("---")
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 Call Delta", f"{bs_model.delta_call():.3f}", 
                 help="Price sensitivity to $1 change in underlying")
    with col2:
        st.metric("📉 Put Delta", f"{bs_model.delta_put():.3f}", 
                 help="Price sensitivity to $1 change in underlying")
    with col3:
        st.metric("🌊 Gamma", f"{bs_model.gamma():.4f}", 
                 help="Delta sensitivity to price changes")
    with col4:
        st.metric("⏰ Theta", f"{bs_model.theta_call():.3f}", 
                 help="Daily time decay")
    with col5:
        st.metric("💨 Vega", f"{bs_model.vega():.3f}", 
                 help="Sensitivity to volatility changes")
    
    # Detailed Greeks Table
    st.markdown("---")
    st.markdown("### 🔢 Detailed Greeks Analysis")
    
    greeks_data = {
        'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
        'Call Value': [
            f"{bs_model.delta_call():.4f}",
            f"{bs_model.gamma():.6f}",
            f"{bs_model.theta_call():.4f}",
            f"{bs_model.vega():.4f}",
            f"{bs_model.rho_call():.4f}"
        ],
        'Put Value': [
            f"{bs_model.delta_put():.4f}",
            f"{bs_model.gamma():.6f}",
            f"{bs_model.theta_put():.4f}",
            f"{bs_model.vega():.4f}",
            f"{bs_model.rho_put():.4f}"
        ],
        'Interpretation': [
            'Price change per $1 move in underlying',
            'Delta change per $1 move in underlying',
            'Daily price decay (time value loss)',
            'Price change per 1% volatility increase',
            'Price change per 1% interest rate increase'
        ]
    }
    
    greeks_df = pd.DataFrame(greeks_data)
    st.dataframe(greeks_df, hide_index=True, use_container_width=True)
    
    # Greeks visualization with better formatting
    st.markdown("---")
    st.markdown("### 📈 Greeks Sensitivity Chart")
    
    # Create price range for Greeks analysis
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
    # Calculate Greeks across price range
    delta_call_range = []
    delta_put_range = []
    gamma_range = []
    
    for price in price_range:
        temp_model = BlackScholesModel(price, strike, time_to_maturity, interest_rate, volatility)
        delta_call_range.append(temp_model.delta_call())
        delta_put_range.append(temp_model.delta_put())
        gamma_range.append(temp_model.gamma())
    
    # Plot Greeks with improved formatting
    fig_greeks = go.Figure()
    
    fig_greeks.add_trace(go.Scatter(
        x=price_range, y=delta_call_range, 
        mode='lines', name='Call Delta', 
        line=dict(color='#32cd32', width=3)
    ))
    
    fig_greeks.add_trace(go.Scatter(
        x=price_range, y=delta_put_range, 
        mode='lines', name='Put Delta', 
        line=dict(color='#ff6b6b', width=3)
    ))
    
    fig_greeks.add_trace(go.Scatter(
        x=price_range, y=gamma_range, 
        mode='lines', name='Gamma', 
        line=dict(color='#4169e1', width=3)
    ))
    
    # Add reference lines with better annotations
    fig_greeks.add_vline(
        x=current_price, 
        line_dash="solid", 
        line_color="gray", 
        line_width=2,
        annotation_text="Current Price",
        annotation_position="top right",
        annotation=dict(
            x=current_price,
            y=1,
            xref="x",
            yref="paper",
            text="Current Price",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            ax=20,
            ay=-30,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="black", size=12)
        )
    )
    
    fig_greeks.add_vline(
        x=strike, 
        line_dash="dash", 
        line_color="orange", 
        line_width=2,
        annotation_text="Strike Price",
        annotation_position="top left",
        annotation=dict(
            x=strike,
            y=0.9,
            xref="x",
            yref="paper",
            text="Strike Price",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange",
            ax=-20,
            ay=-30,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="orange",
            borderwidth=1,
            font=dict(color="black", size=12)
        )
    )
    
    fig_greeks.update_layout(
        title="Greeks Sensitivity Analysis",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Greek Value",
        height=500,
        font=dict(size=14, color="white"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="white")
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True,
        xaxis=dict(gridcolor="gray", color="white"),
        yaxis=dict(gridcolor="gray", color="white"),
        title_font=dict(color="white", size=16)
    )
    
    st.plotly_chart(fig_greeks, use_container_width=True)

# Tab 2: Heatmaps
with tab2:
    st.subheader("Option Price Heatmaps")
    
    # Generate ranges with fewer points for cleaner visualization
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    
    # Calculate option prices for heatmap
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_model = BlackScholesModel(spot, strike, time_to_maturity, interest_rate, vol)
            call_prices[i, j] = temp_model.call_price()
            put_prices[i, j] = temp_model.put_price()
    
    # Create more aesthetically pleasing heatmaps with spacing
    st.markdown("### Call Option Prices")
    
    fig_call_heat = go.Figure(data=go.Heatmap(
        z=call_prices,
        x=np.round(spot_range, 2),
        y=np.round(vol_range * 100, 1),  # Convert to percentage
        colorscale='Viridis',
        text=np.round(call_prices, 2),
        texttemplate='$%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='Spot: $%{x}<br>Vol: %{y}%<br>Price: $%{z:.2f}<extra></extra>'
    ))
    
    fig_call_heat.update_layout(
        xaxis_title='Spot Price ($)',
        yaxis_title='Implied Volatility (%)',
        height=450,
        margin=dict(l=80, r=80, t=50, b=80),
        font=dict(size=14)
    )
    
    fig_call_heat.update_xaxes(tickmode='linear', dtick=np.round((spot_max-spot_min)/5, 0))
    fig_call_heat.update_yaxes(tickmode='linear', dtick=np.round((vol_max-vol_min)*100/5, 0))
    
    st.plotly_chart(fig_call_heat, use_container_width=True)
    
    st.markdown("### Put Option Prices")
    
    fig_put_heat = go.Figure(data=go.Heatmap(
        z=put_prices,
        x=np.round(spot_range, 2),
        y=np.round(vol_range * 100, 1),  # Convert to percentage
        colorscale='Reds',
        text=np.round(put_prices, 2),
        texttemplate='$%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='Spot: $%{x}<br>Vol: %{y}%<br>Price: $%{z:.2f}<extra></extra>'
    ))
    
    fig_put_heat.update_layout(
        xaxis_title='Spot Price ($)',
        yaxis_title='Implied Volatility (%)',
        height=450,
        margin=dict(l=80, r=80, t=50, b=80),
        font=dict(size=14)
    )
    
    fig_put_heat.update_xaxes(tickmode='linear', dtick=np.round((spot_max-spot_min)/5, 0))
    fig_put_heat.update_yaxes(tickmode='linear', dtick=np.round((vol_max-vol_min)*100/5, 0))
    
    st.plotly_chart(fig_put_heat, use_container_width=True)
    
    # 3D Surface plots
    st.subheader("3D Surface Plots")
    
    X, Y = np.meshgrid(spot_range, vol_range)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_3d_call = go.Figure(data=[go.Surface(x=X, y=Y, z=call_prices)])
        fig_3d_call.update_layout(
            title='Call Option Price Surface',
            scene=dict(
                xaxis_title='Spot Price ($)',
                yaxis_title='Volatility',
                zaxis_title='Option Price ($)'
            ),
            height=500
        )
        st.plotly_chart(fig_3d_call, use_container_width=True)
    
    with col2:
        fig_3d_put = go.Figure(data=[go.Surface(x=X, y=Y, z=put_prices)])
        fig_3d_put.update_layout(
            title='Put Option Price Surface',
            scene=dict(
                xaxis_title='Spot Price ($)',
                yaxis_title='Volatility',
                zaxis_title='Option Price ($)'
            ),
            height=500
        )
        st.plotly_chart(fig_3d_put, use_container_width=True)

# Tab 3: Monte Carlo Simulation
with tab3:
    st.subheader("Monte Carlo Option Pricing")
    
    # Run simulations
    mc_call_price, mc_call_se, simulated_prices = monte_carlo_option_price(
        current_price, strike, time_to_maturity, interest_rate, volatility, num_simulations, 'call'
    )
    mc_put_price, mc_put_se, _ = monte_carlo_option_price(
        current_price, strike, time_to_maturity, interest_rate, volatility, num_simulations, 'put'
    )
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MC Call Price", f"${mc_call_price:.2f}", f"SE: ±${mc_call_se:.4f}")
        st.metric("BS Call Price", f"${call_price:.2f}", f"Diff: ${abs(mc_call_price - call_price):.4f}")
    
    with col2:
        st.metric("MC Put Price", f"${mc_put_price:.2f}", f"SE: ±${mc_put_se:.4f}")
        st.metric("BS Put Price", f"${put_price:.2f}", f"Diff: ${abs(mc_put_price - put_price):.4f}")
    
    with col3:
        st.metric("Convergence", f"{(1 - abs(mc_call_price - call_price)/call_price)*100:.2f}%", "Call Option")
        st.metric("Simulations", f"{num_simulations:,}", "Paths Generated")
    
    # Price paths visualization
    st.subheader("Simulated Price Paths")
    
    # Generate sample paths for visualization
    num_paths_to_show = min(100, num_simulations)
    time_steps = 50
    dt = time_to_maturity / time_steps
    
    paths = np.zeros((num_paths_to_show, time_steps + 1))
    paths[:, 0] = current_price
    
    for i in range(num_paths_to_show):
        for t in range(1, time_steps + 1):
            Z = np.random.standard_normal()
            paths[i, t] = paths[i, t-1] * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
    
    # Plot paths
    fig_paths = go.Figure()
    
    for i in range(num_paths_to_show):
        fig_paths.add_trace(go.Scatter(
            x=np.linspace(0, time_to_maturity, time_steps + 1),
            y=paths[i, :],
            mode='lines',
            line=dict(width=0.5),
            opacity=0.3,
            showlegend=False
        ))
    
    # Add strike price line
    fig_paths.add_hline(y=strike, line_dash="dash", line_color="red", annotation_text="Strike Price")
    
    # Add mean path
    mean_path = np.mean(paths, axis=0)
    fig_paths.add_trace(go.Scatter(
        x=np.linspace(0, time_to_maturity, time_steps + 1),
        y=mean_path,
        mode='lines',
        line=dict(color='black', width=3),
        name='Mean Path'
    ))
    
    fig_paths.update_layout(
        title=f"Sample Price Paths ({num_paths_to_show} paths shown)",
        xaxis_title="Time (Years)",
        yaxis_title="Stock Price ($)",
        height=500
    )
    
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # Terminal price distribution
    st.subheader("Terminal Price Distribution")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=simulated_prices, nbinsx=50, name='Simulated Prices'))
    
    # Add reference lines with better positioning
    fig_dist.add_vline(
        x=strike, 
        line_dash="dash", 
        line_color="red",
        annotation=dict(
            x=strike,
            y=0.9,
            xref="x",
            yref="paper",
            text="Strike",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            ax=0,
            ay=-30,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red",
            borderwidth=1,
            font=dict(color="black", size=12)
        )
    )
    
    fig_dist.add_vline(
        x=current_price, 
        line_dash="dash", 
        line_color="green",
        annotation=dict(
            x=current_price,
            y=0.8,
            xref="x",
            yref="paper",
            text="Current",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            ax=0,
            ay=-30,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="green",
            borderwidth=1,
            font=dict(color="black", size=12)
        )
    )
    
    fig_dist.update_layout(
        title="Distribution of Terminal Stock Prices",
        xaxis_title="Stock Price ($)",
        yaxis_title="Frequency",
        height=400,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
        xaxis=dict(gridcolor="gray", color="white"),
        yaxis=dict(gridcolor="gray", color="white"),
        title_font=dict(color="white", size=16)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Tab 4: Market Analysis
with tab4:
    st.subheader("Market Analysis & Implied Volatility")
    
    if use_market_data and hist is not None:
        # Historical price chart
        st.subheader(f"{ticker} Historical Prices")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Volatility analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Volatility Analysis")
            
            # Calculate rolling volatility
            rolling_vol = hist['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='21-Day Rolling Vol'))
            fig_vol.add_hline(y=hist_vol, line_dash="dash", line_color="red", annotation_text="Annual Vol")
            
            fig_vol.update_layout(
                title="Historical Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            st.subheader("Returns Distribution")
            
            returns = hist['Close'].pct_change().dropna()
            
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns'))
            
            # Add normal distribution overlay
            x_range = np.linspace(returns.min(), returns.max(), 100)
            y_normal = norm.pdf(x_range, returns.mean(), returns.std()) * len(returns) * (returns.max() - returns.min()) / 50
            fig_returns.add_trace(go.Scatter(x=x_range, y=y_normal, mode='lines', name='Normal Distribution'))
            
            fig_returns.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Return",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_returns, use_container_width=True)
    
    # Implied Volatility Calculator
    st.subheader("Implied Volatility Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        market_call_price = st.number_input("Market Call Price ($)", value=call_price, step=0.01)
        if st.button("Calculate Call IV"):
            call_iv = implied_volatility(market_call_price, current_price, strike, time_to_maturity, interest_rate, 'call')
            st.success(f"Implied Volatility (Call): {call_iv:.2%}")
            st.info(f"Model Volatility: {volatility:.2%}")
            st.metric("IV/HV Ratio", f"{call_iv/volatility:.2f}x")
    
    with col2:
        market_put_price = st.number_input("Market Put Price ($)", value=put_price, step=0.01)
        if st.button("Calculate Put IV"):
            put_iv = implied_volatility(market_put_price, current_price, strike, time_to_maturity, interest_rate, 'put')
            st.success(f"Implied Volatility (Put): {put_iv:.2%}")
            st.info(f"Model Volatility: {volatility:.2%}")
            st.metric("IV/HV Ratio", f"{put_iv/volatility:.2f}x")

# Tab 5: Portfolio Analysis (moved from tab 5)
with tab5:
    st.subheader("Options Portfolio Builder")
    
    # Portfolio builder
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        position_type = st.selectbox("Position", ["Long Call", "Short Call", "Long Put", "Short Put"])
    with col2:
        position_strike = st.number_input("Strike", value=strike, step=1.0)
    with col3:
        position_quantity = st.number_input("Quantity", value=1, min_value=1)
    with col4:
        if st.button("Add Position"):
            st.session_state.portfolio.append({
                'type': position_type,
                'strike': position_strike,
                'quantity': position_quantity
            })
    
    if st.session_state.portfolio:
        # Display portfolio
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(portfolio_df)
        
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
        
        # Calculate portfolio P&L
        st.subheader("Portfolio Payoff Diagram")
        
        price_range_pnl = np.linspace(current_price * 0.5, current_price * 1.5, 200)
        portfolio_pnl = np.zeros_like(price_range_pnl)
        
        for position in st.session_state.portfolio:
            pos_type = position['type']
            pos_strike = position['strike']
            pos_qty = position['quantity']
            
            # Calculate position value at each price
            for i, price in enumerate(price_range_pnl):
                if pos_type == "Long Call":
                    model = BlackScholesModel(current_price, pos_strike, time_to_maturity, interest_rate, volatility)
                    premium = model.call_price()
                    payoff = np.maximum(price - pos_strike, 0) - premium
                    portfolio_pnl[i] += payoff * pos_qty
                elif pos_type == "Short Call":
                    model = BlackScholesModel(current_price, pos_strike, time_to_maturity, interest_rate, volatility)
                    premium = model.call_price()
                    payoff = premium - np.maximum(price - pos_strike, 0)
                    portfolio_pnl[i] += payoff * pos_qty
                elif pos_type == "Long Put":
                    model = BlackScholesModel(current_price, pos_strike, time_to_maturity, interest_rate, volatility)
                    premium = model.put_price()
                    payoff = np.maximum(pos_strike - price, 0) - premium
                    portfolio_pnl[i] += payoff * pos_qty
                elif pos_type == "Short Put":
                    model = BlackScholesModel(current_price, pos_strike, time_to_maturity, interest_rate, volatility)
                    premium = model.put_price()
                    payoff = premium - np.maximum(pos_strike - price, 0)
                    portfolio_pnl[i] += payoff * pos_qty
        
        # Plot P&L diagram
        fig_pnl = go.Figure()
        
        # Add P&L line
        fig_pnl.add_trace(go.Scatter(
            x=price_range_pnl,
            y=portfolio_pnl,
            mode='lines',
            name='Portfolio P&L',
            line=dict(color='blue', width=3)
        ))
        
        # Add break-even line
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
        
        # Add current price line
        fig_pnl.add_vline(x=current_price, line_dash="dash", line_color="green", annotation_text="Current Price")
        
        # Shade profit/loss regions
        fig_pnl.add_trace(go.Scatter(
            x=price_range_pnl,
            y=portfolio_pnl,
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_pnl.update_layout(
            title="Portfolio Payoff Diagram",
            xaxis_title="Underlying Price at Expiry ($)",
            yaxis_title="Profit/Loss ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Portfolio Greeks
        st.subheader("Portfolio Greeks")
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for position in st.session_state.portfolio:
            pos_type = position['type']
            pos_strike = position['strike']
            pos_qty = position['quantity']
            
            model = BlackScholesModel(current_price, pos_strike, time_to_maturity, interest_rate, volatility)
            
            if "Call" in pos_type:
                delta = model.delta_call()
                theta = model.theta_call()
                rho = model.rho_call()
            else:
                delta = model.delta_put()
                theta = model.theta_put()
                rho = model.rho_put()
            
            # Adjust for short positions
            if "Short" in pos_type:
                delta *= -1
                theta *= -1
                rho *= -1
                gamma = -model.gamma()
                vega = -model.vega()
            else:
                gamma = model.gamma()
                vega = model.vega()
            
            total_delta += delta * pos_qty
            total_gamma += gamma * pos_qty
            total_theta += theta * pos_qty
            total_vega += vega * pos_qty
            total_rho += rho * pos_qty
        
        # Display portfolio Greeks
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Delta", f"{total_delta:.4f}")
        with col2:
            st.metric("Portfolio Gamma", f"{total_gamma:.4f}")
        with col3:
            st.metric("Portfolio Theta", f"{total_theta:.4f}")
        with col4:
            st.metric("Portfolio Vega", f"{total_vega:.4f}")
        with col5:
            st.metric("Portfolio Rho", f"{total_rho:.4f}")
        
        # Risk metrics
        st.subheader("Risk Analysis")
        
        # Calculate maximum profit and loss
        max_profit = np.max(portfolio_pnl)
        max_loss = np.min(portfolio_pnl)
        
        # Find breakeven points
        breakeven_indices = np.where(np.diff(np.sign(portfolio_pnl)))[0]
        breakeven_prices = price_range_pnl[breakeven_indices]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Maximum Profit", f"${max_profit:.2f}" if max_profit != np.inf else "Unlimited")
            st.metric("Maximum Loss", f"${max_loss:.2f}" if max_loss != -np.inf else "Unlimited")
        
        with col2:
            if len(breakeven_prices) > 0:
                st.metric("Breakeven Price(s)", f"${', '.join([f'{p:.2f}' for p in breakeven_prices])}")
            else:
                st.metric("Breakeven Price(s)", "None")
            
            # Probability of profit
            if time_to_maturity > 0:
                prob_profit = 0
                for i, price in enumerate(price_range_pnl):
                    if portfolio_pnl[i] > 0:
                        # Simple probability calculation using lognormal distribution
                        prob = norm.pdf(np.log(price/current_price), 
                                      (interest_rate - 0.5*volatility**2)*time_to_maturity,
                                      volatility*np.sqrt(time_to_maturity))
                        prob_profit += prob * (price_range_pnl[1] - price_range_pnl[0])
                
                st.metric("Probability of Profit", f"{prob_profit:.1%}")
        
        with col3:
            # Portfolio summary
            st.info(f"""
            **Portfolio Summary**
            - Positions: {len(st.session_state.portfolio)}
            - Net Delta: {total_delta:.2f}
            - Daily Theta: ${total_theta:.2f}
            - Vega Risk: ${total_vega:.2f} per 1% vol
            """)

# Tab 6: Options Chain
with tab6:
    st.subheader("Live Options Chain")
    
    if use_market_data:
        options_df, expirations = fetch_options_chain(ticker)
        
        if options_df is not None:
            st.success(f"Loaded options chain for {ticker}")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_expiry = st.selectbox("Expiration Date", 
                                             options=['All'] + list(expirations[:5]))
            
            with col2:
                option_type = st.selectbox("Option Type", ['All', 'Call', 'Put'])
            
            with col3:
                moneyness = st.selectbox("Moneyness", 
                                       ['All', 'ITM', 'ATM', 'OTM'])
            
            # Apply filters
            filtered_df = options_df.copy()
            
            if selected_expiry != 'All':
                filtered_df = filtered_df[filtered_df['expiration'] == selected_expiry]
            
            if option_type != 'All':
                filtered_df = filtered_df[filtered_df['type'] == option_type]
            
            # Moneyness filter
            if moneyness != 'All':
                if moneyness == 'ITM':
                    if option_type == 'Call' or option_type == 'All':
                        call_itm = filtered_df[(filtered_df['type'] == 'Call') & 
                                              (filtered_df['strike'] < current_price)]
                    else:
                        call_itm = pd.DataFrame()
                    
                    if option_type == 'Put' or option_type == 'All':
                        put_itm = filtered_df[(filtered_df['type'] == 'Put') & 
                                             (filtered_df['strike'] > current_price)]
                    else:
                        put_itm = pd.DataFrame()
                    
                    filtered_df = pd.concat([call_itm, put_itm])
                
                elif moneyness == 'ATM':
                    atm_range = current_price * 0.02  # 2% range
                    filtered_df = filtered_df[
                        (filtered_df['strike'] >= current_price - atm_range) & 
                        (filtered_df['strike'] <= current_price + atm_range)
                    ]
                
                else:  # OTM
                    if option_type == 'Call' or option_type == 'All':
                        call_otm = filtered_df[(filtered_df['type'] == 'Call') & 
                                              (filtered_df['strike'] > current_price)]
                    else:
                        call_otm = pd.DataFrame()
                    
                    if option_type == 'Put' or option_type == 'All':
                        put_otm = filtered_df[(filtered_df['type'] == 'Put') & 
                                             (filtered_df['strike'] < current_price)]
                    else:
                        put_otm = pd.DataFrame()
                    
                    filtered_df = pd.concat([call_otm, put_otm])
            
            # Display options chain
            if not filtered_df.empty:
                # Select relevant columns
                display_columns = ['type', 'strike', 'lastPrice', 'bid', 'ask', 
                                 'volume', 'openInterest', 'impliedVolatility', 'expiration']
                
                # Format the dataframe
                display_df = filtered_df[display_columns].copy()
                display_df['impliedVolatility'] = display_df['impliedVolatility'] * 100
                
                # Rename columns for clarity
                display_df.columns = ['Type', 'Strike', 'Last', 'Bid', 'Ask', 
                                    'Volume', 'Open Int', 'IV (%)', 'Expiry']
                
                # Style the dataframe
                def highlight_itm(row):
                    if row['Type'] == 'Call' and row['Strike'] < current_price:
                        return ['background-color: #90ee90'] * len(row)
                    elif row['Type'] == 'Put' and row['Strike'] > current_price:
                        return ['background-color: #ffcccb'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_itm, axis=1)
                styled_df = styled_df.format({
                    'Strike': '${:.2f}',
                    'Last': '${:.2f}',
                    'Bid': '${:.2f}',
                    'Ask': '${:.2f}',
                    'Volume': '{:,.0f}',
                    'Open Int': '{:,.0f}',
                    'IV (%)': '{:.1f}%'
                })
                
                st.dataframe(styled_df, height=400, use_container_width=True)
                
                # Add option selection functionality
                st.subheader("📋 Select an Option to Analyze")
                
                # Create selection interface
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    available_strikes = sorted(filtered_df['strike'].unique())
                    selected_strike = st.selectbox(
                        "Strike Price", 
                        available_strikes,
                        index=len(available_strikes)//2 if available_strikes else 0,
                        format_func=lambda x: f"${x:.2f}"
                    )
                
                with col2:
                    available_types = filtered_df['type'].unique()
                    selected_type = st.selectbox("Option Type", available_types)
                
                with col3:
                    # Filter by selected strike and type
                    option_subset = filtered_df[
                        (filtered_df['strike'] == selected_strike) & 
                        (filtered_df['type'] == selected_type)
                    ]
                    
                    if not option_subset.empty:
                        available_expirations = option_subset['expiration'].unique()
                        selected_expiration = st.selectbox(
                            "Expiration", 
                            available_expirations,
                            format_func=lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                        )
                        
                        # Get the selected option data
                        selected_option_data = option_subset[
                            option_subset['expiration'] == selected_expiration
                        ].iloc[0]
                        
                        if st.button("🔄 Load This Option", type="primary", use_container_width=True):
                            # Update session state with selected option
                            st.session_state.selected_option = {
                                'strike': selected_option_data['strike'],
                                'option_type': selected_option_data['type'],
                                'expiration': selected_option_data['expiration'],
                                'current_price': selected_option_data['lastPrice'],
                                'implied_vol': selected_option_data['impliedVolatility'],
                                'bid': selected_option_data['bid'],
                                'ask': selected_option_data['ask'],
                                'volume': selected_option_data['volume'],
                                'open_interest': selected_option_data['openInterest']
                            }
                            st.success(f"✅ Loaded {selected_type} option with ${selected_strike:.2f} strike!")
                            st.balloons()
                            st.rerun()
                
                # Display loaded option info
                if st.session_state.selected_option:
                    st.markdown("---")
                    st.subheader("🎯 Currently Loaded Option")
                    
                    opt = st.session_state.selected_option
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Type", opt['option_type'])
                        st.metric("Strike", f"${opt['strike']:.2f}")
                    with col2:
                        st.metric("Market Price", f"${opt['current_price']:.2f}")
                        st.metric("Implied Vol", f"{opt['implied_vol']*100:.1f}%")
                    with col3:
                        st.metric("Bid/Ask", f"${opt['bid']:.2f} / ${opt['ask']:.2f}")
                        st.metric("Volume", f"{opt['volume']:,.0f}")
                    with col4:
                        exp_str = opt['expiration'].strftime('%Y-%m-%d') if hasattr(opt['expiration'], 'strftime') else str(opt['expiration'])
                        st.metric("Expiration", exp_str)
                        st.metric("Open Interest", f"{opt['open_interest']:,.0f}")
                    
                    # Update sidebar values button
                    if st.button("📊 Update Analysis with This Option", type="secondary", use_container_width=True):
                        st.info("💡 **Instructions:** Go to the sidebar and the values have been updated! You can now analyze this specific option in the 'Pricing & Greeks' tab.")
                        # Note: We'll handle the actual sidebar update through JavaScript or rerun
            else:
                st.warning("No options data available for the selected filters")
        else:
            st.error(f"Failed to fetch options chain for {ticker}")
    else:
        st.info("Enable 'Use Real Market Data' in the sidebar to view live options chains")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Advanced Black-Scholes Option Pricing Model | Built with Streamlit</p>
    <p>Features: Real-time pricing, Greeks calculation, Monte Carlo simulation, Market analysis, Portfolio builder</p>
</div>
""", unsafe_allow_html=True)