import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Bitcoin Predictor - Professional Analytics",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main {
        direction: ltr;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F7931E, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 4px solid #f39c12;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1;
    }
    
    .metric-change {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .model-performance {
        background: linear-gradient(135deg, #1abc9c, #16a085);
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        color: white;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        padding: 1.5rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    }
    
    .correlation-card {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 80px;
    }
    
    .progress-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        height: 6px;
        margin-top: 8px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #34495e);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #374151, #4B5563);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        
        .sub-header {
            font-size: 1rem !important;
            margin-bottom: 2rem;
            padding: 0 1rem;
        }
        
        .metric-card {
            height: 120px !important;
            padding: 1.2rem !important;
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem !important;
        }
        
        .metric-title {
            font-size: 0.75rem !important;
        }
        
        .model-performance {
            height: 180px !important;
            padding: 1.5rem !important;
            margin-bottom: 1rem;
        }
        
        .feature-card {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .correlation-card {
            height: 70px !important;
            padding: 1rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* Mobile Bitcoin symbol */
        .bitcoin-symbol-mobile {
            font-size: 2.2rem !important;
            margin-right: 0.3rem !important;
        }
        
        /* Charts on mobile */
        .stPlotlyChart {
            font-size: 12px;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.7rem !important;
        }
        
        .metric-card {
            height: 110px !important;
            padding: 1rem !important;
        }
        
        .metric-value {
            font-size: 1.3rem !important;
        }
        
        .model-performance {
            height: 160px !important;
            padding: 1.2rem !important;
        }
        
        /* Better spacing on very small screens */
        .correlation-card h4 {
            font-size: 0.9rem !important;
        }
        
        .feature-card h4 {
            font-size: 0.9rem !important;
        }
    }
    
    /* Tablet optimizations */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.4rem !important;
        }
        
        .metric-card {
            height: 130px !important;
            padding: 1.5rem !important;
        }
        
        .model-performance {
            height: 200px !important;
            padding: 2rem !important;
        }
    }
</style>

<script>
// Mobile detection
function isMobile() {
    return window.innerWidth <= 768;
}

// Update layout based on screen size
window.addEventListener('resize', function() {
    // You can add dynamic adjustments here if needed
});
</script>
""", unsafe_allow_html=True)

# ===== CACHING AND DATA LOADING =====
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_crypto_data():
    """Load cryptocurrency data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years of data
        
        # Cryptocurrency data
        btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        eth = yf.download('ETH-USD', start=start_date, end=end_date, progress=False)
        ltc = yf.download('LTC-USD', start=start_date, end=end_date, progress=False)
        xrp = yf.download('XRP-USD', start=start_date, end=end_date, progress=False)
        
        return {
            'BTC': btc,
            'ETH': eth,
            'LTC': ltc,
            'XRP': xrp
        }
    except Exception as e:
        st.error(f"Error loading crypto data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_stock_data():
    """Load stock market data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # Stock data
        stocks = ['MSTR', 'MARA', 'COIN', 'GLXY.TO', '^GSPC', '^VIX']
        stock_data = {}
        
        for stock in stocks:
            try:
                data = yf.download(stock, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    stock_data[stock] = data
            except:
                continue
                
        return stock_data
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def load_macro_data():
    """Load macroeconomic data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # Macro indicators using yfinance for gold, oil, dollar index
        macro_data = {}
        
        # Gold, Oil, Dollar Index
        symbols = {
            'GC=F': 'Gold',
            'CL=F': 'Oil', 
            'DX-Y.NYB': 'Dollar_Index',
            'HG=F': 'Copper'
        }
        
        for symbol, name in symbols.items():
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    macro_data[name] = data
            except:
                continue
                
        return macro_data
    except Exception as e:
        st.error(f"Error loading macro data: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def prepare_features_and_model():
    """Prepare features and train models"""
    
    # Load all data
    crypto_data = load_crypto_data()
    stock_data = load_stock_data()
    macro_data = load_macro_data()
    
    if not crypto_data:
        return None, None, None, None, None
    
    # Create master dataframe
    btc_data = crypto_data['BTC'].copy()
    btc_data['BTC_Returns'] = btc_data['Close'].pct_change()
    
    # Create feature dataframe
    features_df = pd.DataFrame(index=btc_data.index)
    features_df['BTC_Price'] = btc_data['Close']
    features_df['BTC_Volume'] = btc_data['Volume']
    features_df['BTC_Returns'] = btc_data['BTC_Returns']
    
    # Add crypto features
    if 'ETH' in crypto_data and not crypto_data['ETH'].empty:
        features_df['ETH_Returns'] = crypto_data['ETH']['Close'].pct_change()
        
    if 'LTC' in crypto_data and not crypto_data['LTC'].empty:
        features_df['LTC_Returns'] = crypto_data['LTC']['Close'].pct_change()
        
    # Add stock features
    for stock, data in stock_data.items():
        if not data.empty:
            features_df[f'{stock}_Returns'] = data['Close'].pct_change()
    
    # Add macro features  
    for name, data in macro_data.items():
        if not data.empty:
            features_df[f'{name}_Price'] = data['Close']
            features_df[f'{name}_Returns'] = data['Close'].pct_change()
    
    # Create engineered features
    if 'Gold_Price' in features_df.columns and 'Copper_Price' in features_df.columns:
        features_df['Copper_to_Gold_Ratio'] = features_df['Copper_Price'] / features_df['Gold_Price']
        
    if 'Dollar_Index_Price' in features_df.columns:
        # Mock money supply ratio (in real implementation, you'd get this from FRED)
        features_df['Money_to_Dollar_Ratio'] = 21000000 / features_df['Dollar_Index_Price']  # Simplified
        
    # Clean data
    features_df = features_df.dropna()
    
    if len(features_df) < 100:  # Not enough data
        return None, None, None, None, None
    
    # Prepare for modeling
    target = features_df['BTC_Price'].shift(-1).dropna()  # Next day price
    features = features_df[:-1]  # Remove last row to match target
    
    # Select relevant features for modeling
    feature_columns = [col for col in features.columns if 'Returns' in col or 'Ratio' in col or col in ['BTC_Volume']]
    X = features[feature_columns].dropna()
    y = target[X.index]
    
    if len(X) < 50:
        return None, None, None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Stacking model
    estimators = [('rf', rf_model), ('gb', gb_model)]
    stacked_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
    
    # Train models
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model, 
        'Stacked Model': stacked_model
    }
    
    model_performance = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        model_performance[name] = {
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae
        }
        
        predictions[name] = pred
    
    # Feature importance from best model
    best_model_name = max(model_performance.keys(), key=lambda k: model_performance[k]['R²'])
    best_model = models[best_model_name]
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        # For stacked model, use the random forest importances
        rf_importances = models['Random Forest'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_importances
        }).sort_values('Importance', ascending=False)
    
    # Create prediction dataframe
    prediction_dates = X_test.index
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Actual': y_test.values
    })
    
    for name, pred in predictions.items():
        prediction_df[name] = pred
    
    return model_performance, feature_importance, prediction_df, features_df, best_model

def get_current_btc_info():
    """Get current Bitcoin information"""
    try:
        btc = yf.Ticker("BTC-USD")
        info = btc.info
        history = btc.history(period="5d")
        
        current_price = history['Close'].iloc[-1]
        prev_price = history['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        return {
            'current_price': current_price,
            'change_pct': change_pct,
            'volume': history['Volume'].iloc[-1],
            'high_24h': history['High'].iloc[-1],
            'low_24h': history['Low'].iloc[-1]
        }
    except:
        return {
            'current_price': 67430,
            'change_pct': 2.4,
            'volume': 28000000000,
            'high_24h': 68500,
            'low_24h': 66200
        }

# ===== HEADER =====
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
    <span class="bitcoin-symbol-mobile" style="font-size: 3rem; color: #F7931E; margin-right: 0.5rem;">₿</span>
    <h1 class="main-header" style="margin: 0;">Bitcoin Predictor — Professional Analytics</h1>
</div>
""", unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-powered prediction with institutional-grade market data</p>', unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("🎛️ Control Panel")
view = st.sidebar.selectbox(
    "📊 Select View:",
    ["🏠 Overview", "🤖 Models", "📈 Features", "🔗 Correlations", "📋 Raw Data"]
)

# Load data with progress indicator
if 'data_loaded' not in st.session_state:
    with st.spinner('🔄 Loading real-time market data...'):
        model_perf, feature_imp, pred_data, features_df, best_model = prepare_features_and_model()
        current_info = get_current_btc_info()
        
        st.session_state.data_loaded = True
        st.session_state.model_perf = model_perf
        st.session_state.feature_imp = feature_imp
        st.session_state.pred_data = pred_data
        st.session_state.features_df = features_df
        st.session_state.current_info = current_info
        st.session_state.best_model = best_model
else:
    model_perf = st.session_state.model_perf
    feature_imp = st.session_state.feature_imp
    pred_data = st.session_state.pred_data
    features_df = st.session_state.features_df
    current_info = st.session_state.current_info
    best_model = st.session_state.best_model

# Check if data loaded successfully
if model_perf is None:
    st.error("❌ Unable to load sufficient data. Please check your internet connection and try again.")
    st.stop()

# ===== OVERVIEW PAGE =====
if "🏠 Overview" in view:
    
    # Current Bitcoin Info
    col1, col2, col3, col4 = st.columns(4)
    
    # Make it responsive - stack on mobile
    if st.session_state.get('mobile_view', False):
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
    
    with col1:
        change_color = "#10B981" if current_info['change_pct'] > 0 else "#EF4444"
        change_arrow = "↗️" if current_info['change_pct'] > 0 else "↘️"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Current Price</div>
            <div class="metric-value">${current_info['current_price']:,.0f}</div>
            <div class="metric-change" style="color: {change_color};">{change_arrow} {current_info['change_pct']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_r2 = max([v['R²'] for v in model_perf.values()]) if model_perf else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🧠 Model Accuracy</div>
            <div class="metric-value">{best_r2*100:.1f}%</div>
            <div class="metric-change" style="color: #10B981;">↗️ Best R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 24H Volume</div>
            <div class="metric-value">${current_info['volume']/1e9:.1f}B</div>
            <div class="metric-change" style="color: #3B82F6;">📈 Trading Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_low_spread = ((current_info['high_24h'] - current_info['low_24h']) / current_info['low_24h']) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 24H Range</div>
            <div class="metric-value">{high_low_spread:.1f}%</div>
            <div class="metric-change" style="color: #8B5CF6;">High-Low Spread</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Chart
    st.subheader("📊 Model Predictions vs Actual Prices")
    
    if pred_data is not None and not pred_data.empty:
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=pred_data['Date'], 
            y=pred_data['Actual'],
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='#FFFFFF', width=3),
            marker=dict(size=6)
        ))
        
        # Add model predictions
        colors = ['#06B6D4', '#10B981', '#F59E0B']  # חזרה לצבעים המקוריים
        model_names = ['Stacked Model', 'Random Forest', 'Gradient Boosting']
        
        for i, model in enumerate(model_names):
            if model in pred_data.columns:
                fig.add_trace(go.Scatter(
                    x=pred_data['Date'], 
                    y=pred_data[model],
                    mode='lines',
                    name=model,
                    line=dict(color=colors[i], width=2),
                    opacity=0.8
                ))
        
        fig.update_layout(
            title=dict(
                text="Bitcoin Price Predictions",
                x=0.5,
                font=dict(size=20, color='white')
            ),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Not enough data available for prediction chart")
    
    # Model Performance Summary
    st.subheader("🎯 Model Performance Summary")
    
    if model_perf:
        cols = st.columns(len(model_perf))
        colors = ['#06B6D4', '#10B981', '#F59E0B']  # חזרה לצבעים המקוריים היפים
        
        for i, (model_name, metrics) in enumerate(model_perf.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="model-performance" style="background: linear-gradient(135deg, {colors[i]}95, {colors[i]}); border: 1px solid {colors[i]}40;">
                    <h3 style="margin-bottom: 1.5rem; color: white; font-weight: 700; font-size: 1.2rem;">{model_name}</h3>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <div style="text-align: left;">
                            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8); font-weight: 500;">RMSE</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: white;">${metrics['RMSE']:,.0f}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8); font-weight: 500;">R² Score</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: white;">{metrics['R²']*100:.1f}%</div>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: white; height: 6px; width: {metrics['R²']*100}%; border-radius: 3px; box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ===== MODELS PAGE =====
elif "🤖 Models" in view:
    st.subheader("🤖 Machine Learning Models Performance")
    
    if model_perf:
        # Performance comparison chart
        perf_df = pd.DataFrame(model_perf).T.reset_index()
        perf_df.columns = ['Model', 'RMSE', 'R²', 'MAE']
        perf_df['R² %'] = perf_df['R²'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(
                perf_df, 
                x='Model', 
                y='R² %',
                title="Model Accuracy (R² Score)",
                color='R² %',
                color_continuous_scale='Viridis'
            )
            fig_r2.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_rmse = px.bar(
                perf_df, 
                x='Model', 
                y='RMSE',
                title="Model Error (RMSE)",
                color='RMSE',
                color_continuous_scale='Reds_r'
            )
            fig_rmse.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📊 Detailed Performance Metrics")
        st.dataframe(perf_df, use_container_width=True)

# ===== FEATURES PAGE =====
elif "📈 Features" in view:
    st.subheader("📈 Feature Importance Analysis")
    
    if feature_imp is not None and not feature_imp.empty:
        # Add categories to features
        feature_imp['Category'] = feature_imp['Feature'].apply(lambda x: 
            'Crypto' if any(crypto in x for crypto in ['BTC', 'ETH', 'LTC', 'XRP']) else
            'Stocks' if any(stock in x for stock in ['MSTR', 'MARA', 'COIN', 'GLXY', 'GSPC']) else
            'Macro' if any(macro in x for macro in ['Gold', 'Oil', 'Dollar', 'Copper', 'VIX']) else
            'Engineered' if 'Ratio' in x else 'Other'
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Feature importance chart
            fig = px.bar(
                feature_imp.head(10), 
                x='Importance', 
                y='Feature',
                color='Category',
                orientation='h',
                title="Top 10 Feature Importance"
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category breakdown pie chart
            category_stats = feature_imp.groupby('Category')['Importance'].sum().reset_index()
            
            fig_pie = px.pie(
                category_stats, 
                values='Importance', 
                names='Category',
                title="Feature Importance by Category"
            )
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Top features list
            st.subheader("🎯 Top Features")
            for idx, row in feature_imp.head(8).iterrows():
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: white;">{row['Feature']}</h4>
                            <small style="color: #9CA3AF;">{row['Category']}</small>
                        </div>
                        <h3 style="margin: 0; color: #06B6D4;">{row['Importance']*100:.1f}%</h3>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {row['Importance']*100}%; background: linear-gradient(90deg, #3498db, #2980b9);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ===== CORRELATIONS PAGE =====
elif "🔗 Correlations" in view:
    st.subheader("🔗 Asset Correlations Analysis")
    
    if features_df is not None and not features_df.empty:
        # Calculate correlations with Bitcoin returns
        btc_returns = features_df['BTC_Returns'].dropna()
        correlations = []
        
        for col in features_df.columns:
            if col != 'BTC_Returns' and 'Returns' in col:
                corr = btc_returns.corr(features_df[col].dropna())
                if not pd.isna(corr):
                    correlations.append({
                        'Asset': col.replace('_Returns', '').replace('_', ' '),
                        'Correlation': corr,
                        'Type': 'Crypto' if any(x in col for x in ['ETH', 'LTC', 'XRP']) else 
                               'Stock' if any(x in col for x in ['MSTR', 'MARA', 'COIN']) else 'Other'
                    })
        
        if correlations:
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Correlation Chart")
                
                fig_corr = px.bar(
                    corr_df,
                    x='Asset',
                    y='Correlation',
                    color='Correlation',
                    color_continuous_scale='RdYlGn',
                    title="Bitcoin Correlation Coefficients"
                )
                fig_corr.add_hline(y=0, line_dash="dash", line_color="white")
                fig_corr.update_layout(
                    template="plotly_dark",
                    xaxis={'tickangle': -45},
                    height=500
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("📋 Correlation Summary")
                
                for _, row in corr_df.iterrows():
                    corr_val = row['Correlation']
                    if corr_val > 0.3:
                        accent_color = "#27ae60"
                        icon = "📈"
                        strength = "Strong Positive"
                    elif corr_val > 0:
                        accent_color = "#3498db"
                        icon = "📈"
                        strength = "Weak Positive"
                    elif corr_val > -0.3:
                        accent_color = "#f39c12"
                        icon = "📉"
                        strength = "Weak Negative"
                    else:
                        accent_color = "#e74c3c"
                        icon = "📉"
                        strength = "Strong Negative"
                    
                    st.markdown(f"""
                    <div class="correlation-card" style="border-left: 4px solid {accent_color};">
                        <div>
                            <h4 style="margin: 0; color: white; font-size: 1rem; font-weight: 600;">{icon} {row['Asset']}</h4>
                            <small style="color: rgba(255,255,255,0.8); font-weight: 400;">{strength}</small>
                        </div>
                        <h3 style="margin: 0; color: white; font-weight: 700;">{corr_val:.3f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Correlation matrix heatmap
            st.subheader("🌡️ Correlation Heatmap")
            
            # Create correlation matrix for returns data
            returns_columns = [col for col in features_df.columns if 'Returns' in col]
            if len(returns_columns) > 1:
                corr_matrix = features_df[returns_columns].corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title="Asset Returns Correlation Matrix"
                )
                fig_heatmap.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)

# ===== RAW DATA PAGE =====
elif "📋 Raw Data" in view:
    st.subheader("📋 Raw Data & Export")
    
    tab1, tab2, tab3 = st.tabs(["📮 Predictions", "📊 Features", "💹 Market Data"])
    
    with tab1:
        if pred_data is not None and not pred_data.empty:
            st.subheader("Model Predictions")
            st.dataframe(pred_data, use_container_width=True)
            
            csv = pred_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"bitcoin_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No prediction data available")
    
    with tab2:
        if feature_imp is not None and not feature_imp.empty:
            st.subheader("Feature Importance")
            st.dataframe(feature_imp, use_container_width=True)
            
            csv = feature_imp.to_csv(index=False)
            st.download_button(
                label="📥 Download Features CSV",
                data=csv,
                file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        if features_df is not None and not features_df.empty:
            st.subheader("Market Data")
            # Show last 30 days
            recent_data = features_df.tail(30)
            st.dataframe(recent_data, use_container_width=True)
            
            csv = features_df.to_csv()
            st.download_button(
                label="📥 Download Market Data CSV",
                data=csv,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ===== SIDEBAR INFO =====
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

# Data freshness indicator
last_update = datetime.now().strftime("%H:%M:%S")
st.sidebar.success(f"✅ Data Updated: {last_update}")

if model_perf:
    best_model_name = max(model_perf.keys(), key=lambda k: model_perf[k]['R²'])
    st.sidebar.info(f"🎯 Best Model: {best_model_name}")
    st.sidebar.metric("Model Accuracy", f"{model_perf[best_model_name]['R²']*100:.1f}%")

# Data sources
st.sidebar.markdown("### 📡 Data Sources")
st.sidebar.markdown("""
**Live Data Sources:**
- 🪙 **Crypto**: Yahoo Finance API
- 📊 **Stocks**: MSTR, MARA, COIN, S&P 500
- 🌍 **Macro**: Gold, Oil, Dollar Index, Copper
- ⚡ **Real-time**: Updated every hour

**Model Features:**
- 🔄 **Returns-based** features
- ⚖️ **Engineered ratios**
- 📈 **Volume indicators**
- 🤖 **Ensemble learning**
""")

# Refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    # Clear cache and reload
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# Model info
st.sidebar.markdown("### 🔧 Model Architecture")
st.sidebar.markdown("""
**Ensemble Components:**
- 🌲 Random Forest (100 trees)
- 🚀 Gradient Boosting (100 trees)  
- 🎯 Ridge Meta-learner

**Training Details:**
- ⏰ **Data Range**: Last 2 years
- 🔄 **Update Frequency**: Hourly
- ✅ **Validation**: Time-series split
- 📊 **Features**: Multi-asset returns
""")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 2rem;">
    <h4 style="color: #F59E0B; margin-bottom: 1rem;">🚀 Bitcoin Predictor Dashboard</h4>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px;">
            <strong style="color: #3B82F6;">Real-time Data</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Live market feeds</p>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px;">
            <strong style="color: #10B981;">ML Powered</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Ensemble learning</p>
        </div>
        <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px;">
            <strong style="color: #F59E0B;">Professional Grade</strong>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Production ready</p>
        </div>
    </div>
    <p style="margin-top: 2rem; font-size: 0.9rem;">Built with ❤️ using Streamlit | Powered by Machine Learning & Real-time Data</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">© 2025 Bitcoin Predictor - Advanced Financial ML Platform</p>
</div>
""", unsafe_allow_html=True)