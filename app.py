import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Bitcoin Predictor - Macro & Markets",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f39c12;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f39c12;
    }
    .model-performance {
        background: linear-gradient(135deg, #1abc9c, #16a085);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .correlation-positive {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem;
    }
    .correlation-negative {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #34495e);
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<h1 class="main-header">🪙 Bitcoin Predictor — Macro & Markets</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered prediction with macroeconomic indicators</p>', unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("🎛️ Control Panel")
view = st.sidebar.selectbox(
    "📊 Select View:",
    ["🏠 Overview", "🤖 Models", "📈 Features", "🔗 Correlations", "📋 Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=[datetime(2024, 1, 1), datetime.now()],
    key="date_range"
)

# ===== LOAD DATA FUNCTION =====
@st.cache_data
def load_sample_data():
    """Generate sample data based on your notebook structure"""
    
    # Mock model performance (based on your results)
    model_performance = {
        'Stacked Model': {'RMSE': 4847.23, 'R²': 0.8934},
        'Random Forest': {'RMSE': 5124.67, 'R²': 0.8756},
        'Gradient Boosting': {'RMSE': 5298.45, 'R²': 0.8642}
    }
    
    # Mock feature importance (from your analysis)
    feature_importance = pd.DataFrame({
        'Feature': [
            'Ethereum_Returns', 'MicroStrategy_Returns', 'Money_to_Dollar_Ratio',
            'Market Volatility (VIX)', 'Copper_to_Gold_Ratio', 'S&P 500_Returns', 
            'Global_Velocity', 'Federal Funds Rate', 'Litecoin_Returns', 'GDP'
        ],
        'Importance': [0.234, 0.187, 0.156, 0.143, 0.124, 0.098, 0.058, 0.045, 0.038, 0.023],
        'Category': ['Crypto', 'Stocks', 'Macro', 'Macro', 'Commodities', 'Indices', 
                    'Macro', 'Macro', 'Crypto', 'Macro']
    })
    
    # Mock correlation data
    correlation_data = pd.DataFrame({
        'Asset': ['Ethereum', 'MicroStrategy', 'Marathon Digital', 'Coinbase', 
                 'S&P 500', 'Gold Futures', 'U.S. Dollar Index', 'Federal Funds Rate'],
        'Correlation': [0.89, 0.76, 0.72, 0.68, 0.45, -0.23, -0.34, -0.45],
        'Type': ['Crypto', 'Stock', 'Stock', 'Stock', 'Index', 'Commodity', 'Currency', 'Macro']
    })
    
    # Generate prediction data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-08-24', freq='D')
    base_price = 45000
    
    actual_prices = []
    stacked_pred = []
    rf_pred = []
    gb_pred = []
    
    for i, date in enumerate(dates):
        # Simulate price trend
        trend = base_price + i * 100 + np.random.normal(0, 5000)
        actual = max(trend, 20000)  # Minimum price floor
        
        actual_prices.append(actual)
        stacked_pred.append(actual + np.random.normal(0, 2000))
        rf_pred.append(actual + np.random.normal(0, 3000))
        gb_pred.append(actual + np.random.normal(0, 3500))
    
    prediction_data = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Stacked_Model': stacked_pred,
        'Random_Forest': rf_pred,
        'Gradient_Boosting': gb_pred
    })
    
    return model_performance, feature_importance, correlation_data, prediction_data

# Load data
model_perf, feature_imp, corr_data, pred_data = load_sample_data()

# ===== OVERVIEW PAGE =====
if "🏠 Overview" in view:
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Future Prediction</h3>
            <h2>$67,430</h2>
            <p style="color: #2ecc71;">↗️ +12.4%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧠 Model Accuracy</h3>
            <h2>{model_perf['Stacked Model']['R²']*100:.1f}%</h2>
            <p style="color: #2ecc71;">↗️ Best R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 VIX Volatility</h3>
            <h2>23.45</h2>
            <p style="color: #e74c3c;">↘️ -8.7%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💵 Money Supply M2</h3>
            <h2>$21.2T</h2>
            <p style="color: #2ecc71;">↗️ +2.3%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction Chart
    st.subheader("📊 Model Predictions vs Actual Prices")
    
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=pred_data['Date'], 
        y=pred_data['Actual'],
        mode='lines',
        name='Actual Price',
        line=dict(color='white', width=3, dash='dash')
    ))
    
    # Add model predictions
    fig.add_trace(go.Scatter(
        x=pred_data['Date'], 
        y=pred_data['Stacked_Model'],
        mode='lines',
        name='Stacked Model',
        line=dict(color='#00bcd4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_data['Date'], 
        y=pred_data['Random_Forest'],
        mode='lines',
        name='Random Forest',
        line=dict(color='#4caf50', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_data['Date'], 
        y=pred_data['Gradient_Boosting'],
        mode='lines',
        name='Gradient Boosting',
        line=dict(color='#ff9800', width=1)
    ))
    
    fig.update_layout(
        title="Bitcoin Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Macro Ratios
    st.subheader("🌍 Engineered Macro Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f39c12, #e67e22); padding: 1rem; border-radius: 10px;">
            <h4>🥇 Copper/Gold Ratio</h4>
            <h2>2.34</h2>
            <p>+5.2% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); padding: 1rem; border-radius: 10px;">
            <h4>💰 Money/Dollar Ratio</h4>
            <h2>185.7K</h2>
            <p>-1.8% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00bcd4, #0097a7); padding: 1rem; border-radius: 10px;">
            <h4>🌐 Global Velocity</h4>
            <h2>1.14</h2>
            <p>+2.1% this month</p>
        </div>
        """, unsafe_allow_html=True)

# ===== MODELS PAGE =====
elif "🤖 Models" in view:
    st.subheader("🤖 Machine Learning Models Performance")
    
    # Model comparison
    col1, col2, col3 = st.columns(3)
    
    models = ['Stacked Model', 'Random Forest', 'Gradient Boosting']
    colors = ['#00bcd4', '#4caf50', '#ff9800']
    
    for i, (col, model) in enumerate(zip([col1, col2, col3], models)):
        with col:
            perf = model_perf[model]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {colors[i]}, {colors[i]}90); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3>{model}</h3>
                <div style="margin: 1rem 0;">
                    <h4>RMSE</h4>
                    <h2>${perf['RMSE']:,.0f}</h2>
                </div>
                <div>
                    <h4>R² Score</h4>
                    <h2>{perf['R²']*100:.2f}%</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance comparison chart
    st.subheader("📊 Model Performance Comparison")
    
    perf_df = pd.DataFrame({
        'Model': list(model_perf.keys()),
        'R² Score': [v['R²'] * 100 for v in model_perf.values()],
        'RMSE': [v['RMSE'] for v in model_perf.values()]
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=perf_df['Model'], y=perf_df['R² Score'], 
               name='R² Score (%)', marker_color='#00bcd4'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=perf_df['Model'], y=perf_df['RMSE'], 
               name='RMSE', marker_color='#ff5722', opacity=0.7),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Models")
    fig.update_yaxes(title_text="R² Score (%)", secondary_y=False)
    fig.update_yaxes(title_text="RMSE", secondary_y=True)
    
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ===== FEATURES PAGE =====
elif "📈 Features" in view:
    st.subheader("📈 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top Features")
        
        # Feature importance cards
        for idx, row in feature_imp.iterrows():
            color_map = {
                'Crypto': '#ff9800', 'Stocks': '#2196f3', 'Macro': '#9c27b0',
                'Commodities': '#ffeb3b', 'Indices': '#4caf50'
            }
            color = color_map.get(row['Category'], '#607d8b')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20, {color}40); 
                        padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                        border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">{row['Feature']}</h4>
                        <small style="color: {color};">{row['Category']}</small>
                    </div>
                    <h3 style="margin: 0; color: {color};">{row['Importance']*100:.1f}%</h3>
                </div>
                <div style="background: #333; border-radius: 10px; height: 8px; margin-top: 10px;">
                    <div style="background: {color}; height: 8px; border-radius: 10px; 
                                width: {row['Importance']*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Importance Distribution")
        
        # Feature importance chart
        fig = px.bar(
            feature_imp, 
            x='Importance', 
            y='Feature',
            color='Category',
            orientation='h',
            title="Feature Importance by Category"
        )
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Category breakdown
    st.subheader("🏷️ Feature Categories Breakdown")
    category_stats = feature_imp.groupby('Category')['Importance'].agg(['sum', 'count', 'mean'])
    
    fig = px.pie(
        values=category_stats['sum'], 
        names=category_stats.index,
        title="Total Importance by Category"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ===== CORRELATIONS PAGE =====
elif "🔗 Correlations" in view:
    st.subheader("🔗 Asset Correlations with Bitcoin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Correlation Summary")
        
        for idx, row in corr_data.iterrows():
            corr_val = row['Correlation']
            if corr_val > 0:
                bg_color = f"linear-gradient(135deg, #4caf50, #2e7d32)"
                icon = "📈"
            else:
                bg_color = f"linear-gradient(135deg, #f44336, #c62828)"
                icon = "📉"
            
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 1rem; margin: 0.5rem 0; 
                        border-radius: 10px; display: flex; justify-content: space-between; 
                        align-items: center;">
                <div>
                    <h4 style="margin: 0; color: white;">{icon} {row['Asset']}</h4>
                    <small style="color: #ffffffcc;">{row['Type']}</small>
                </div>
                <h3 style="margin: 0; color: white;">{corr_val:.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Correlation Chart")
        
        # Correlation bar chart
        fig = px.bar(
            corr_data,
            x='Asset',
            y='Correlation',
            color='Correlation',
            color_continuous_scale='RdYlGn',
            title="Bitcoin Correlation Coefficients"
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis={'tickangle': -45},
            height=400
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🌡️ Correlation Heatmap")
    
    # Create sample correlation matrix
    assets = corr_data['Asset'].tolist() + ['Bitcoin']
    corr_matrix = np.random.rand(len(assets), len(assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1)  # Diagonal should be 1
    
    # Insert Bitcoin correlations
    bitcoin_idx = len(assets) - 1
    for i, corr_val in enumerate(corr_data['Correlation']):
        corr_matrix[i, bitcoin_idx] = corr_val
        corr_matrix[bitcoin_idx, i] = corr_val
    
    fig = px.imshow(
        corr_matrix,
        x=assets,
        y=assets,
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ===== RAW DATA PAGE =====
elif "📋 Raw Data" in view:
    st.subheader("📋 Raw Data & Export")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "📊 Features", "🔗 Correlations", "⚙️ Model Performance"])
    
    with tab1:
        st.subheader("Prediction Data")
        st.dataframe(pred_data, use_container_width=True)
        
        csv = pred_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="bitcoin_predictions.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Feature Importance")
        st.dataframe(feature_imp, use_container_width=True)
        
        csv = feature_imp.to_csv(index=False)
        st.download_button(
            label="📥 Download Features CSV",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Correlation Data")
        st.dataframe(corr_data, use_container_width=True)
        
        csv = corr_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Correlations CSV",
            data=csv,
            file_name="correlations.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("Model Performance")
        perf_df = pd.DataFrame(model_perf).T
        st.dataframe(perf_df, use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <h4>🚀 Bitcoin Predictor Dashboard</h4>
    <p>Built with Streamlit | Powered by Machine Learning & Macro Analysis</p>
    <p>Made with ❤️ using your Jupyter Notebook data</p>
</div>
""", unsafe_allow_html=True)

# ===== SIDEBAR INFO =====
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")
st.sidebar.success("✅ Models Loaded")
st.sidebar.info("ℹ️ Data Updated: Aug 24, 2025")
st.sidebar.warning("⚠️ Demo Mode - Using Sample Data")

st.sidebar.markdown("### 🔧 Model Info")
st.sidebar.markdown("""
**Features Used:**
- 📈 Crypto Returns
- 🏢 Stock Returns  
- 🌍 Macro Indicators
- 🥇 Commodities
- 📊 Engineered Ratios

**Best Model:** Stacked Ensemble
- Random Forest + Gradient Boosting
- Meta-learner: Ridge Regression
""")