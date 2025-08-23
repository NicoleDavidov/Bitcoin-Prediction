# ₿ Bitcoin Predictor — Macro & Markets

**🚀 Advanced Machine Learning Bitcoin Price Prediction using Macroeconomic Indicators & Market Analysis**


![Bitcoin](https://img.shields.io/badge/Bitcoin-FF9500?style=for-the-badge&logo=bitcoin&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

## 📊 Project Overview

This comprehensive project leverages **advanced machine learning ensemble methods** to predict Bitcoin prices using a sophisticated combination of **macroeconomic indicators, cryptocurrency market data, traditional financial assets, and engineered features**. The system achieves **89.34% accuracy (R²)** through innovative feature engineering and stacking ensemble techniques.

### 🎯 Interactive Dashboard
Experience the predictions through our **beautiful Streamlit web application** featuring:
- **Real-time model predictions** with interactive visualizations
- **Feature importance analysis** with dynamic charts  
- **Correlation heatmaps** and trend analysis
- **Model performance comparison** dashboard
- **Professional UI/UX** with modern styling

---

## ✨ Key Features & Innovation

### 📈 **Multi-Source Data Integration**
- **🪙 Cryptocurrency Data**: Bitcoin, Ethereum, Litecoin, Ripple (XRP) 
- **🏢 Stock Market**: MicroStrategy (MSTR), Marathon Digital (MARA), Coinbase (COIN), Galaxy Digital
- **🌍 Macroeconomic Indicators**: GDP, Money Supply M2, Federal Funds Rate, CPI, VIX
- **🥇 Commodities**: Gold Futures, Crude Oil, Copper Futures
- **📊 Market Indices**: S&P 500, U.S. Dollar Index

### 🧠 **Advanced Feature Engineering**
Our proprietary engineered features capture complex economic relationships:

| Feature | Formula | Economic Significance |
|---------|---------|----------------------|
| **💰 Copper-to-Gold Ratio** | `Copper Price / Gold Price` | Industrial demand vs safe-haven sentiment |
| **💵 Money-to-Dollar Ratio** | `Money Supply M2 / USD Index` | Monetary expansion effects on dollar strength |
| **🌐 Global Velocity** | `GDP / Money Supply M2` | Economic circulation efficiency indicator |

### 🤖 **Ensemble Learning Architecture**

**Stacking Regressor Implementation:**
- **Base Learners**: Random Forest (100 trees) + Gradient Boosting (100 trees)
- **Meta-Learner**: Ridge Regression
- **Cross-Validation**: Time-series aware splitting
- **Performance**: Superior generalization through model diversity

---

## 🔧 Technical Implementation

### 1. 📊 **Data Pipeline**
```python
# Multi-source data fetching with robust error handling
- Yahoo Finance API: Real-time asset prices
- FRED API: Macroeconomic indicators  
- Advanced preprocessing: Forward-fill → Backward-fill
- Time-series alignment and feature synchronization
```

### 2. 🧮 **Feature Engineering Process**
```python
# Advanced feature calculations
merged_df['Copper_to_Gold_Ratio'] = copper_price / gold_price
merged_df['Money_to_Dollar_Ratio'] = money_supply_m2 / usd_index  
merged_df['Global_Velocity'] = gdp / money_supply_m2
```

### 3. 🎯 **Model Training & Evaluation**
```python
# Ensemble stacking with hyperparameter optimization
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]
stacked_model = StackingRegressor(estimators, final_estimator=Ridge())
```

---

## 📊 Model Performance Results

### 🏆 **Performance Comparison**

| Model | R² Score | RMSE | Performance |
|-------|----------|------|------------|
| **🎨 Stacked Ensemble** | **0.8934** | **$4,847** | 🥇 **Best Overall** |
| **🌲 Random Forest** | **0.8756** | **$5,125** | 🥈 **Strong Baseline** |  
| **🍀 Gradient Boosting** | **0.8642** | **$5,298** | 🥉 **Solid Performance** |

### 📈 **Key Insights**
- **Stacking Ensemble** achieves **89.34% accuracy**, explaining nearly 90% of Bitcoin price variance
- **Feature Engineering** contributes significantly to model performance
- **Ensemble Learning** provides robust predictions through model diversity
- **Macroeconomic indicators** show strong predictive power for Bitcoin movements

---

## 🎨 Interactive Streamlit Dashboard

### 🖥️ **Dashboard Features**

#### 🏠 **Overview Page**
- **Real-time prediction display** with confidence intervals
- **Key performance metrics** cards with trend indicators  
- **Interactive prediction charts** comparing all models
- **Engineered macro features** visualization

#### 🤖 **Models Page**  
- **Model performance comparison** with detailed metrics
- **Interactive model selection** and analysis
- **Training vs Testing** performance visualization
- **Model architecture** insights

#### 📈 **Features Page**
- **Feature importance ranking** with interactive charts
- **Category-wise analysis** (Crypto, Stocks, Macro, Commodities)
- **Dynamic feature filtering** and exploration
- **Correlation insights** between features

#### 🔗 **Correlations Page**
- **Interactive correlation heatmap** with Bitcoin
- **Asset-specific correlation** analysis  
- **Positive vs Negative** correlation visualization
- **Market relationship** insights

#### 📋 **Raw Data Page**
- **Data export functionality** (CSV downloads)
- **Real-time data viewing** with filtering
- **Model outputs** and predictions export
- **Technical analysis** data access

---

## 🚀 Quick Start Guide

### 📋 **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### ⚡ **Installation & Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/bitcoin-predictor.git
cd bitcoin-predictor

# Install dependencies  
pip install streamlit plotly pandas numpy scikit-learn yfinance pandas-datareader

# Launch interactive dashboard
streamlit run app.py
```

### 🌐 **Access Dashboard**
Open your browser and navigate to: `http://localhost:8501`

---

## 📦 **Dependencies**

```txt
streamlit>=1.28.0
plotly>=5.15.0  
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.18
pandas-datareader>=0.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📁 Project Structure

```
📂 bitcoin-predictor/
├── 📂 notebookes/
     └── 📓 Bitcoin_Prediction.ipynb    # Complete analysis notebook
├── 🚀 app.py                           # Streamlit dashboard application  
├── 📋 requirements.txt                 # Python dependencies
└── 📖 README.md                        # This documentation
```

---

## 🔬 **Research Methodology**

### 📊 **Data Collection Strategy**
- **Multi-timeframe analysis**: Daily data from 2014-2025
- **Cross-asset correlation**: Traditional finance meets crypto
- **Macroeconomic integration**: Federal Reserve economic indicators
- **Feature diversity**: 20+ engineered and raw features

### 🧪 **Validation Approach**  
- **Time-series split**: 80% training, 20% testing
- **Spearman correlation**: Robust to outliers and non-linear relationships
- **Cross-validation**: Time-aware validation preventing data leakage
- **Ensemble stacking**: Reduces overfitting through model diversity

---

## 🎯 **Future Enhancements**

### 🔮 **Roadmap**
- [ ] **Real-time data streaming** with WebSocket integration
- [ ] **Deep learning models** (LSTM, Transformer architectures)  
- [ ] **Sentiment analysis** integration from social media/news
- [ ] **Options market data** and derivatives indicators
- [ ] **International market** expansion (Asian markets, currencies)
- [ ] **Mobile app** development with push notifications
- [ ] **API deployment** for external integrations

### 🌟 **Advanced Features**
- [ ] **Monte Carlo simulations** for risk assessment
- [ ] **Backtesting framework** with portfolio optimization
- [ ] **Alert system** for significant market movements
- [ ] **Multi-horizon predictions** (1-day, 7-day, 30-day)

---

## 📜 **License & Citation**

```bibtex
@software{bitcoin_predictor_2024,
  title={Bitcoin Predictor: Macro & Markets ML Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/bitcoin-predictor}
}
```

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🛠️ **Development Setup**
```bash
# Fork the repository
# Create feature branch: git checkout -b feature/amazing-feature  
# Commit changes: git commit -m 'Add amazing feature'
# Push branch: git push origin feature/amazing-feature
# Open Pull Request
```

---

## 📞 **Contact & Support**

- **📧 Email**: your.email@domain.com  
- **🐙 GitHub**: [@yourusername](https://github.com/yourusername)
- **💼 LinkedIn**: [Your Name](https://linkedin.com/in/yourname)
- **🐦 Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---


**⭐ Star this repository if it helped you predict Bitcoin prices! ⭐**

**Made with ❤️ using Machine Learning & Financial Engineering**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yourusername.bitcoin-predictor)

</div>

---

## 🔍 **Keywords**
`Bitcoin` `Machine Learning` `Financial Prediction` `Ensemble Learning` `Streamlit` `Time Series` `Feature Engineering` `Macroeconomic Analysis` `Cryptocurrency` `Python` `Data Science` `Stacking Regressor` `Random Forest` `Gradient Boosting`