# ⚡ AI-Powered Stock Trading Assistant 🚀

Welcome to the future of finance.  
This project is your **personal AI trading sidekick**—built to predict stock prices, analyze trends, and whisper profitable moves straight into your terminal. Powered by deep learning and technical indicators, this tool cuts through the noise and gives you **data-backed trading signals** that just hit different.

---

## 🔥 Features

- 📈 **LSTM-based Deep Learning Model** for time series forecasting
- 🧠 **Technical Indicator Enrichment** (MACD, RSI, MA, Volatility, Momentum)
- 📊 **Actual vs Predicted Visualizations**
- 📉 **Buy/Sell Signal Generator**
- 💾 **Offline Backup using YFinance API**
- ⚙️ **Configurable & Modular Code Structure**

---

## 🧠 How It Works

1. **Fetch stock data** using APIs or fallback to yfinance.
2. **Engineer features** like moving averages, RSI, and MACD.
3. **Normalize + Reshape** data into sequences.
4. **Train LSTM model** on historical data.
5. **Predict future prices** and generate signals.
6. **Visualize trends** and performance with powerful plots.

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow, YFinance
- **Model:** LSTM (Long Short-Term Memory)
- **Prediction Logic:** Trend % + Direction → Suggest Buy / Hold / Sell

---

## 📦 Installation

Clone this repo:

```bash
git clone https://github.com/yourusername/ai-stock-trader.git
cd ai-stock-trader
