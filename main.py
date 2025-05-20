import os
import requests
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import joblib
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# ===================== SETUP =====================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================== FETCH DATA =====================

def fetch_from_financeapi(ticker, start_date, end_date, api_key):
    try:
        logging.info(f"Fetching data from FinanceAPI for {ticker}")
        url = f"https://yfapi.net/v8/finance/chart/{ticker}"
        headers = {
            "X-API-KEY": api_key,
            "accept": "application/json"
        }
        params = {
            "interval": "1d",
            "range": "max",
            "period1": int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
            "period2": int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()['chart']['result'][0]
        quotes = data['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['timestamp'], unit='s'),
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        }).dropna()

        logging.info(f"Fetched {len(df)} rows from FinanceAPI.")
        return df
    except Exception as e:
        logging.error(f"FinanceAPI failed: {e}")
        return None

def get_historical_data(ticker, start_date, end_date, api_key):
    df = fetch_from_financeapi(ticker, start_date, end_date, api_key)
    if df is not None and not df.empty:
        return df
    try:
        logging.info("Falling back to Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }).reset_index()
        return df
    except Exception as e:
        logging.error(f"Yahoo Finance fallback failed: {e}")
        raise

# ===================== FEATURE ENGINEERING =====================

def add_technical_indicators(df):
    """Add technical indicators to improve prediction performance"""
    # Moving averages
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    
    # Volatility (Standard deviation of closing prices)
    df['volatility'] = df['close'].rolling(window=14).std()
    
    # Price Momentum (Rate of change)
    df['momentum'] = df['close'].pct_change(periods=5)
    
    # MACD (Moving Average Convergence Divergence)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values that result from calculations
    df.fillna(method='bfill', inplace=True)
    
    return df

# ===================== MODEL =====================

def create_lstm_model(input_shape, neurons=100):
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(neurons // 2),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(scaled_data, window_size, features_count=1):
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, 0])  # We only predict the closing price
    return np.array(X), np.array(y).reshape(-1, 1)

# ===================== VISUALIZATION =====================

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    logging.info(f"Training history plot saved as training_loss.png")
    plt.close()

def plot_prediction_results(dates, actual, predicted, ticker):
    plt.figure(figsize=(16, 8))
    
    # Convert dates to matplotlib format if they're not already
    if isinstance(dates[0], str):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    # Plot actual vs predicted
    plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    
    # Shade the area between the two lines
    plt.fill_between(dates, actual.flatten(), predicted.flatten(), color='gray', alpha=0.2)
    
    # Format the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'{ticker} Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend(fontsize=12)
    
    # Add performance metrics to the plot
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    plt.figtext(0.15, 0.02, f'RMSE: ${rmse:.2f}   R²: {r2:.4f}', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_prediction.png")
    logging.info(f"Prediction results plot saved as {ticker}_prediction.png")
    plt.show()

def plot_recent_with_forecast(df, forecast_days, ticker, predicted_price):
    """Plot recent price movement with forecast point"""
    plt.figure(figsize=(14, 7))
    
    # Get last 90 days for visualization
    recent_df = df.tail(90).copy()
    dates = recent_df['date'].tolist()
    prices = recent_df['close'].tolist()
    
    # Add the forecasted point
    forecast_date = dates[-1] + timedelta(days=1)
    dates.append(forecast_date)
    prices.append(predicted_price)
    
    # Plot
    plt.plot(dates[:-1], prices[:-1], color='blue', label='Historical Price')
    plt.plot(dates[-2:], prices[-2:], color='red', linestyle='--', label='Forecasted Price')
    plt.scatter(dates[-1], prices[-1], color='red', s=100, zorder=5)
    
    # Annotate the prediction point
    plt.annotate(f'${predicted_price:.2f}', 
                 (mdates.date2num(dates[-1]), predicted_price),
                 xytext=(10, 10),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
    
    # Format the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'{ticker} Stock Price - Recent Movement & Next Day Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend(fontsize=12)
    
    # Calculate trend indicators
    price_change = predicted_price - prices[-2]
    pct_change = (price_change / prices[-2]) * 100
    trend = "🔼 UP" if price_change > 0 else "🔽 DOWN"
    
    plt.figtext(0.15, 0.02, 
                f'Forecast: {trend} | Change: ${price_change:.2f} ({pct_change:.2f}%)', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_forecast.png")
    logging.info(f"Recent price with forecast plot saved as {ticker}_forecast.png")
    plt.show()

# ===================== PREDICTION =====================

def predict_next_day(model, df, scaler, window_size, feature_columns):
    # Get the last window_size days of data
    last_window = df[feature_columns].values[-window_size:]
    
    # Scale the data
    last_window_scaled = scaler.transform(last_window)
    
    # Reshape for LSTM [samples, time steps, features]
    last_window_reshaped = last_window_scaled.reshape(1, window_size, len(feature_columns))
    
    # Predict
    next_day_scaled = model.predict(last_window_reshaped)
    
    # Inverse transform to get actual price
    # Create a dummy array with the same shape as the training data
    dummy = np.zeros((1, len(feature_columns)))
    dummy[0, 0] = next_day_scaled[0, 0]  # Set the first column (close price) to our prediction
    
    next_day_price = scaler.inverse_transform(dummy)[0, 0]
    return next_day_price

# ===================== MAIN =====================

def main():
    # Get user input for ticker
    ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
    
    # Configuration
    financeapi_key = os.getenv("FINANCEAPI_KEY", "sCSfpzY6l24yRsGpcZyCh4SvcE56LlV877li5bvQ")
    start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    window_size = 60  # 60 days of historical data for prediction
    
    logging.info(f"====== Starting LSTM pipeline for {ticker} ======")
    
    # Get data
    df = get_historical_data(ticker, start_date, end_date, financeapi_key)
    if df is None or df.empty:
        raise ValueError("No data to work with.")
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Define features for model
    feature_columns = ['close', 'ma7', 'ma21', 'volatility', 'momentum', 'macd', 'rsi']
    
    # Remove rows with NaN values (resulting from technical indicators calculation)
    df = df.dropna()
    
    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    # Create sequences
    X, y = prepare_sequences(scaled_data, window_size, len(feature_columns))
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = create_lstm_model((window_size, len(feature_columns)))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model & scaler
    model_path = f'{MODEL_DIR}/{ticker}_model.h5'
    scaler_path = f'{MODEL_DIR}/{ticker}_scaler.save'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Model & scaler saved to {MODEL_DIR}")
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Create dummy arrays for inverse transformation
    y_test_full = np.zeros((len(y_test), len(feature_columns)))
    y_test_full[:, 0] = y_test.flatten()
    
    y_pred_full = np.zeros((len(y_pred), len(feature_columns)))
    y_pred_full[:, 0] = y_pred.flatten()
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    logging.info(f"Test RMSE: ${rmse:.2f}")
    logging.info(f"Test R²: {r2:.4f}")
    
    # Plot prediction results
    plot_prediction_results(df['date'].iloc[split_idx+window_size:].values, 
                          y_test_inv.reshape(-1, 1), 
                          y_pred_inv.reshape(-1, 1), 
                          ticker)
    
    # Predict next day
    next_day_price = predict_next_day(model, df, scaler, window_size, feature_columns)
    logging.info(f"🔮 Next day predicted price for {ticker}: ${next_day_price:.2f}")
    
    # Get current price for comparison
    current_price = df['close'].iloc[-1]
    price_change = next_day_price - current_price
    percent_change = (price_change / current_price) * 100
    
    # Determine trend
    trend = "INCREASE" if price_change > 0 else "DECREASE"
    
    # Display prediction results
    print("\n" + "="*60)
    print(f"📊 {ticker} STOCK PREDICTION SUMMARY")
    print("="*60)
    print(f"Current Price: ${current_price:.2f} (as of {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"Predicted Next Day Price: ${next_day_price:.2f}")
    print(f"Predicted Change: ${price_change:.2f} ({percent_change:.2f}%)")
    print(f"Prediction indicates: {trend}")
    print(f"Model Accuracy (RMSE): ${rmse:.2f}")
    print(f"Model Fit (R²): {r2:.4f}")
    print("="*60)
    
    # Plot recent days with forecast
    plot_recent_with_forecast(df, 1, ticker, next_day_price)

if __name__ == '__main__':
    main()