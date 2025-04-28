from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objs as go
import plotly
import json
import joblib

app = Flask(__name__)

# Load model and scaler
model = load_model('multi_asset_stock_predictor.h5')
scaler = joblib.load('scaler.save')

# Asset information
assets = {
    'TSLA': 'Tesla',
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'GC=F': 'Gold Futures',
    'GOOGL': 'Google',
    'META': 'Meta',
    'NVDA': 'NVIDIA'
}

# Technical indicators calculation
def calculate_technical_indicators(df):
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['Momentum'] = df['Close'].diff(4)
    df['Volatility'] = df['Close'].rolling(window=14).std()
    return df.dropna()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        return predict(ticker)
    return render_template('index.html', assets=assets)

def predict(ticker):
    try:
        # Download data
        data = yf.download(ticker, period="1y")
        if len(data) == 0:
            return jsonify({'error': f"No data found for {ticker}"})
        
        # Preprocess data
        data = calculate_technical_indicators(data)
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_14', 'EMA_14', 'Momentum', 'Volatility']
        
        # Scale data
        scaled_data = scaler.transform(data[features])
        
        # Create sequences
        sequence_length = 60
        X = []
        for i in range(sequence_length, len(scaled_data)):
            seq = scaled_data[i-sequence_length:i]
            
            # Add ticker encoding
            ticker_encoded = np.zeros(len(assets))
            ticker_idx = list(assets.keys()).index(ticker) if ticker in assets else 0
            ticker_encoded[ticker_idx] = 1
            ticker_encoded_reshaped = np.tile(ticker_encoded, (sequence_length, 1))
            
            X.append(np.concatenate([seq, ticker_encoded_reshaped], axis=1))
        
        X = np.array(X)
        
        # Predict
        predictions = model.predict(X)
        
        # Inverse transform
        close_scaler = MinMaxScaler()
        close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
        predicted_prices = close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_prices = data['Close'].values[sequence_length:]
        dates = data.index[sequence_length:]
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=actual_prices, name='Actual Price',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=dates, y=predicted_prices, name='Predicted Price',
                                line=dict(color='red', width=2, dash='dot')))
        
        fig.update_layout(
            title=f'{assets.get(ticker, ticker)} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            showlegend=True
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        mse = np.mean((actual_prices - predicted_prices)**2)
        
        return render_template('index.html', 
                             graphJSON=graphJSON,
                             ticker=ticker,
                             asset_name=assets.get(ticker, ticker),
                             mae=f"{mae:.2f}",
                             mse=f"{mse:.2f}",
                             assets=assets)
    
    except Exception as e:
        return render_template('index.html', 
                             error=str(e),
                             assets=assets)

if __name__ == '__main__':
    app.run(debug=True)