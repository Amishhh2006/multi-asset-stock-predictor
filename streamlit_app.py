import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objects as go
import joblib

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

# Streamlit app
st.title('Multi-Asset Stock Predictor')

# Sidebar controls
ticker = st.sidebar.selectbox('Select Asset', list(assets.keys()), format_func=lambda x: f"{assets[x]} ({x})")
days_to_predict = st.sidebar.slider('Days to Predict', 1, 30, 7)

if st.sidebar.button('Predict'):
    with st.spinner('Fetching data and making predictions...'):
        try:
            # Download data
            data = yf.download(ticker, period="1y")
            if len(data) == 0:
                st.error(f"No data found for {ticker}")
                st.stop()
            
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
                ticker_idx = list(assets.keys()).index(ticker)
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
                title=f'{assets[ticker]} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            mse = np.mean((actual_prices - predicted_prices)**2)
            
            st.subheader('Prediction Accuracy')
            col1, col2 = st.columns(2)
            col1.metric("Mean Absolute Error", f"${mae:.2f}")
            col2.metric("Mean Squared Error", f"${mse:.2f}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
