from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bitcoin Price Predictor API is running"}

@app.get("/predict")
def predict_price():
    # Fetch OHLC data from CoinGecko
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1'
    response = requests.get(url)
    ohlc_data = response.json()

    # Prepare DataFrame
    df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    close_prices = df[['close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # LSTM input
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model (note: retraining every time — better to load pre-trained model in production)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Predict next price
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    return {
        "predicted_price": round(predicted_price, 2)
    }
