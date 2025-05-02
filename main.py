from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    try:
        # Fetch OHLC data from CoinGecko
        url = 'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1'
        response = requests.get(url)

        # Cek jika request gagal (rate limit, dsb.)
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Gagal mengambil data dari CoinGecko"})

        ohlc_data = response.json()

        # Persiapan DataFrame
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        close_prices = df[['close']].values

        # Validasi data cukup panjang
        sequence_length = 10
        if len(close_prices) <= sequence_length:
            return JSONResponse(status_code=400, content={"error": "Data tidak cukup untuk prediksi"})

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Siapkan data untuk LSTM
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        # Bangun model (saran: load model terlatih untuk production)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)  # dikurangi epoch untuk kecepatan

        # Prediksi harga berikutnya
        last_sequence = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))
        predicted_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        return {"predicted_price": round(predicted_price, 2)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
