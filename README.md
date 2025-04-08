# smartgrid-energy-forecasting
Predicting energy consumption patterns using time series forecasting models for smart grids
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Generate Sample Smart Meter Data
def generate_sample_data(start_date="2023-01-01", days=30):
    rng = pd.date_range(start=start_date, periods=24 * days, freq='H')
    np.random.seed(0)
    base = 1.5 + 0.7 * np.sin(2 * np.pi * rng.hour / 24)
    noise = np.random.normal(0, 0.3, len(rng))
    values = base + noise
    df = pd.DataFrame({"timestamp": rng, "consumption_kwh": values})
    return df

# Step 2: Store in SQLite
def save_to_sqlite(df, db_name="smartgrid.db"):
    conn = sqlite3.connect(db_name)
    df.to_sql("energy_data", conn, if_exists="replace", index=False)
    conn.close()

# Step 3: Query data using SQL
def load_from_sqlite(db_name="smartgrid.db"):
    conn = sqlite3.connect(db_name)
    query = "SELECT * FROM energy_data ORDER BY timestamp"
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    conn.close()
    return df

# Step 4: Prepare data for LSTM
def create_lstm_dataset(series, look_back=24):
    dataX, dataY = [], []
    for i in range(len(series) - look_back):
        dataX.append(series[i:i + look_back])
        dataY.append(series[i + look_back])
    return np.array(dataX), np.array(dataY)

# Step 5: Build and train LSTM model
def train_lstm(X, y, look_back):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model

# Step 6: Main function
if __name__ == "__main__":
    df = generate_sample_data()
    save_to_sqlite(df)
    
    df = load_from_sqlite()
    values = df["consumption_kwh"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    look_back = 24
    X, y = create_lstm_dataset(scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = train_lstm(X, y, look_back)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title("Smart Grid Energy Forecasting (Using SQL + LSTM)")
    plt.xlabel("Hours")
    plt.ylabel("Energy Consumption (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
