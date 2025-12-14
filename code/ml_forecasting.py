import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # Not explicitly used, but good to keep if needed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Data Loading and Preprocessing ---
# NOTE: Using synthetic data since the file path is not accessible here.
def load_demo_data():
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2024-12-31', freq='D')
    series = np.cumsum(np.random.normal(0, 1, len(dates))) + 100 
    df = pd.DataFrame({'DATETIME': dates, 'CARBON_INTENSITY': series})
    return df

try:
    df = pd.read_csv('../data/df_fuel_ckan.csv')
except FileNotFoundError:
    print("File not found. Using synthetic data for demonstration.")
    df = load_demo_data()

df['DATETIME'] = pd.to_datetime(df['DATETIME'])
df_filtered = df[df['DATETIME'].dt.year < 2025].copy()
df_filtered.set_index('DATETIME', inplace=True)
data_train_test = df_filtered['CARBON_INTENSITY'].resample('ME').mean().dropna()
print(f"Original Data shape: {data_train_test.shape}")


# --- 1. SCALING ---
scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape the data for the scaler: (n_samples, 1)
data_for_scaling = data_train_test.values.reshape(-1, 1)

# Fit and transform the entire dataset
scaled_data = scaler.fit_transform(data_for_scaling)
print("Data successfully scaled.")


# --- 2. SEQUENCE FRAMING FUNCTION ---
def create_lstm_sequences(data, lookback_window=12):
    X, Y = [], []
    # Data is a NumPy array (2D) at this point, so use data[i, 0] for the value
    data_1d = data.flatten() # Ensure we are working with a 1D array for easier indexing
    
    for i in range(len(data_1d) - lookback_window):
        # X is the sequence from time i to i + lookback_window - 1
        X.append(data_1d[i:(i + lookback_window)])
        
        # Y is the single value immediately following the sequence
        Y.append(data_1d[i + lookback_window])
        
    return np.array(X), np.array(Y)


# --- 3. APPLY FRAMING AND SPLIT ---
lookback_window = 12
X_sequenced, Y_targets = create_lstm_sequences(scaled_data, lookback_window=lookback_window)
print(f"Sequenced X shape before reshape: {X_sequenced.shape}")
print(f"Sequenced Y shape before reshape: {Y_targets.shape}")

# Split point (80% train / 20% test)
split_point = int(len(X_sequenced) * 0.8)

# Chronological Split
X_train, X_test = X_sequenced[:split_point], X_sequenced[split_point:]
y_train, y_test = Y_targets[:split_point], Y_targets[split_point:]


# --- 4. RESHAPE FOR LSTM (3D Input) ---
# Required shape: (n_samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Reshape Y targets to (n_samples, 1) for consistency with Keras outputs/inputs
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# --- Final Shape Check ---
print("\n--- Final Data Shapes for LSTM ---")
print(f"X_train shape: {X_train.shape}") # (Samples, 12, 1)
print(f"y_train shape: {y_train.shape}") # (Samples, 1)
print(f"X_test shape: {X_test.shape}")   # (Samples, 12, 1)
print(f"y_test shape: {y_test.shape}")   # (Samples, 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

test_loss = model.evaluate(X_test, y_test, verbose=0) 

print(f"\nTest Loss (e.g., MSE): {test_loss:.4f}")
