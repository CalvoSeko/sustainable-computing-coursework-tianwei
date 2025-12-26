import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from joblib import Parallel, delayed
import sys, os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
# Add path to power_monitor
sys.path.append(os.path.abspath('LibreHardwareMonitor-net472'))
from power_monitor import PowerMonitor
# Dictionary to store execution times for each code block
execution_times = {}
power_stats = {}
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()

    df = pd.read_parquet('data/data_CI.parquet')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)

    # Resample to hourly means
    df_hourly = df['CARBON_INTENSITY'].resample('H').mean().reset_index()

    df_hourly['CARBON_INTENSITY'] = df_hourly['CARBON_INTENSITY'].interpolate(method='linear')

    # Check data
    print("Hourly Data Head:")
    print(df_hourly.head())
    print("\nShape:", df_hourly.shape)

    execution_times['Data Processing'] = time.time() - start_time
    print(f"Data Processing Time: {execution_times['Data Processing']:.4f} seconds")
    execution_times['Data Processing'] = time.time() - start_time
power_stats["Data Processing"] = pmon.stats()
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()
    def compute_single_feature(data, feat_type, window):
        """Function to be run in a separate thread."""
        if feat_type == 'lag':
            return pd.Series(data['CARBON_INTENSITY'].shift(window), name=f'lag_{window}')
        elif feat_type == 'rolling':
            # Shift(1) to avoid data leakage
            return pd.Series(data['CARBON_INTENSITY'].shift(1).rolling(window=window).mean(), 
                            name=f'rolling_mean_{window}')

    def create_hourly_features_parallel(data, n_jobs=-1):
        df = data.copy()
        
        # 1. Time-based features (These are extremely fast/vectorized, keep on main thread)
        df['hour'] = df['DATETIME'].dt.hour
        df['day_of_week'] = df['DATETIME'].dt.dayofweek
        df['month'] = df['DATETIME'].dt.month
        df['day_of_year'] = df['DATETIME'].dt.dayofyear
        df['year'] = df['DATETIME'].dt.year
        
        # 2. Define tasks for parallel execution
        # Format: (type, window_size)
        tasks = [
            ('lag', 1), ('lag', 24), ('lag', 168),
            ('rolling', 24), ('rolling', 168)
        ]
        
        # 3. Execute tasks in parallel
        # n_jobs=-1 uses all available CPU cores
        feature_columns = Parallel(n_jobs=n_jobs)(
            delayed(compute_single_feature)(df[['CARBON_INTENSITY']], t, w) for t, w in tasks
        )
        
        # 4. Concatenate results back to the main dataframe
        df = pd.concat([df] + feature_columns, axis=1)
        
        return df.dropna()

    df_features = create_hourly_features_parallel(df_hourly, n_jobs=-1)
    print("Features Head:")
    print(df_features.head())

    execution_times['Feature Engineering'] = time.time() - start_time
    print(f"Feature Engineering Time: {execution_times['Feature Engineering']:.4f} seconds")

power_stats["Feature Engineering"] = pmon.stats()
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()

    # Split: Train < 2024, Val = 2024, Test = 2025
    train_data = df_features[df_features['year'] < 2024].copy()
    val_data = df_features[df_features['year'] == 2024].copy()
    test_data = df_features[df_features['year'] == 2025].copy()

    feature_cols = [c for c in df_features.columns if c not in ['DATETIME', 'CARBON_INTENSITY', 'year']]

    X_train = train_data[feature_cols]
    y_train = train_data['CARBON_INTENSITY']

    X_val = val_data[feature_cols]
    y_val = val_data['CARBON_INTENSITY']

    X_test = test_data[feature_cols]
    y_test = test_data['CARBON_INTENSITY']

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    execution_times['Data Splitting'] = time.time() - start_time
    print(f"Data Splitting Time: {execution_times['Data Splitting']:.4f} seconds")
power_stats["Data Splitting"] = pmon.stats()
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale features
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    # We don't scale y_test for prediction, but for internal eval comparison if needed
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    execution_times['Scaling'] = time.time() - start_time
    print(f"Scaling Time: {execution_times['Scaling']:.4f} seconds")
power_stats["Scaling"] = pmon.stats()
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()

    # MLP Regressor Configuration
    # Using more capacity for hourly patterns
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        batch_size = 256,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
    )

    mlp.fit(X_train_scaled, y_train_scaled)
    execution_times['Model Training'] = time.time() - start_time
    print(f"Model Training Time: {execution_times['Model Training']:.4f} seconds")
power_stats["Model Training"] = pmon.stats()
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()

    val_pred_scaled = mlp.predict(X_val_scaled)
    val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()

    test_pred_scaled = mlp.predict(X_test_scaled)
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Validation MAE: {val_mae:.2f}")
    print(f"Test MAE (2025): {test_mae:.2f}")
    print(f"Test MSE (2025): {test_mse:.2f}")

    plt.figure(figsize=(18, 8))

    plt.plot(test_data['DATETIME'], y_test, label='Actual 2025', color='green', alpha=0.4, linewidth=1)

    plt.plot(test_data['DATETIME'], test_pred, label='MLP Forecast 2025', color='red', alpha=0.8, linestyle='--', linewidth=0.8)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.title(f'Full Year 2025: Hourly Carbon Intensity Forecast (MLP)\nMAE: {test_mae:.2f}, MSE: {test_mse:.2f}', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Carbon Intensity (gCO2/kWh)')
    plt.legend(loc='upper right')
    plt.grid(True, which='major', linestyle='-', alpha=0.5)

    plt.show()

    execution_times['Forecasting'] = time.time() - start_time
    print(f"Forecasting Time: {execution_times['Forecasting']:.4f} seconds")
power_stats["Forecasting"] = pmon.stats()
print("EXECUTION TIME SUMMARY")
total_time = 0
for task, duration in execution_times.items():
    print(f"{task}: {duration:.4f} seconds")
    total_time += duration
print(f"Total Execution Time: {total_time:.4f} seconds")
with PowerMonitor(filename = 'full_MLP.csv') as pmon:
    start_time = time.time()
    if 'DATETIME' in df_hourly.columns:
        df_hourly["DATETIME"] = pd.to_datetime(df_hourly["DATETIME"])
        df_hourly = df_hourly.set_index("DATETIME")

    df_monthly = df_hourly.resample("ME").mean()

    def compute_lag(df, l):
        return pd.Series(df['CARBON_INTENSITY'].shift(l), name=f'lag_{l}')

    lags = [1, 2, 3, 6, 12]
    lag_results = Parallel(n_jobs=-1)(delayed(compute_lag)(df_monthly, l) for l in lags)

    # Concatenate and keep DATETIME as a column
    df_features = pd.concat([df_monthly] + lag_results, axis=1).reset_index()

    # Add year column
    df_features['year'] = df_features['DATETIME'].dt.year

    # Drop rows with NaN values (created by lag features)
    df_features = df_features.dropna()

    # Split the data BEFORE resampling to define the sets clearly
    train_data_raw = df_features[df_features['year'] < 2024].copy()
    valid_data_raw = df_features[df_features['year'] == 2024].copy()
    test_data_raw = df_features[df_features['year'] == 2025].copy()

    # FIX: Use 'on="DATETIME"' so resampling works with the column instead of the index
    train_data = train_data_raw.resample("ME", on="DATETIME").mean().reset_index()
    valid_data = valid_data_raw.resample("ME", on="DATETIME").mean().reset_index()
    test_data = test_data_raw.resample("ME", on="DATETIME").mean().reset_index()

    # Prepare Feature Matrix (X) and Target Vector (y)
    X_train = train_data.drop(columns=['DATETIME', 'CARBON_INTENSITY', 'year'], errors='ignore')
    y_train = train_data['CARBON_INTENSITY']

    X_val = valid_data.drop(columns=['DATETIME', 'CARBON_INTENSITY', 'year'], errors='ignore')
    y_val = valid_data['CARBON_INTENSITY']

    X_test = test_data.drop(columns=['DATETIME', 'CARBON_INTENSITY', 'year'], errors='ignore')
    y_test = test_data['CARBON_INTENSITY']

    execution_times["data split monthly"] = time.time() - start_time
power_stats["data split monthly"] = pmon.stats()
# Print power usage summary
print("\nPOWER USAGE SUMMARY")
for task, stats in power_stats.items():
    print(f"{task}: {stats}")

# Calculate and print total energy used by the program
print("\n" + "="*50)
print("TOTAL ENERGY USED (Average Power  Runtime)")
print("="*50)

total_energy_joules = 0
for task, stats in power_stats.items():
    if stats and task in execution_times:
        # Sum up average power from all components (CPU Package, CPU Cores, CPU Memory, CPU Platform, GPU Power)
        total_avg_power = 0
        for component, data in stats.items():
            if isinstance(data, dict) and 'avg' in data:
                total_avg_power += data['avg']
        
        runtime = execution_times[task]
        energy = total_avg_power * runtime  # Energy in Joules = Power (Watts)  Time (seconds)
        total_energy_joules += energy
        print(f"{task}: {total_avg_power:.2f}W × {runtime:.4f}s = {energy:.2f}J")

print(f"\nTotal Energy Used: {total_energy_joules:.2f} Joules")
print(f"Total Energy Used: {total_energy_joules/1000:.4f} kJ (kilojoules)")
print("="*50)