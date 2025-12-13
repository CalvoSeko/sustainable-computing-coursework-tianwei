import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def main():
    # Load data
    # Assumes the script is in 'code/' and data is in 'data/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '../data/df_fuel_ckan.csv')
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Preprocessing
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df_filtered = df[df['DATETIME'].dt.year < 2025].copy()
    df_filtered.set_index('DATETIME', inplace=True)
    
    # Resample to monthly mean
    print("Resampling data...")
    data_train = df_filtered['CARBON_INTENSITY'].resample('M').mean().dropna()
    
    print("Training SARIMA model...")
    # Fit SARIMA model as per the notebook configuration
    model = SARIMAX(data_train, 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12), 
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    print("Model trained.")
    print(model_fit.summary())

    # Forecast for 2025 (12 steps)
    print("Generating forecast for 2025...")
    forecast_result = model_fit.get_forecast(steps=12)
    predicted_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    print("\nForecasted Mean for 2025:")
    print(predicted_mean)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot historical data
    plt.plot(data_train.index, data_train, label='Historical Data')
    
    # Plot forecast
    plt.plot(predicted_mean.index, predicted_mean, label='2025 Forecast', color='red')
    
    # Plot confidence interval
    plt.fill_between(conf_int.index, 
                     conf_int.iloc[:, 0], 
                     conf_int.iloc[:, 1], 
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    
    plt.title('Carbon Intensity Forecast for 2025')
    plt.xlabel('Date')
    plt.ylabel('Carbon Intensity')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_plot_path = os.path.join(script_dir, '../forecast_2025_plot.png')
    plt.savefig(output_plot_path)
    print(f"\nPlot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
