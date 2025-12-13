import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np

def perform_iterative_adf_test(data_train):
    """
    Performs the iterative Augmented Dickey-Fuller (ADF) test 
    to find the required non-seasonal differencing order (d) for stationarity.
    
    Returns: 
        tuple: (d, final_stationary_series)
    """
    current_series = data_train.copy()
    d = 0
    max_d = 5 # Safety break

    print("\nStarting Iterative ADF Test...")
    
    while d < max_d:
        print(f"\n--- Differencing Order d={d} ---")
        
        if len(current_series) == 0:
            print("Error: Series became empty after differencing. Stopping.")
            return None, None
            
        result = adfuller(current_series)
        adf_stat = result[0]
        p_value = result[1]
        
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"p-value < 0.05. The time series is stationary at order d={d}.")
            return d, current_series
        else:
            print(f"p-value >= 0.05. The time series is non-stationary. Applying differencing...")
            current_series = current_series.diff().dropna()
            d += 1
    
    if d == max_d:
        print(f"\n⚠️ Reached maximum differencing order {max_d} without achieving stationarity.")
        return None, None

def plot_acf_pacf(series, m=12, title_suffix=""):
    """
    Generates both ACF and PACF plots for parameter identification.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF Plot
    plot_acf(series, lags=3*m, ax=axes[0], title=f'ACF {title_suffix}')
    axes[0].set_title(f'ACF (Correlations) {title_suffix}')
    
    # PACF Plot
    plot_pacf(series, lags=3*m, ax=axes[1], title=f'PACF {title_suffix}')
    axes[1].set_title(f'PACF (Partial Correlations) {title_suffix}')
    
    # Highlight seasonal lags
    for ax in axes:
        ax.axvline(x=m, color='red', linestyle='--', label=f'Lag m={m}')
        ax.axvline(x=2*m, color='red', linestyle='--')
        ax.axvline(x=3*m, color='red', linestyle='--')
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.show()

def apply_seasonal_differencing(series, m):
    """
    Applies first-order seasonal differencing (D=1) to the series.
    """
    return series.diff(m).dropna()


# --- Execution Block ---
if __name__ == "__main__":
    # Preprocessing
    df = pd.read_csv('data/df_fuel_ckan.csv')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df_filtered = df[df['DATETIME'].dt.year < 2025].copy()
    df_filtered.set_index('DATETIME', inplace=True)
    monthly_intensity = df_filtered['CARBON_INTENSITY'].resample('M').mean().dropna()
    m_period = 6

    # --- 1. Find non-seasonal differencing order (d) ---
    d, series_d_stationary = perform_iterative_adf_test(monthly_intensity)
    print(f"\nDetermined non-seasonal differencing order (d): {d}")

    # --- 2. Initial Seasonal Check (d-differenced series) ---
    if series_d_stationary is not None:
        print("\n" + "="*50)
        print(f"Step 2: Checking Seasonal Patterns (ACF/PACF for d={d})")
        print("="*50)
        
        plot_acf_pacf(series_d_stationary, m=m_period, 
                      title_suffix=f'Series after d={d} differencing')

        # --- User Decision Point for D ---
        print("\nINTERPRETATION: Examine the seasonal lags (red lines) on the ACF plot.")
        print("If you see a slowly decaying sin wave or strong spikes, seasonal differencing (D=1) is needed.")
        
        apply_d1 = input("Do you think Seasonal Differencing (D=1) is required? (yes/no): ").strip().lower()

        final_series = series_d_stationary
        D = 0
        
        # --- 3. Conditional Seasonal Differencing (D=1) ---
        if apply_d1 == 'yes':
            final_series = apply_seasonal_differencing(series_d_stationary, m_period)
            D = 1
            print(f"\nApplied Seasonal Differencing D={D}.")
            
            # --- Re-run Plots on D-differenced series ---
            print("\n" + "="*50)
            print(f"Step 3: Re-checking Patterns (ACF/PACF for d={d}, D={D})")
            print("="*50)
            plot_acf_pacf(final_series, m=m_period, 
                          title_suffix=f'Final Series after d={d} and D={D} differencing')
            
            print("\nINTERPRETATION: These plots are now of the fully stationary series. Use the cut-offs to determine p, q, P, and Q.")
        
        else:
            print("\nNo Seasonal Differencing applied (D=0). The current plots show the fully stationary series.")
        
        # --- Final Parameter Summary ---
        print("\n" + "="*50)
        print("SARIMA Parameter Identification Summary:")
        print(f"Non-seasonal Differencing (d): {d}")
        print(f"Seasonal Differencing (D): {D}")
        print(f"Seasonal Period (m): {m_period}")
        print("\nNext: Use the final ACF/PACF plots to determine p, q, P, and Q.")
        print("p (AR) -> PACF cutoff (non-seasonal lags)")
        print("q (MA) -> ACF cutoff (non-seasonal lags)")
        print("P (SAR) -> PACF cutoff (seasonal lags)")
        print("Q (SMA) -> ACF cutoff (seasonal lags)")
    else:
        print("Cannot proceed to seasonal checks as non-seasonal stationarity was not achieved.")