"""
Water Usage Forecasting with SARIMA Model
----------------------------------------
This script analyzes and forecasts water usage data using a SARIMA 
(Seasonal AutoRegressive Integrated Moving Average) model.

Key components:
1. Data preprocessing and visualization
2. SARIMA model parameter selection
3. Model diagnostics and validation
4. Forecasting future water usage

Author: [Your Name]
Date: April 5, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from scipy import stats
import seaborn as sns
import time
from datetime import timedelta

def format_time(seconds):
    """Convert seconds to a human-readable format"""
    return str(timedelta(seconds=round(seconds)))

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the water usage data.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.Series: Preprocessed time series data
    """
    start_time = time.time()
    try:
        df = pd.read_csv(file_path)
        print("\n--- Available columns in DataFrame ---")
        print(df.columns.tolist())
        
        # Data info
        print("\n--- Initial DataFrame Information ---")
        df.info()
        print("\n--- First 5 rows of data ---")
        print(df.head())
        
        # Convert to time series
        df_ts = df[['Last update', 'Usage']].copy()
        df_ts['timestamp'] = pd.to_datetime(df_ts['Last update'])
        df_ts.dropna(subset=['timestamp'], inplace=True)
        df_ts.set_index('timestamp', inplace=True)
        df_ts.sort_index(inplace=True)
        
        execution_time = time.time() - start_time
        print(f"\nData loading and preprocessing completed in: {format_time(execution_time)}")
        
        return df_ts['Usage']
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit()

def analyze_time_series(data):
    """
    Perform detailed time series analysis.
    
    Args:
        data (pd.Series): Time series data
    """
    start_time = time.time()
    
    # Basic statistics
    print("\n--- Time Series Statistics ---")
    print(data.describe())
    
    # Check for stationarity
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(data)
    print("\n--- Augmented Dickey-Fuller Test ---")
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    
    # *** BỔ SUNG CODE VẼ BOX PLOT VÀO ĐÂY ***
    print("\n--- Generating Box Plot for Usage Data ---")
    plt.figure(figsize=(8, 6)) # Tạo một khung hình mới cho biểu đồ
    sns.boxplot(y=data)       # Sử dụng seaborn để vẽ box plot cho dữ liệu 'Usage'
    plt.title('Biểu đồ Box Plot của Mức tiêu thụ nước hàng giờ') # Đặt tiêu đề
    plt.ylabel('Mức tiêu thụ (Usage)') # Đặt nhãn trục Y
    plt.grid(True, axis='y', linestyle='--', alpha=0.7) # Thêm lưới ngang cho dễ nhìn
    try:
        plt.savefig('usage_boxplot.png') # Lưu biểu đồ thành file ảnh
        print("--- Box Plot đã được lưu vào file: usage_boxplot.png ---")
    except Exception as e:
        print(f"Lỗi khi lưu Box Plot: {e}")
    plt.close() # Đóng hình vẽ để tránh hiển thị nếu chạy script tự động
    # *** KẾT THÚC PHẦN BỔ SUNG ***
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    sm.graphics.tsa.plot_acf(data, lags=48, ax=ax1)
    sm.graphics.tsa.plot_pacf(data, lags=48, ax=ax2)
    ax1.set_title('Autocorrelation Function')
    ax2.set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig('acf_pacf_plot.png')
    plt.close()
    
    execution_time = time.time() - start_time
    print(f"\nTime series analysis completed in: {format_time(execution_time)}")

def train_sarima_model(data, order, seasonal_order):
    """
    Train SARIMA model with specified parameters.
    
    Args:
        data (pd.Series): Time series data
        order (tuple): SARIMA order parameters (p,d,q)
        seasonal_order (tuple): Seasonal order parameters (P,D,Q,m)
        
    Returns:
        SARIMAXResults: Fitted model
    """
    start_time = time.time()
    
    model = sm.tsa.SARIMAX(data,
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    
    results = model.fit(disp=False)
    
    execution_time = time.time() - start_time
    print(f"Model training completed in: {format_time(execution_time)}")
    
    return results

def analyze_model_diagnostics(results):
    """
    Perform detailed model diagnostics.
    
    Args:
        results (SARIMAXResults): Fitted SARIMA model
    """
    start_time = time.time()
    
    # Model summary
    print("\n--- Detailed Model Diagnostics ---")
    print(results.summary())
    
    # Residual analysis
    residuals = results.resid
    print("\n--- Residual Statistics ---")
    print(residuals.describe())
    
    # Ljung-Box test for autocorrelation
    lb_test = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[10, 20, 30])
    print("\n--- Ljung-Box Test Results ---")
    print(lb_test)
    
    # Plot residual diagnostics
    fig = results.plot_diagnostics(figsize=(15, 12))
    plt.tight_layout()
    plt.savefig('detailed_diagnostics_plot.png')
    plt.close()
    
    execution_time = time.time() - start_time
    print(f"\nModel diagnostics completed in: {format_time(execution_time)}")

def make_forecast(results, data, steps=48*7):  # Extended to 7 days
    """
    Generate and visualize forecasts.
    
    Args:
        results (SARIMAXResults): Fitted SARIMA model
        data (pd.Series): Original time series data
        steps (int): Number of steps to forecast
        
    Returns:
        pd.Series: Forecast values
    """
    start_time = time.time()
    
    # Generate forecast
    forecast_object = results.get_forecast(steps=steps)
    forecast_values = forecast_object.predicted_mean
    confidence_intervals = forecast_object.conf_int()
    
    # Create forecast index
    try:
        data_freq = pd.infer_freq(data.index)
        if data_freq is None:
            print("Frequency could not be inferred, defaulting to 'H'")
            data_freq = 'H'
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1),
                                     periods=steps,
                                     freq=data_freq)
    except Exception as e:
        print(f"Error creating forecast index: {e}")
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1),
                                     periods=steps,
                                     freq='H')
    
    # Create forecast series
    forecast_series = pd.Series(forecast_values, index=forecast_index, name='Forecast')
    lower_ci = pd.Series(confidence_intervals.iloc[:, 0].values,
                        index=forecast_index,
                        name='Lower CI')
    upper_ci = pd.Series(confidence_intervals.iloc[:, 1].values,
                        index=forecast_index,
                        name='Upper CI')
    
    # Plot forecast
    plt.figure(figsize=(15, 7))
    plt.plot(data.last('14D'), label='Historical Data (14 days)')
    plt.plot(forecast_series, label='SARIMA Forecast', color='red')
    plt.fill_between(forecast_index,
                    lower_ci,
                    upper_ci,
                    color='red',
                    alpha=0.2,
                    label='95% Confidence Interval')
    plt.title('Water Usage Forecast with SARIMA')
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.legend()
    plt.grid(True)
    plt.savefig('extended_forecast_plot.png')
    plt.close()
    
    execution_time = time.time() - start_time
    print(f"\nForecast generation completed in: {format_time(execution_time)}")
    
    return forecast_series

# --- Main Execution ---
if __name__ == "__main__":
    total_start_time = time.time()
    
    # Load and preprocess data
    file_path = 'merged_and_sorted_meter_usages.csv'
    data = load_and_preprocess_data(file_path)
    
    # Analyze time series
    analyze_time_series(data)
    
    # Try different model parameters
    models = [
        ((1, 0, 1), (1, 1, 1, 24)),  # Original model
        ((2, 0, 1), (1, 1, 1, 24)),  # Increased AR order
        ((1, 0, 2), (1, 1, 1, 24)),  # Increased MA order
        ((2, 0, 2), (1, 1, 1, 24))   # Increased both
    ]
    
    results_dict = {}
    for order, seasonal_order in models:
        print(f"\n--- Training SARIMA{order}x{seasonal_order} ---")
        model_results = train_sarima_model(data, order, seasonal_order)
        results_dict[(order, seasonal_order)] = {
            'aic': model_results.aic,
            'bic': model_results.bic,
            'results': model_results
        }
    
    # Find best model
    best_model = min(results_dict.items(), key=lambda x: x[1]['aic'])
    print("\n--- Best Model ---")
    print(f"Parameters: SARIMA{best_model[0][0]}x{best_model[0][1]}")
    print(f"AIC: {best_model[1]['aic']}")
    
    # Use best model for analysis and forecasting
    results = best_model[1]['results']
    
    # Analyze model diagnostics
    analyze_model_diagnostics(results)
    
    # Generate extended forecast (7 days)
    forecast = make_forecast(results, data, steps=24*7)
    
    print("\n--- Extended Forecast (7 days) ---")
    print(forecast)
    
    # Save model results to file
    with open('model_analysis_report.txt', 'w') as f:
        f.write("SARIMA Model Analysis Report\n")
        f.write("=========================\n\n")
        f.write(f"Total execution time: {format_time(time.time() - total_start_time)}\n\n")
        f.write(str(results.summary()))
        f.write("\n\nModel Comparison:\n")
        for params, metrics in results_dict.items():
            f.write(f"\nSARIMA{params[0]}x{params[1]}:\n")
            f.write(f"AIC: {metrics['aic']}\n")
            f.write(f"BIC: {metrics['bic']}\n")
    
    print(f"\nTotal execution time: {format_time(time.time() - total_start_time)}")