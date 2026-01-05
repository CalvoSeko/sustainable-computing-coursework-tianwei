# sustainable computing coursework tianwei #

## LibreHardwareMonitor-net472 ##
It contains a .NET 4.7.2 version of the LibreHardwareMonitor library, which is used to monitor hardware components such as CPU, GPU, and RAM. With an extra power_monitor.py file, it can be used to monitor the power consumption of the hardware components in 100 ms intervals.

## data ##
It contains the data used for the coursework, including the CI data and the power consumption data.
### calculate_CO2_emission.py ###
Simple function which calculates the CO2 emission based on the power consumption and the CI.
### CI_API.py ###
Simple function which fetches the CI data from the API https://api.carbonintensity.org.uk
### df_fuel_ckan.csv ###
Csv file downloaded from https://www.neso.energy/data-portal/historic-generation-mix
### data_CI.parquet ###
Parquet file converted from df_fuel_ckan.csv using /code/convert_to_parquet.py

## code ##
It contains the code used for the coursework, including the CI data and the power consumption data. There aree csv files with the same name as their related python files as they are used by the power monitoring library functions.
### visualization.py ###
It visualizes the CI data and the power consumption data. Also tries to fit a linear regression and exponential regression model to the data.
### hourly_SARIMA.py, monthly_SARIMA.py ###
It uses SARIMA to generate a forecast for the CI data at an hourly and monthly interval.
### hourly_MLP_forecasting.py, monthly_MLP_forecasting.py ###
It uses MLP to generate a forecast for the CI data at an hourly and monthly interval, using lags and moving averages.
### convert_to_parquet.py ###
It converts the CI data to a parquet file.

## optimization ##
Using multiple systematic optimization methods to reduce functions runtime and hence the power consumption.
### bulk_hourly_MLP.ipynb, bulk_monthly_MLP.ipynb ###
Use multithreading for faster data processing.
### bulk_hourly_SARIMA.ipynb, bulk_monthly_SARIMA.ipynb ###
Use multiprocessing for faster data processing and parameters searching.
### subset_hourly_MLP.ipynb, subset_monthly_MLP.ipynb, subset_hourly_SARIMA.ipynb, subset_monthly_SARIMA.ipynb ###
Use subset of data for faster data processing and parameters searching.
### full_MLP.ipynb, full_SARIMA.ipynb ###
Use all the aforementioned optimization techniques and reading from the parquet file alongside exogenous varibales in SARIMA.