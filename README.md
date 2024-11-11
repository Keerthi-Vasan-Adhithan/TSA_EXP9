# EX.NO.09        A project on Time Series Analysis on Apple Stock Data Using the ARIMA Model 

### Date: 
### Developed by: KEERTHI VASAN A
### Register Number: 212222240048

### AIM:
To create a project on time series analysis of Apple Stock data using the ARIMA model in Python and compare it with other models.

### ALGORITHM:
1. Explore the Dataset of Apple stock Data 
   - Load the Apple Stock dataset and perform initial exploration, focusing on the `year` and `value` columns. Plot the time series to visualize trends.

2. Check for Stationarity of the Time Series 
   - Plot the time series data and use the following methods to assess stationarity:
     - Time Series Plot: Visualize the data for seasonality or trends.
     - ACF and PACF Plots**: Inspect autocorrelation and partial autocorrelation plots to understand the lag structure.
     - ADF Test**: Apply the Augmented Dickey-Fuller (ADF) test to check if the series is stationary.

3. Transform to Stationary (if needed)  
   - If the series is not stationary (as indicated by the ADF test), apply differencing to remove trends and make the series stationary.

4. Determine ARIMA Model Parameters (p, d, q) 
   - Use insights from the ACF and PACF plots to select the AR and MA terms (`p` and `q` values).
   - Choose `d` based on the differencing applied to achieve stationarity.

5. Fit the ARIMA Model 
   - Fit an ARIMA model with the selected `(p, d, q)` parameters on the historical Apple Stock data values.

6. Make Time Series Predictions  
   - Forecast future values for a specified time period (e.g., 12 years) using the fitted ARIMA model.

7. Auto-Fit the ARIMA Model (if applicable) 
   - Use auto-fitting methods (such as grid search or auto_arima from `pmdarima`) to automatically determine the best parameters for the model if needed.

8. Evaluate Model Predictions  
   - Compare the predicted values with actual values using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to assess the model's accuracy.
   - Plot the historical data and forecasted values to visualize the model's performance.

### PROGRAM:
```PY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# Load the dataset from the uploaded file
data = pd.read_csv('/content/apple_stock.csv')  # Replace 'apple_stock.csv' with your uploaded file

# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract the 'Close' column for analysis (or another relevant column)
series = data['Close'].dropna()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(series)
plt.title("Stock Closing Prices Over Time")
plt.xlabel("Year")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# Augmented Dickey-Fuller Test for stationarity
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # Returns True if series is stationary

# Check if series is stationary, otherwise difference it
is_stationary = adf_test(series)
if not is_stationary:
    series_diff = series.diff().dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(series_diff)
    plt.title("Differenced Data")
    plt.grid(True)
    plt.show()
else:
    series_diff = series

# Plot ACF and PACF to determine p and q
plot_acf(series_diff, lags=20)
plt.title("ACF of Differenced Series")
plt.show()

plot_pacf(series_diff, lags=20)
plt.title("PACF of Differenced Series")
plt.show()

# Set ARIMA parameters based on insights (adjust as necessary)
p, d, q = 1, 1, 1  # Example values; update based on ACF/PACF

# Fit the ARIMA model
model = ARIMA(series, order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast future values
forecast_steps = 12  # Forecasting 12 periods (e.g., months if monthly data)
forecast = fitted_model.forecast(steps=forecast_steps)

# Set up the forecast index
last_date = series.index[-1]
forecast_index = pd.date_range(last_date, periods=forecast_steps + 1, freq='M')[1:]  # Adjust frequency if needed

# Plot forecast vs historical data
plt.figure(figsize=(12, 6))
plt.plot(series, label="Historical Data")
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("Stock Data Forecast")
plt.xlabel("Year")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

```

### OUTPUT:
Stock Prices over time:

![image](https://github.com/user-attachments/assets/b8d62320-3d65-4017-8ed6-7e750d2ebf82)


Autocorrelation:

![image](https://github.com/user-attachments/assets/3d13c368-8b8f-4c0c-9851-aafaf477a948)


Partial Autocorrelation:

![image](https://github.com/user-attachments/assets/9257b176-d601-4554-ba80-c193fd853e1b)


Model summary:

![image](https://github.com/user-attachments/assets/98fbce85-7a72-4dd5-8dd5-614ee73349e0)


Apple Stock Forecast:

![image](https://github.com/user-attachments/assets/caf29c7d-9fae-4cd9-90b7-4a835c975055)





### RESULT:
Thus the project on Time series analysis on Ev Sales based on the ARIMA model using python is executed successfully.
