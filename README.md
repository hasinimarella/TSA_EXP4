# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 16-09-2025
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data=pd.read_csv('/content/blood_donor_dataset.csv')
print(data.columns)
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['number_of_donation']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
## OUTPUT:
ORIGINAL DATA:
<img width="1102" height="621" alt="image" src="https://github.com/user-attachments/assets/a12e5103-e340-4ccc-afb1-91ae614b09d9" />

SIMULATED ARMA(1,1) PROCESS:
<img width="1008" height="555" alt="image" src="https://github.com/user-attachments/assets/a7f83d11-8f1a-4485-90a2-299332f59caf" />

Autocorrelation
<img width="1008" height="532" alt="image" src="https://github.com/user-attachments/assets/cf130dd6-5395-4c95-8871-dbdc772eeee8" />

Partial Autocorrelation

<img width="1002" height="530" alt="image" src="https://github.com/user-attachments/assets/b5b2297d-b9c6-42c0-966d-0574a13a97c1" />

SIMULATED ARMA(2,2) PROCESS:
<img width="1076" height="528" alt="image" src="https://github.com/user-attachments/assets/ba5b2676-669a-4344-8ee9-e4b60bc523b0" />

Autocorrelation
<img width="1012" height="530" alt="Screenshot 2025-09-20 084935" src="https://github.com/user-attachments/assets/8ac7f102-e34f-4db5-9ba2-2f73f1586582" />


Partial Autocorrelation
<img width="1037" height="536" alt="image" src="https://github.com/user-attachments/assets/8badb414-1dda-4eec-8395-fb8e198835da" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
