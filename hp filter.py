import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1995-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdpjapan = web.DataReader('JPNRGDPEXP', 'fred', start_date, end_date)
gdpfrance = web.DataReader('CLVMNACSCAB1GQFR', 'fred', start_date, end_date)
log_gdpjapan = np.log(gdpjapan)
log_gdpfrance = np.log(gdpfrance)

# calculate the quarterly percent change in real GDP
gdpjapan_pct_change = gdpjapan.pct_change(4)
gdpfrance_pct_change = gdpfrance.pct_change(4)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle, trendjapan = sm.tsa.filters.hpfilter(log_gdpjapan, lamb=1600)
cycle, trendfrance = sm.tsa.filters.hpfilter(log_gdpfrance, lamb=1600)

# Plot the original time series data
plt.plot(log_gdpjapan, label="Original GDP (in log)")
plt.plot(log_gdpfrance, label="Original GDP (in log)")

# Plot the trend component
plt.plot(trendjapan, label="Trend Japan")
plt.plot(trendfrance, label="Trend France")

# Add a legend and show the plot
plt.legend()
plt.show()

trendjapan_std = np. std(trendjapan)
trendfrance_std = np. std(trendfrance)

