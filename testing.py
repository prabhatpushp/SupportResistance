import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import pytz

# Read CSV file into a DataFrame
df = pd.read_csv('./test_dataset/AXISBANK.csv')
df = df.iloc[:1000]

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')
ist = pytz.timezone('Asia/Kolkata')
df['Date'] = df['Date'].dt.tz_localize(pytz.utc).dt.tz_convert(ist)

# Create a DataFrame with the required columns
ohlc_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')


# # Define the indexes where you want to draw the line
# indexes = [0, 10, 15, 19, 50]
#
# # Create a new DataFrame with the 'Close' values at the specified indexes
# line_data_dates = df.index[indexes]
# line_data = df.loc[line_data_dates, 'Close']
#
# date_close_tuples = list(zip(line_data_dates, line_data))

# Define the indexes
indexes = [0, 10, 15, 19, 50]

# Initialize an empty list to store the lists of tuples
dates_values = ohlc_data.index[indexes]
close_values = ohlc_data.loc[dates_values, 'Close']
date_close_lists = list(zip(dates_values,close_values))

# print(date_close_lists)

# Plot using mplfinance
mpf.plot(ohlc_data, type='candle', style='yahoo', volume=True, ylabel='Price', title='Stock Price', xrotation=45,alines=date_close_lists)
