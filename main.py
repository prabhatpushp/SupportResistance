import pandas as pd
import numpy as np
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt


class SupportResistance:
    def __init__(self, df):
        self.df = df.copy()
        self.levels = []
        self.level_types = []
        self.mean = np.mean(self.df['High'] - self.df['Low'])

    def is_support_level(self, i):
        support = self.df['Low'][i] < self.df['Low'][i - 1] and self.df['Low'][i] < self.df['Low'][i + 1] and self.df['Low'][i + 1] < self.df['Low'][i + 2] and self.df['Low'][i - 1] < self.df['Low'][i - 2]
        return support

    def is_resistance_level(self, i):
        resistance = self.df['High'][i] > self.df['High'][i - 1] and self.df['High'][i] > self.df['High'][i + 1] and self.df['High'][i + 1] > self.df['High'][i + 2] and self.df['High'][i - 1] > self.df['High'][i - 2]
        return resistance

    def distance_from_mean(self, level):
        return np.sum([abs(level - y[1]) < self.mean for y in self.levels]) == 0

    def identify_levels(self):
        for i in range(2, self.df.shape[0] - 2):
            if self.is_support_level(i):
                self.levels.append((i, df['Low'][i].round(2)))
                self.level_types.append('Support')
            elif self.is_resistance_level(i):
                self.levels.append((i, df['High'][i].round(2)))
                self.level_types.append('Resistance')

    def plot_levels(self):
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, df.values, width=1, colorup='green', colordown='red', alpha=0.0)
        date_format = mpl_dates.DateFormatter('%y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        fig.tight_layout()

        for level, level_type in zip(self.levels, self.level_types):
            plt.hlines(level[1],
                       xmin=df['Date'][level[0]],
                       xmax=max(df['Date']),
                       colors='blue')
            plt.text(df['Date'][level[0]], level[1], (str(level_type) + ': ' + str(level[1]) + ' '), ha='right',
                     va='center', fontweight='bold', fontsize='x-small')
            plt.title('Support and Resistance levels', fontsize=24, fontweight='bold')
            plt.show(block=True)

    def analyze(self):
        self.identify_levels()
        self.plot_levels()



# Assuming 'data.csv' is your CSV file
df = pd.read_csv('./test_dataset/AXISBANK.csv')

# # Convert the 'Date' column to datetime format
# df['Date'] = pd.to_datetime(df['Date'], unit='s')  # Change 's' to 'm' for minute data
#
# # Convert the 'Date' column to matplotlib date format
# df['Date'] = df['Date'].apply(mpl_dates.date2num)
#
# print(df.head())
#
# # Assuming `df` is your DataFrame
# sr = SupportResistance(df.iloc[:100])
# sr.analyze()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()

df = df.iloc[:200]
# Define support levels
support_levels = [708, 710, 712.5, 715.5, 719.5]

# Create support lines
support_lines = [mpf.make_addplot([support_level] * len(df), color='blue', secondary_y=False) for support_level in support_levels]

# Plot the candlestick chart with colored candles and support lines
mpf.plot(df, type='candle', addplot=[
    mpf.make_addplot(df['SMA_5'], color='orange'),
    mpf.make_addplot(df['SMA_10'], color='blue'),
    *support_lines
], style='yahoo', volume=True)