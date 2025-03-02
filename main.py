import pandas as pd
import numpy as np
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pytz


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

    def isPivot(self, df, l, n1,n2):
        """
        :param df:
        :param l:
        :param n1:
        :param n2:
        :return:
        """



    def identify_levels(self):
        for i in range(2, self.df.shape[0] - 2):
            if self.is_support_level(i):
                self.levels.append((i, df['Low'][i].round(2)))
                self.level_types.append('Support')
            elif self.is_resistance_level(i):
                self.levels.append((i, df['High'][i].round(2)))
                self.level_types.append('Resistance')
        return self.levels

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

def find_local_minima_maxima(series, prominence_threshold=0.5, distance=6):
    """
    Find local minima and maxima in a series.

    Parameters:
    - series: The input series.
    - prominence_threshold: The minimum prominence of peaks, higher values will result in fewer peaks being detected.
    - distance: Minimum horizontal distance in samples between peaks.

    Returns:
    - local_minima: Indices of local minima.
    - local_maxima: Indices of local maxima.
    """
    # Pad the series to include the start and end
    series = np.pad(series, (1, 1), mode='edge')

    # Find peaks
    peaks, _ = find_peaks(series, prominence=prominence_threshold, distance=distance)

    # Find valleys (negative peaks)
    neg_series = -series
    valleys, _ = find_peaks(neg_series, prominence=prominence_threshold, distance=distance)

    # Remove the padding indices
    local_minima = valleys - 1
    local_maxima = peaks - 1

    return local_minima, local_maxima

def find_local_minima(series, prominence_threshold=0.6, distance=7):
    # Pad the series to include the start and end
    padded_series = np.pad(series, (1, 1), mode='edge')

    # Find valleys (negative peaks)
    neg_series = -padded_series
    valleys, _ = find_peaks(neg_series, prominence=prominence_threshold, distance=distance)

    # Remove the padding indices
    local_minima = valleys - 1
    # Add start and end of the series

    local_minima = np.append(np.argmin(series[:7]), local_minima)  # add start
    local_minima = np.append(local_minima, len(series) - 7 + np.argmin(series[-7:]))  # add end

    return local_minima

def find_local_maxima(series, prominence_threshold=0.6, distance=7):
    # Pad the series to include the start and end
    padded_series = np.pad(series, (1, 1), mode='edge')

    # Find peaks
    peaks, _ = find_peaks(padded_series, prominence=prominence_threshold, distance=distance)

    # Remove the padding indices
    local_maxima = peaks - 1

    # Add start and end of the series
    local_maxima = np.append(np.argmax(series[:7]), local_maxima)  # add start
    local_maxima = np.append(local_maxima, len(series) - 7 + np.argmax(series[-7:]))  # add end

    return local_maxima


def convert_dates_to_alines(indexes, df, column_name):
    dates_values = df.index[indexes]
    close_values = df.loc[dates_values, column_name]
    return list(zip(dates_values, close_values))



# Assuming 'data.csv' is your CSV file
df = pd.read_csv('./test_dataset/AXISBANK.csv')
df = df.iloc[:1950]



df['Date'] = pd.to_datetime(df['Date'], unit='s')
ist = pytz.timezone('Asia/Kolkata')
df['Date'] = df['Date'].dt.tz_localize(pytz.utc).dt.tz_convert(ist)

df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()

# Define support levels
support_resistance = SupportResistance(df)
support_levels = support_resistance.identify_levels()

ohlc_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')

high_maxima = find_local_maxima(ohlc_data['High'].values)
low_minima = find_local_minima(ohlc_data['Low'].values)

low_alines = convert_dates_to_alines(low_minima, ohlc_data, "Low")
high_alines = convert_dates_to_alines(high_maxima, ohlc_data, "High")


support_lines = [mpf.make_addplot([support_level] * len(df), color='blue',type='line', secondary_y=False) for (i,support_level) in support_levels]

# Plot using mplfinance
mpf.plot(ohlc_data, type='candle', style='yahoo', volume=True, ylabel='Price', title='Stock Price', xrotation=45,
         # alines=dict(alines=[low_alines, high_alines],colors=['blue','orange']),
         addplot=[
            # mpf.make_addplot(df['SMA_5'], color='orange',type='line'),
            # mpf.make_addplot(df['SMA_10'], color='blue' ,type='line'),
            *support_lines
         ]
)
