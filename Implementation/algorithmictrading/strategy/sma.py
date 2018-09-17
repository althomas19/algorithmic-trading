import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever

# LOOK AT EMA - SHOULD HELP WITH LAGGED PROBLEM
def execute(stock, start_date, end_date):
    stock = stockDataRetriever(stock, start_date, end_date).getStock()

    # Initialize the short and long windows and buy sell in df
    short_window = 40
    long_window = 100
    df = pd.DataFrame(index=stock.index)
    df['signal'] = 0.0

    # Create short and long simple moving average
    df['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    df['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create df
    df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1.0, 0.0) 

    # when signal changes fromn 1 to 0 or 0 to 1 - is a buy or sell
    df['positions'] = df['signal'].diff()

    ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')

    # Plot the closing price, long and short moving average
    stock['Close'].plot(ax=ax1, color='r', lw=2.)
    df[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the buy and sell df
    ax1.plot(df.loc[df.positions == 1.0].index,
             df.short_mavg[df.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(df.loc[df.positions == -1.0].index,
             df.short_mavg[df.positions == -1.0],
             'v', markersize=10, color='k')

    return plt
