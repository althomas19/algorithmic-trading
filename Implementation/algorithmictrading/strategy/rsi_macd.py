import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever


def execute():
    raw_file = stockDataRetriever("wiki/intraday", "none", "none").fetchStock(False)
    stock_groups = raw_file.groupby(raw_file['Ticker'])  # seperate out each stock into groups

    for name, stock in stock_groups:
        rsi_df = RSI(name, stock)
        macd_df = MACD(name, stock)

        # merge rsi and macd strategies
        df = pd.merge(rsi_df, macd_df)

        # scan over 3 periods to find common buy signals
        for i in range(3,len(df)):
            if (scan_buy_signal(df['signal_rsi'][i-3:i])) and (scan_buy_signal(df['signal_macd'][i-3:i])):
                df['positions'][i] = 1
            else:
                df['positions'][i] = 0

        # create sell signals 
        sell = np.where((df['signal_rsi'] == -1) | (df['signal_macd'] == -1), -1, 0)
        df['positions'] += sell

        # #plot trading signals
        # ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
        # df['Close'].plot(ax=ax1, color='r', lw=2.)
        # ax1.set_title(name + ": RSI_MACD")
        # plt.xlabel("Time (Minutes)")
        # ax1.plot(df.loc[df.positions == 1].index, df.Close[df.positions == 1], '^', markersize=10, color='m')
        # ax1.plot(df.loc[df.positions == -1].index, df.Close[df.positions == -1], 'v', markersize=10, color='k')
        # plt.show()
        # for denom in [10,100,1000]:
        for denom in [100]:
            profit(df, name, denom)


def scan_buy_signal(signal):
    for val in signal.tolist():
        if val > 0:
            return True
    return False


def profit(df, name, denom):
    base = 10000
    money = base
    shares = 0

    for i in range(len(df)):
        if (df['positions'][i] < 0): # sell all postions
            money += shares * df['Close'][i]
            shares = 0
        elif (df['positions'][i] > 0):
            money -= denom * df['Close'][i]
            shares += denom

    overall_return = shares * df['Close'][i] + money

    percent_return = (overall_return-base)/base * 100
    #baseline = (df['Close'][i] - df['Close'][0]) / df['Close'][0] * 100
    baseline = (df['Close'][i] - df['Close'][0]) * denom

    print(name, "return:", overall_return-base, "Hold: ", baseline, "percent return:", percent_return)



def RSI(name, stock):
    df = pd.DataFrame(index=stock.index)
    df['Close'] = stock['ClosePrice'].copy()
    df['TimeStamp'] = stock['Timestamp'].copy()
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    window = 14

    # create frames of roll ups and roll downs
    delta = stock['ClosePrice'].diff()
    upgains, downloss = delta.copy(), delta.copy()
    upgains[upgains < 0] = 0
    downloss[downloss > 0] = 0

    # use moving average to create roll up/down
    roll_up = upgains.rolling(window=window, min_periods=1, center=False).mean()
    roll_down = downloss.abs().rolling(window=window, min_periods=1, center=False).mean()

    # create RSI
    RS = roll_up/roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    df['RSI'] = RSI.copy()

    # Create df for buy and sell signal
    df['buy_signal'][window:] = np.where(df['RSI'][window:] < 30 , 1, 0)
    df['sell_signal'][window:] = np.where(df['RSI'][window:] > 70, -1, 0)
    df['signal_rsi'] = df['buy_signal'] + df['sell_signal']

    return df


# MACD = 12 day EMA - 26 day EMA
def MACD(name, stock):
    short_window = 12
    long_window = 26
    df = pd.DataFrame(index=stock.index)
    df['Close'] = stock['ClosePrice'].copy()
    df['TimeStamp'] = stock['Timestamp'].copy()
    df['signal_macd'] = 0.0

    # Create short and long ema
    df['short_ema'] = stock['ClosePrice'].ewm(span=short_window, adjust=False).mean()
    df['long_ema'] = stock['ClosePrice'].ewm(span=long_window, adjust=False).mean()

    # create macd and its signal line
    df['macd'] = df['short_ema'] - df['long_ema']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()  # signal line is 9 day EMA of macd

    # create buy and sell signals
    df['signal_macd'][short_window:] = np.where(df['macd'][short_window:] > df['signal_line'][short_window:], 1.0, 0.0)
    df['positions'] = df['signal_macd'].diff()

    # # SIGNALS ON THE MACD
    # ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    # stock['ClosePrice'].plot(ax=ax1, color='b', lw=2.)
    # df[['macd', 'signal_line']].plot(ax=ax1, lw=2.)
    # ax1.set_title(name + ": MACD")
    # # Plot the buy and sell df
    # ax1.plot(df.loc[df.positions == 1.0].index,
    #          df.signal_line[df.positions == 1.0],
    #          '^', markersize=10, color='m')
    # ax1.plot(df.loc[df.positions == -1.0].index,
    #          df.signal_line[df.positions == -1.0],
    #          'v', markersize=10, color='k')

    # # SIGNALS ON THE PRICE
    # ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    # stock['ClosePrice'].plot(ax=ax1, color='b', lw=2.)
    # ax1.set_title(name + ": MACD")
    # # Plot the buy and sell df
    # ax1.plot(df.loc[df.positions == 1.0].index,
    #          stock.ClosePrice[df.positions == 1.0],
    #          '^', markersize=10, color='g')
    # ax1.plot(df.loc[df.positions == -1.0].index,
    #          stock.ClosePrice[df.positions == -1.0],
    #          'v', markersize=10, color='r')

    return df
