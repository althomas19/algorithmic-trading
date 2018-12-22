import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever


def execute(stock_name, start_date, end_date, fetchStocks, share_amount):
    stock = stockDataRetriever(stock_name, start_date, end_date).fetchStock(fetchStocks)

    window = 12
    df = pd.DataFrame(index=stock.index)
    df['low_signal'] = 0.0
    df['high_signal'] = 0.0

    # Create bands
    df['middle_band'] = stock['Close'].rolling(window=window, min_periods=1, center=False).mean()
    df['moving_deviation'] = stock['Close'].rolling(window=window, min_periods=1, center=False).std()
    df['upper_band'] = df['middle_band'] + df['moving_deviation'] * 2
    df['lower_band'] = df['middle_band'] - df['moving_deviation'] * 2
    df['closing_price'] = stock['Close']

    # Create df for high and low bands signal
    df['low_signal'][window:] = np.where(df['closing_price'][window:] < df['lower_band'][window:], 1.0, 0)
    df['high_signal'][window:] = np.where(df['closing_price'][window:] > df['upper_band'][window:], 0, 1.0)

    # when signal changes from 1 to 0 or 0 to 1 - is a buy or sell
    df['buy_positions'] = df['low_signal'].diff()
    df['sell_positions'] = df['high_signal'].diff()

    # merge positions of buying - when price crosses lower band and selling - when price crosses upper band
    df['buy_positions'] = np.where(df['buy_positions'] < 0, 0, df['buy_positions']) 
    df['sell_positions'] = np.where(df['sell_positions'] > 0, 0, df['sell_positions'])
    df['positions'] = df['buy_positions'] + df['sell_positions']

    # combine low,high signals to create stock ownership signal
    df['signal'] = createSignals(df['positions'].copy())

    ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    df[['upper_band', 'lower_band', 'middle_band', 'closing_price']].plot(ax=ax1, lw=2.)
    ax1.set_title(stock_name[5:] + ": Bollinger Bands")
    ax1.plot(df.loc[df.positions == 1.0].index,
             df.lower_band[df.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(df.loc[df.positions == -1.0].index,
             df.upper_band[df.positions == -1.0],
             'v', markersize=10, color='k')

    #share_amount = 100
    initial_capital = float(100000.0)

    positions = pd.DataFrame(index=df.index).fillna(0.0)
    positions[stock_name] = share_amount*df['signal']
    portfolio = positions.multiply(stock['Adj. Close'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(stock['Adj. Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(stock['Adj. Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    # compare against baseline long position
    baseline_profit = (stock['Adj. Close'].iloc[-1] - stock['Adj. Close'].iloc[0])*share_amount
    strategy_profit = portfolio['total'].iloc[-1] - initial_capital
    print(baseline_profit, strategy_profit)
    if strategy_profit < baseline_profit:
        percentage_difference_profit = round((float(strategy_profit - abs(baseline_profit))) / initial_capital * 100, 2)
    else:
        percentage_difference_profit = round((float(strategy_profit - baseline_profit)) / initial_capital * 100, 2)
    return plt, percentage_difference_profit


# function that creates stock ownership signals from positions
def createSignals(positions):
    signals = positions
    for x in range(1, len(signals)):
        if (signals[x] == 0):
            if (signals[x-1] == 1):
                signals[x] = 1
        elif (signals[x] == -1):
            signals[x] = 0
    return signals
