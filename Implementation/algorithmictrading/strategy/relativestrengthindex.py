import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever


def execute(stock_name, start_date, end_date, fetchStocks, share_amount):
    stock = stockDataRetriever(stock_name, start_date, end_date).fetchStock(fetchStocks)

    df = pd.DataFrame(index=stock.index)
    df['Close'] = stock['Close'].copy()
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    window = 14

    # create frames of roll ups and roll downs
    delta = stock['Close'].diff()
    upgains, downloss = delta.copy(), delta.copy()
    upgains[upgains < 0] = 0
    downloss[downloss > 0] = 0

    # use moving average to create roll up/down
    roll_up = upgains.rolling(window=window, min_periods=1, center=False).mean()
    roll_down = downloss.abs().rolling(window=window, min_periods=1, center=False).mean()

    # create RSI
    RS = roll_up/roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    # plot RSI
    fig = plt.figure()
    fig.suptitle(stock_name + ": RSI")
    plt.xlabel("Time (Days)")
    RSI[:601].plot()

    df['RSI'] = RSI.copy()

    # Create df for buy and sell signal
    df['buy_signal'][window:] = np.where(df['RSI'][window:] < 30, 1.0, 0)
    df['sell_signal'][window:] = np.where(df['RSI'][window:] > 70, 0, 1.0)

    # when signal changes from 1 to 0 or 0 to 1 - is a buy or sell
    df['buy_positions'] = df['buy_signal'].diff()
    df['sell_positions'] = df['sell_signal'].diff()

    # merge positions of buying - when RSI crosses 30 and selling - when price crosses 70
    df['buy_positions'] = np.where(df['buy_positions'] < 0, 0, df['buy_positions']) 
    df['sell_positions'] = np.where(df['sell_positions'] > 0, 0, df['sell_positions'])
    df['positions'] = df['buy_positions'] + df['sell_positions']

    df['signal'] = createSignals(df['positions'].copy())

    dfplot = df.copy(deep=True)
    dfplot = dfplot.loc[:600]
    ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    dfplot[['Close']].plot(ax=ax1, color='r', lw=2.)
    ax1.set_title(stock_name[5:] + ": RSI")
    plt.xlabel("Time (Days)")
    ax1.plot(dfplot.loc[dfplot.positions == 1.0].index,
             dfplot.Close[dfplot.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(dfplot.loc[dfplot.positions == -1.0].index,
             dfplot.Close[dfplot.positions == -1.0],
             'v', markersize=10, color='k')

    #share_amount = 100
    initial_capital = float(10000.0)

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
    return percentage_difference_profit, plt


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
