import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever


def execute(stock_name, start_date, end_date, fetchStocks, share_amount):
    stock = stockDataRetriever(stock_name, start_date, end_date).fetchStock(fetchStocks)

    short_window = 50
    long_window = 200
    df = pd.DataFrame(index=stock.index)
    df['signal'] = 0.0

    # Create short and long simple moving average
    df['short_ema'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    df['long_ema'] = stock['Close'].ewm(span=long_window, adjust=False).mean()

    # Create df
    df['signal'][short_window:] = np.where(df['short_ema'][short_window:] > df['long_ema'][short_window:], 1.0, 0.0) 

    # when signal changes fromn 1 to 0 or 0 to 1 - is a buy or sell
    df['positions'] = df['signal'].diff()

    ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    stock['Close'].plot(ax=ax1, color='r', lw=2.)
    df[['short_ema', 'long_ema']].plot(ax=ax1, lw=2.)
    ax1.set_title(stock_name[5:] + ": EMA")
    # Plot the buy and sell df
    ax1.plot(df.loc[df.positions == 1.0].index,
             df.short_ema[df.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(df.loc[df.positions == -1.0].index,
             df.short_ema[df.positions == -1.0],
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
    return plt, percentage_difference_profit
