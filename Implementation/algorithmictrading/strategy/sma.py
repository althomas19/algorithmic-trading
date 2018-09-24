import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from algorithmictrading.stockData.getStock import stockDataRetriever


def execute( stock_name, start_date, end_date):
    stock = stockDataRetriever(stock_name, start_date, end_date).getStock()

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

    share_amount = 100
    initial_capital = float(100000.0)

    # create a portfolio that simulates owning and buying stocks
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
    percentage_difference_profit = round((float(strategy_profit - baseline_profit)) / initial_capital * 100, 2)

    # plot earnings curve
    fig = plt.figure()
    fig.suptitle(stock_name)
    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
    portfolio['total'].plot(ax=ax1, lw=2.)
    ax1.plot(portfolio.loc[df.positions == 1.0].index, 
             portfolio.total[df.positions == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[df.positions == -1.0].index, 
             portfolio.total[df.positions == -1.0],
             'v', markersize=10, color='k')


    # Show the plot
    return plt, percentage_difference_profit
