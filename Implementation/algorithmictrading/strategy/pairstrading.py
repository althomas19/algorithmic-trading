import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
from statsmodels.tsa.stattools import coint


from algorithmictrading.stockData.getStock import stockDataRetriever
from algorithmictrading.stockData.benchmarks import benchmark


def execute(stock_name1, stock_name2, start_date, end_date, fetchStocks):
    stock1 = stockDataRetriever(stock_name1, start_date, end_date).fetchStock(fetchStocks)
    stock2 = stockDataRetriever(stock_name2, start_date, end_date).fetchStock(fetchStocks)

    print("Executing: ", stock_name1, stock_name2)

    # generate df by merging both stocks on date
    df = pd.merge(stock1[['Date','Close']], stock2[['Date','Close']], on="Date")

    # generate zscore from pricing ratios
    ratios = df['Close_x']/df['Close_y']
    ratios_mavg = df['Close_x'].rolling(window=60).mean()/df['Close_y'].rolling(window=60).mean()
    zscore = z_score(ratios)
    df['zscore'] = zscore
    # plot_zscore(zscore)

    # buy stock1 sell stock2 when zscore < -1, buy stock2 sell stock1 when zscore > 1
    df['buy'] = np.where(zscore < -1, 1, 0)
    df['sell'] = np.where(zscore > 1, -1, 0)
    df['signals'] = df['buy'] + df['sell']
    df['positions'] = df['signals'].diff()

    plot_signals(df, stock_name1, stock_name2)

    for denom in [10,100,1000]:
        percent_profit = generate_profit(df,denom)
        print(percent_profit)
    print(benchmark().get_SPY_benchmark())

    return percent_profit, plt


def generate_profit(df, denom):

    money = start_amount = 1000000
    S1_shares = 0
    S2_shares = 0

    for i in range(len(df)):
        if (df.zscore[i] < -1):  # buy Stock1, sell Stock2
            if S2_shares > denom:
                money += denom * df['Close_y'][i]
                S2_shares -= denom
            else:
                money += S2_shares * df['Close_y'][i]
                S2_shares = 0

            S1_shares += denom
            money -= denom * df['Close_x'][i]

        elif (df.zscore[i] > 1):  # buy Stock2, sell Stock1
            if S1_shares > denom:
                money += denom * df['Close_x'][i]
                S1_shares -= denom
            else:
                money += S1_shares * df['Close_x'][i]
                S1_shares = 0

            S2_shares += denom
            money -= denom * df['Close_y'][i]

    overall_return = money + S1_shares * df['Close_x'][i] + S2_shares*df['Close_y'][i]
    print("overall return:",denom,overall_return)
    return (overall_return - start_amount)/ start_amount * 100


def z_score(ratios):
    return (ratios - ratios.mean())/ np.std(ratios)


def plot_zscore(zscore):
    plt.axhline(zscore.mean(), color='gray')
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    ax = zscore.plot()
    ax.set_title("Z_score chart")


def plot_signals(df, stock_name1, stock_name2):
    ax1 = plt.figure().add_subplot(111,  ylabel='Price in $')
    df[['Close_x', 'Close_y']].plot(ax=ax1, lw=2.)
    ax1.set_title(stock_name1[5:] + " and " + stock_name2[5:] + ": Pairs Trading")
    plt.legend([stock_name1[5:],stock_name2[5:]])
    # Plot the buy and sell for both stocks
    ax1.plot(df.loc[df.signals == 1.0].index,
             df.Close_x[df.signals == 1.0],
             '^', markersize=5, color='g')
    ax1.plot(df.loc[df.signals == 1.0].index,
             df.Close_y[df.signals == 1.0],
             'v', markersize=5, color='r')

    ax1.plot(df.loc[df.signals == -1.0].index,
             df.Close_y[df.signals == -1.0],
             '^', markersize=5, color='g')
    ax1.plot(df.loc[df.signals == -1.0].index,
             df.Close_x[df.signals == -1.0],
             'v', markersize=5, color='r')


def cointegration(stock_name1, stock_name2, start_date, end_date, fetchStocks):
    stock1 = stockDataRetriever(stock_name1, start_date, end_date).fetchStock(fetchStocks)
    stock2 = stockDataRetriever(stock_name2, start_date, end_date).fetchStock(fetchStocks)

    score, pvalue, _ = coint(stock1["Close"], stock2["Close"])
    return (stock_name1, stock_name2, pvalue)
