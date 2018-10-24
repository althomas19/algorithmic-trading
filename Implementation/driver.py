from algorithmictrading.strategy import sma, ema, bollingerbands, pairstrading

stock_tickers = ["WIKI/AAPL", "WIKI/GOOG", "WIKI/DIS", 'WIKI/FB', 'WIKI/INTC', 'WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/AAL', 'WIKI/PYPL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/VOD', 'WIKI/MNST']
fetch_stocks = False  # if array of stockTickers updated run driver with this set to true, will download csv


def momentum_strategies():
    plots = []

    for stock in stock_tickers:
        plot, percent_performance_sma = sma.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks)
        plots.append(plot)

        plot, percent_performance_ema = ema.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks)
        plots.append(plot)

        plot, percent_performance_bbands = bollingerbands.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks)
        plots.append(plot)

        print(stock, "SMA", percent_performance_sma)
        print(stock, "EMA", percent_performance_ema)
        print(stock, "Bollinger Bands", percent_performance_bbands)

    # for plot in plots:
    #     plot.show()


stock_tickers_pairs_trading = ["WIKI/AAPL", "WIKI/DIS", 'WIKI/INTC', 'WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/AAL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/VOD', 'WIKI/MNST']
pairs = [('WIKI/QCOM', 'WIKI/SBUX'), ('WIKI/INTC', 'WIKI/MSFT'), ('WIKI/AMD', 'WIKI/SBUX'), ('WIKI/AMD', 'WIKI/EBAY'), ('WIKI/AMD', 'WIKI/VOD')]
#pairs = [('WIKI/INTC', 'WIKI/MSFT')]


def pairs_strategy():
    plots = []

    for stock in pairs:
        percent_performance, plot = pairstrading.execute(stock[0], stock[1], "2006-10-01", "2017-01-01", fetch_stocks)
        plots.append(plot)

    # for plot in plots:
    #     plot.show()


def pairs_cointegration():
    cointegration_list = []
    for i in range(len(stock_tickers_pairs_trading)):
        for j in range(i+1,len(stock_tickers_pairs_trading)):
            cointegration_list.append(pairstrading.cointegration(stock_tickers_pairs_trading[i], stock_tickers_pairs_trading[j], "2006-10-01", "2017-01-01", fetch_stocks))

    return sorted(cointegration_list, key=lambda x: x[2])
    ### RESULTS WITH P < .05
    # ('WIKI/QCOM', 'WIKI/SBUX', 0.0078132627461340292)
    # ('WIKI/INTC', 'WIKI/MSFT', 0.015817118985228147)
    # ('WIKI/AMD', 'WIKI/SBUX', 0.025901277434079627)
    # ('WIKI/AMD', 'WIKI/EBAY', 0.027462171694469979)
    # ('WIKI/AMD', 'WIKI/VOD', 0.037477501569076914)



if __name__ == '__main__':
    #momentum_strategies()
    pairs_strategy()
    #pairs_cointegration()
