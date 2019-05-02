from algorithmictrading.strategy import sma, ema, bollingerbands, pairstrading, relativestrengthindex, rsi_macd, twitter_sentiment

stock_tickers = ["WIKI/AAPL", "WIKI/GOOG", "WIKI/DIS", 'WIKI/FB', 'WIKI/INTC', 'WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/AAL', 'WIKI/PYPL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/VOD', 'WIKI/MNST']
#stock_tickers = ["WIKI/GOOG"]
fetch_stocks = False  # if array of stockTickers updated run driver with this set to true, will download csv
stock_tickers_pairs_trading = ["WIKI/AAPL", "WIKI/DIS", 'WIKI/INTC', 'WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/AAL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/VOD', 'WIKI/MNST']
pairs = [('WIKI/QCOM', 'WIKI/SBUX'), ('WIKI/INTC', 'WIKI/MSFT'), ('WIKI/AMD', 'WIKI/SBUX'), ('WIKI/AMD', 'WIKI/EBAY'), ('WIKI/AMD', 'WIKI/VOD')]

def momentum_strategies():
    plots = []
    #denominations = [10, 100, 1000]
    denominations = [100]
    for stock in stock_tickers:
        for denom in denominations:
            plot, percent_performance_sma = sma.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks, denom)
            plots.append(plot)

            plot, percent_performance_ema = ema.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks, denom)
            plots.append(plot)

            plot, percent_performance_bbands = bollingerbands.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks, denom)
            plots.append(plot)

            percent_performance_rsi, plot = relativestrengthindex.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks, denom)
            plots.append(plot)

            print(stock, denom, "SMA", percent_performance_sma)
            print(stock, denom, "EMA", percent_performance_ema)
            print(stock, denom, "Bollinger Bands", percent_performance_bbands)
            print(stock, denom, "RSI", percent_performance_rsi)

    # for plot in plots:
    #     plot.show()


def intraday_strategies():
    rsi_macd.execute()


def pairs_strategy():
    plots = []

    for stock in pairs:
        percent_performance, plot = pairstrading.execute(stock[0], stock[1], "2006-10-01", "2017-01-01", fetch_stocks)
        plots.append(plot)

    for plot in plots:
        plot.show()


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

def twitter_sentiment_strategy():  # THESIS WORK!
    #stocks = ["WIKI/DIS", "WIKI/FB", "WIKI/INTC",' WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/PYPL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/MNST', "WIKI/BIG", 'WIKI/COLM', 'WIKI/GOOGL', 'WIKI/NKE', 'WIKI/VOD']
    stocks = ["WIKI/HAS","WIKI/QCOM", "WIKI/FB",' WIKI/GOOGL', "WIKI/INTC", "WIKI/MSFT", "WIKI/VOD",' WIKI/PYPL']
    for stock in stocks:
        twitter_sentiment.execute(stock, "2006-10-01", "2017-01-01", fetch_stocks)

if __name__ == '__main__':
    #momentum_strategies()
    #pairs_strategy()
    #pairs_cointegration()
    intraday_strategies()
    #twitter_sentiment_strategy()
