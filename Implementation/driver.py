from algorithmictrading.strategy import sma,ema

stockTickers = ["WIKI/AAPL", "WIKI/GOOG", "WIKI/DIS", 'WIKI/FB', 'WIKI/INTC', 'WIKI/MSFT', 'WIKI/AMD', 'WIKI/HAS', 'WIKI/EBAY', 'WIKI/AAL', 'WIKI/PYPL', 'WIKI/QCOM', 'WIKI/SBUX', 'WIKI/VOD', 'WIKI/MNST']
plots = []
fetchStocks = False  # if array of stockTickers updated run driver with this set to true, will download csv

for stock in stockTickers:
    plot, percent_performance_sma = sma.execute(stock, "2006-10-01", "2017-01-01", fetchStocks)
    plots.append(plot)

    plot, percent_performance_ema = ema.execute(stock, "2006-10-01", "2017-01-01", fetchStocks)
    plots.append(plot)

    print(stock, "SMA", percent_performance_sma)
    print(stock, "EMA", percent_performance_ema)

for plot in plots:
    plot.show()
