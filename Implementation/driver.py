from algorithmictrading.strategy import sma

stockTickers = ["WIKI/AAPL", "WIKI/GOOG", "WIKI/DIS"]
plots = []

# perhaps better to download data at once into csv - and then just access csv??

for stock in stockTickers:
    plots.append(sma.execute(stock, "2006-10-01", "2017-01-01"))

for plot in plots:
    plot.show()
