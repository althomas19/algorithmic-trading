from algorithmictrading.strategy import sma

stockTickers = ["WIKI/AAPL", "WIKI/GOOG", "WIKI/DIS", 'WIKI/FB', 'WIKI/INTC']
plots = []

for stock in stockTickers:
    plot, percent_performance = sma.execute(stock, "2006-10-01", "2017-01-01")
    plots.append(plot)

    print(stock, percent_performance)

# for plot in plots:
#     plot.show()
