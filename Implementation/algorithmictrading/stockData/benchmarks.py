import quandl
import pandas as pd

from algorithmictrading.stockData.getStock import stockDataRetriever

class benchmark:
    def __init__(self):
        #self.SPY = stockDataRetriever('ETF/SPY', "2006-10-01", "2017-01-01").fetchStock(True)
        #self.percentage_increase = SPY['Close'][len(i)]/SPY['Close'][0]
        SPY_close_2006 = 133.08
        SPY_close_2017 = 223.53
        self.percentage_increase = (SPY_close_2017 - SPY_close_2006)/SPY_close_2006 * 100

    def get_SPY_benchmark(self):
        return self.percentage_increase