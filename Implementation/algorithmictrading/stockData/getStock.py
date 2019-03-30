import quandl
import pandas as pd

quandl.ApiConfig.api_key = ''  # fill in here


class stockDataRetriever:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def getStock(self):
        return quandl.get(self.stock, start_date=self.start_date, end_date=self.end_date, returns="pandas")

    def getStockCSV(self):
        return quandl.get(self.stock, start_date=self.start_date, end_date=self.end_date, returns="pandas").to_csv("./stockCSV/" + self.stock[5:] + ".csv")

    def fetchStock(self, fetchStocks):
        if (fetchStocks):
            self.getStockCSV()
        return pd.read_csv("./stockCSV/" + self.stock[5:] + ".csv")
