import quandl
quandl.ApiConfig.api_key = ''  # fill in here


class stockDataRetriever:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def getStock(self):
        return quandl.get(self.stock, start_date=self.start_date, end_date=self.end_date, returns="pandas")
