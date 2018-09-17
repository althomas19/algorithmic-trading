import quandl
quandl.ApiConfig.api_key = ##fill in


class stockDataRetriever:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def getStock(self):
        return quandl.get(self.stock, start_date=self.start_date, end_date=self.end_date, returns="pandas")

#mydata = quandl.get("EIA/PET_RWTC_D")

#mydata = quandl.get("FRED/GDP", start_date="2001-12-31", end_date="2005-12-31", returns="numpy",collapse="monthly")

#data = quandl.get_table('ZACKS/FC', ticker='AAPL')

# aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01", returns="numpy")
