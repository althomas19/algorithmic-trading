import pandas as pd

class tweetDataRetriever():
	def __init__(self, stock):
		self.stock = stock

	def fetchTweet(self):
		return pd.read_csv("./twitterCSV/" + self.stock + ".csv")