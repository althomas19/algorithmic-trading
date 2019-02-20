import pandas as pd

class tweetDataRetriever():
	def __init__(self, stock):
		self.stock = stock

	def fetch_tweet(self):
		return pd.read_csv("./twitterCSV/" + self.stock[5:] + ".csv")

	def fetch_merged_tweet(self):
		return pd.read_csv("./twitterCSV/" + self.stock[5:] + "_MERGED.csv")