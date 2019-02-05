import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re

from algorithmictrading.stockData.getTweets import tweetDataRetriever

def execute():
	data = tweetDataRetriever("google").fetchTweet()

	# data["Sentiment"] = np.array(tweet_sentiment(tweet) for tweet in data["text"])
	# print(data.head(10))
	for tweet in data["text"]:
		tweet_sentiment(tweet)


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def tweet_sentiment(tweet):
	print(tweet)
	blob = TextBlob(clean_tweet(tweet))
	print("polarity", blob.sentiment.polarity)
	return float(blob.sentiment.polarity)