import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re

from algorithmictrading.stockData.getTweets import tweetDataRetriever
from algorithmictrading.stockData.getStock import stockDataRetriever

''' twitter part of df is grouped by 3 -- 0 = sentiment, 1= favorite count,
2= retweet count '''
def execute(stock_name, start_date, end_date, fetchStocks):
    if fetchStocks:  # build twitter sentiment and merge into stock data 
        # get tweet data
        data = tweetDataRetriever(stock_name).fetchTweet()
        data["Date"] = ""
        sentiment = []
        for tweet in data["text"]:
            sentiment.append(tweet_sentiment(tweet))
        data["Sentiment"] = sentiment

        # convert twitter date format for merge with stock data 
        date_conv = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', "May": '05', "Jun": '06', "Jul": '07', "Aug": '08', "Sep": '09', "Oct": '10', "Nov": '11', "Dec": '12'}
        for i in range(len(data)):
            raw = data["created_at"][i].split()
            data["Date"][i] = raw[5] + "-" + date_conv[raw[1]] + "-" + raw[2] 

        # drop tweet replies
        for row in range(len(data)): 
            if isinstance(data["in_reply_to_screen_name"][row],str):
                print(row, data["in_reply_to_screen_name"][row])
                data = data.drop(row)

        # aggregate tweets on same day 
        data = data.groupby("Date")[["Sentiment", "favorite_count", "retweet_count"]].apply(lambda x: x.values.tolist())
        data = data.to_frame('twitter').reset_index()

        # merge tweets with stock data 
        data_stock = stockDataRetriever(stock_name, start_date, end_date).fetchStock(fetchStocks)
        df = data_stock.merge(data, how="left", on='Date')
        df.to_csv(stock[5:] + "_MERGED.csv")

    else:
        df = tweetDataRetriever(stock_name).fetch_merged_tweet()

    # create buy/sell signals using different techniques
    df['signal'] = 0
    naive_sentiment(df)

    # graph


    # calculate returns


''' basic twitter trading approach that sells when average
sentiment is negative for the day and buys when > .25 '''
def naive_sentiment(df):
    for i in range(len(df)):
        if len(str(df["twitter"][i])) > 3:  # ignore nan
            sent = df["twitter"][i].split(',')
            trim = "[]"
            sentiment = []
            for s in sent:
                sentiment.append(float(''.join(i for i in s if i not in trim)))

            tot_sentiment = 0
            for j in range(0,len(sentiment),3):
                tot_sentiment += sentiment[j]
            avg_sentiment = tot_sentiment/(len(sentiment)/3)

            if avg_sentiment > .25:
                df["signal"][i] = 1
            elif avg_sentiment < 0:
                df["signal"][i] = -1


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def tweet_sentiment(tweet):
    blob = TextBlob(clean_tweet(tweet))
    return float(blob.sentiment.polarity)
