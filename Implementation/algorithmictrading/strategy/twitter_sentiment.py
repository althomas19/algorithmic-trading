import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


from algorithmictrading.stockData.getTweets import tweetDataRetriever
from algorithmictrading.stockData.getStock import stockDataRetriever

''' twitter part of df is grouped by 3 -- 0 = sentiment, 1= favorite count,
2= retweet count '''
def execute(stock_name, start_date, end_date, fetchStocks):
    if fetchStocks:  # build twitter sentiment and merge into stock data 
        df = build_df(stock_name, start_date, end_date)
    else:
        df = tweetDataRetriever(stock_name).fetch_merged_tweet()

    # create buy/sell signals using different techniques
    #machine_learning_sentiment(df, stock_name)
    deep_learning_sentiment(df, stock_name)

    # graph
    # plot average tweet sentiment to rolling average of stock price -- to try to see if we can get good signals


# create columns for average sentiment, num tweets,std dev? 
# look through to see any sort of stock related announcement -- boolean for 

def build_df(stock_name, start_date, end_date):
    # get tweet sentiment
    print("getting tweet sentiment")
    data = tweetDataRetriever(stock_name).fetch_tweet()
    data["Date"] = ""
    sentiment = []
    for tweet in data["text"]:
        sentiment.append(tweet_sentiment(tweet))
    data["Sentiment"] = sentiment

    # convert twitter date format for merge with stock data 
    print("converting dates")
    date_conv = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', "May": '05', "Jun": '06', "Jul": '07', "Aug": '08', "Sep": '09', "Oct": '10', "Nov": '11', "Dec": '12'}
    for i, row in data.iterrows():
        raw = data["created_at"][i].split()
        #data["Date"][i] = raw[5] + "-" + date_conv[raw[1]] + "-" + raw[2] 
        data.loc[i, "Date"] = raw[5] + "-" + date_conv[raw[1]] + "-" + raw[2] 

    # drop tweet replies
    print("dropping replies")
    for i, row in data.iterrows():
        if isinstance(data["in_reply_to_screen_name"][i], str):
            #print(row, data["in_reply_to_screen_name"][row])
            data = data.drop(i)

    # aggregate tweets on same day 
    print("aggregating tweets")
    data = data.groupby("Date")[["Sentiment", "favorite_count", "retweet_count"]].apply(lambda x: x.values.tolist())
    data = data.to_frame('twitter').reset_index()

    # merge tweets with stock data 
    print("merging")
    data_stock = stockDataRetriever(stock_name, start_date, end_date).fetchStock(True)
    df = data_stock.merge(data, how="left", on='Date')

    # create average sentiment
    print("creating average sentiment, other important values")
    average_sentiment_other_values(df)

    # create indicators for supervised learning
    print("creating indicators for learning")
    df["change"] = df["Close"].diff()
    df["change"] = df["change"].fillna(0)
    df["change_int"] = 0
    for i, row in df.iterrows():
        val = -1
        if float(df["change"][i]) > 0:
            val = 1
        df.loc[i, "change_int"] = val

    df.to_csv("./twitterCSV/" + stock_name[5:] + "_MERGED.csv")
    return df

''' basic twitter trading approach that sells when average
sentiment is negative for the day and buys when > .25 '''
def naive_sentiment(df):
    df['signal'] = 0
    for i in range(len(df)):
        if df["avg_sentiment"][i] > .25:
            df["signal"][i] = 1
        elif df["avg_sentiment"][i] < 0:
            df["signal"][i] = -1


def machine_learning_sentiment(df, stock_name):  # model may see the trend for us!
    # SUPRESSES WARNINGS
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn  
    # find first tweet date
    for row in df.itertuples():
        if len(str(df["twitter"][row.Index])) > 3:
            break
    first_tweet = row.Index 
    #pre_df = df[first_tweet:].drop(["Date", "twitter", "Ex-Dividend", "Split Ratio"], axis=1) # get rid of strings
    #pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment", "total_tweets", "total_retweets", "total_favorites", "change", "change_int"]]
    # pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment", "total_tweets", "total_retweets", "total_favorites", "change", "change_int"]]
    # # pre_df["SMA"] = pre_df['Close'].rolling(window=60, min_periods=1, center=False).mean()
    # # pre_df["EMA"] = pre_df['Close'].ewm(span=40, adjust=False).mean()
    pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment", "change", "change_int"]]
    pre_df["change_predicted"] = 0
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(pre_df)
    # pre_df.loc[:,:] = x_scaled
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(pre_df["avg_sentiment"].reshape(-1,1))
    pre_df["avg_sentiment"] = x_scaled
    # print(np.corrcoef(pre_df["avg_sentiment"],pre_df["Close"]))
    # print(np.corrcoef(pre_df["total_favorites"],pre_df["Close"]))
    # print(np.corrcoef(pre_df["total_retweets"],pre_df["Close"]))
    # print(np.corrcoef(pre_df["total_tweets"],pre_df["Close"]))
    print("testing effectiveness of", stock_name)



    # Using Random Forest Regressor 
    train = pre_df[:int(len(pre_df)*2/3)].drop(["change_int"], axis=1)
    test = pre_df[-int(len(pre_df)*1/3):].drop(["change", "change_int"], axis=1)
    X = train.drop(["change"],axis=1)
    y = train["change"]
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X,y)
    print(type(rf), rf.score(X,y)) # gives r2
    test["change_predicted"] = rf.predict(test)
    test = test.reset_index(drop=True)
    #print(rf.predict(test))
    # test["change"] = 0
    # test["change_int"] = 0
    # transformed = pd.DataFrame(scaler.inverse_transform(test))
    # transformed.columns = pre_df.columns
    # profit(transformed, stock_name, type(rf))
    profit(test, stock_name, type(rf))

    train = pre_df[:int(len(pre_df)*2/3)].drop(["change"], axis=1)
    test = pre_df[-int(len(pre_df)*1/3):].drop(["change_int", "change"], axis=1)
    test_profit = test.copy(deep=True)
    X = train.drop(["change_int"], axis=1)
    y = train["change_int"] # NEED TO USE INTS FOR THE NEURAL NETWORK STUFF
    

    clfs = [
        MLPClassifier(alpha=1),
        DecisionTreeClassifier(),  # BY FAR THE BEST!
        KNeighborsClassifier(n_neighbors=3)]
        #QuadraticDiscriminantAnalysis()]

    for clf in clfs:
        clf.fit(X,y)
        #if cross_val_score(clf, X, y, scoring='accuracy').mean() == 1:
        #print(clf.predict(test))
        print(type(clf), cross_val_score(clf, X, y, scoring='accuracy').mean()) #THIS ONLYWORKS FOR CLASSIFIERS
        predicted_movement = clf.predict(test)
        #print(predicted_movement)
        test_profit["change_predicted"] = predicted_movement
        test_profit = test_profit.reset_index(drop=True)
        # test_profit["change"] = 0
        # test_profit["change_int"] = 0
        # print(list(test_profit.columns.values))
        # transformed = pd.DataFrame(scaler.inverse_transform(test_profit))
        # transformed.columns = pre_df.columns
        # transformed.to_csv('test.csv')
        # profit(transformed, stock_name, type(clf))
        profit(test_profit, stock_name, type(clf))

def deep_learning_sentiment(df, stock_name):
    # Multiple input series LSTM
    for row in df.itertuples():
        if len(str(df["twitter"][row.Index])) > 3:
            break
    first_tweet = row.Index 
    #pre_df = df[first_tweet:].drop(["Date", "twitter", "Ex-Dividend", "Split Ratio"], axis=1) # get rid of strings
    #pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment", "total_tweets", "total_retweets", "total_favorites", "change", "change_int"]]
    #pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment", "total_tweets", "total_retweets", "total_favorites"]]
    pre_df = df.loc[first_tweet:, ["Close", "avg_sentiment"]]  # Close must be first in df!
    dataset = pre_df.values

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    data_train = dataset[0:int(len(dataset)*2/3), :]
    data_test = dataset[int(len(dataset)*2/3):len(dataset), :]

    look_back = 3

    x_train, y_train = [], []
    for i in range(look_back, len(data_train)):
        x_train.append(data_train[i-look_back:i])
        y_train.append(data_train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_test, y_test = [], []
    for i in range(look_back, len(data_test)):
        x_test.append(data_test[i-look_back:i])
        y_test.append(data_test[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    model = Sequential() # input shape - first is the time period used, second is amount of inputed columns (features)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, x_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    predict = model.predict(x_test, verbose=0)

    predicted = np.append(predict, np.ones([len(predict),1]),1)

    predicted = scaler.inverse_transform(predicted)

    #print(predicted)

    test_df = pre_df[int(len(dataset)*2/3) + look_back:]

    testScore = math.sqrt(mean_squared_error(test_df["Close"], predicted[:, 0]))
    print('Test Score of %s: %.2f RMSE' % (stock_name, testScore))

    print(test_df.columns)

    test_df["predicted"] = predicted[:, 0]
    test_df = test_df.reset_index(drop=True)

    for i, row in test_df.iterrows(): # THIS ISNT ACTUALLY WORKING!!!!
        if test_df["Close"].iloc[i] < test_df["predicted"].iloc[i]:
            change_predicted = 1
        else:
            change_predicted = -1
        test_df.loc[i, "change_predicted"] = change_predicted

    # MAYBE LOOK TO FIX THE NP.RANDOMSEED

    profit(test_df, stock_name, type(model))


def average_sentiment_other_values(df):
    df["avg_sentiment"] = 0
    df["total_tweets"] = 0
    df["total_retweets"] = 0
    df["total_favorites"] = 0
    for row in df.itertuples():
        i = row.Index
        if len(str(df["twitter"][i])) > 3:  # ignore nan
            sent = str(df["twitter"][i]).split(',')
            trim = "[]"
            sentiment = []
            for s in sent:
                sentiment.append(float(''.join(i for i in s if i not in trim)))

            tot_sentiment = 0
            tot_rt = 0
            tot_fave = 0
            for sent,rt,fv in zip(*[iter(sentiment)]*3):
                tot_sentiment += sent
                tot_rt += rt
                tot_fave += fv

            avg_sentiment = float(tot_sentiment)/(len(sentiment)/3)

            df.loc[i, "avg_sentiment"] = avg_sentiment
            df.loc[i, "total_tweets"] = len(sentiment)/3
            df.loc[i, "total_retweets"] = tot_rt
            df.loc[i, "total_favorites"] = tot_fave


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def tweet_sentiment(tweet):
    blob = TextBlob(clean_tweet(tweet))
    return float(blob.sentiment.polarity)


def profit(df, name, clf_type):

    #df = df.dropna()

    bank = 1000
    base = 1000
    shares = 0

    baseline_shares = int(bank / df["Close"].iloc[0])
    baseline_value = bank - df["Close"].iloc[0]*baseline_shares

    for i, row in df.iterrows():
        if df["change_predicted"][i] > 0:
            purchased = int(float(bank) / df["Close"].iloc[i])
            bank -= purchased * df["Close"].iloc[i]
            shares += purchased
        else:
            bank += float(df["Close"].iloc[i]) * shares
            shares = 0
        #print(bank, shares)
    bank += df["Close"].iloc[i] * shares

    baseline_value += baseline_shares * df["Close"].iloc[len(df) - 1]

    print(name, clf_type, "Total Money:", bank, "percent_return", (bank-base)/base * 100, "baseline:",
          baseline_value, "years of trading:", "{0:.2f}".format((len(df)-1)/float(252)))
