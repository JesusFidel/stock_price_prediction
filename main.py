import os
import time
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

import bs4 as bs
import pickle
import requests
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from time_vars import TimeUtils

# Init
style.use('ggplot')
start = dt.datetime(2000, 1, 1)
end = dt.datetime.now()


def set_moving_averages_in_df(df, mas):
    for days_ma in mas:
        ma = str(days_ma) + 'ma'
        df[ma] = df['Adj Close'].rolling(window=days_ma, min_periods=0).mean()


def plot_df_with_moving_averages(df, mas):
    set_moving_averages_in_df(df, mas)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.plot(df.index, df['Adj Close'])
    for days_ma in mas:
        ax1.plot(df.index, df[str(days_ma) + 'ma'])
    ax2.bar(df.index, df['Volume'])
    plt.show()


def plot_df_candlesticks(df, dayspercandle):
    days = str(dayspercandle) + 'D'
    df_ohlc = df['Adj Close'].resample(days).ohlc()
    df_volume = df['Volume'].resample(days).sum()
    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()
    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.show()


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)
    return tickers


def get_df_sticker(ticker, start, end):
    if "." in ticker:
        ticker = ticker.replace('.', '-')
    try:
        df = pdr.DataReader(ticker, 'yahoo', start, end)
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True, drop=False)
        df.dropna(inplace=True)
        if not os.path.exists('stock_dfs'):
            os.makedirs('stock_dfs')
        df.to_csv('stock_dfs/' + ticker + '.csv', mode='a', header=True, index=False)
        print('stock_dfs/{}.csv stored'.format(ticker))
        return df

    except Exception as e:
        print(e)
        print('Failed to get ticker: {}'.format(ticker))

    return None


# For each ticker, get the complete data in a csv file. If there is existing data in the csv file,
# it gets only the remaining data until today, and append it into the csv file
def get_data_from_sp500(reload_sp500=True):
    global start, end
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    for ticker in tickers:
        ticker = str(ticker).rstrip('\n')
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = get_df_sticker(ticker, start, end)
        else:
            df = pd.read_csv('stock_dfs/' + ticker + '.csv')
            df.set_index("Date", inplace=True, drop=False)
            last_date_in_csv = df['Date'].iloc[-1]
            csv_date = str(last_date_in_csv).split('-')

            if int(csv_date[0]) == dt.datetime.today().year and int(csv_date[1]) == dt.datetime.today().month and int(
                    csv_date[2]) == dt.datetime.today().day:
                print('stock_dfs/' + ticker + '.csv already updated')
            else:
                start2 = dt.datetime(int(csv_date[0]), int(csv_date[1]), int(csv_date[2]))
                start2 = start2 + timedelta(days=1)
                end2 = dt.datetime.now()
                try:
                    df2 = pdr.DataReader(ticker, 'yahoo', start2, end2)
                    df2.reset_index(inplace=True)
                    df2.dropna(inplace=True)
                    last_date_in_df2 = df2['Date'].iloc[-1]
                    df2_date = str(last_date_in_df2).split('-')
                    df2_day = df2_date[2]
                    if df2_date[0] != csv_date[0] or df2_date[1] != csv_date[1] or df2_day[:2] != csv_date[
                        2]:  # Today is not always the last date in the cvs, due to the weekends
                        df2.to_csv('stock_dfs/' + ticker + '.csv', mode='a', header=False, index=False)
                    print('stock_dfs/' + ticker + '.csv updated')
                except Exception as e:
                    print(e)
                    print('Failed to update ticker: {}'.format(ticker))


def compile_sp500_data():

    """
    compile all data frames data inside stock_dfs into one large csv called: sp500_joined_closes.csv

    :return:
    """
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        ticker = str(ticker).rstrip('\n')
        ticker = ticker.replace(".", "-")
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        # df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        # df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


def visualize_data():
    """
    Read compiled sp500 data "sp500_joined_closes.csv" and visualize the correlation between all stocks
    :return:
    """
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    # That's seriously it. The .corr() automatically will look at the entire DataFrame,
    # and determine the correlation of every column to every column.
    df_corr.to_csv('sp500corr.csv')
    # Instead, we're going to graph it. To do this, we're going to make a heatmap. There isn't a super simple heat
    # map built into Matplotlib, but we have the tools to make on anyway.
    data1 = df_corr.values  # To do this, first we need the actual data itself to graph:
    # This will give us a numpy array of just the values, which are the correlation numbers. Next, we'll build our
    # figure and axis:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # Now, we create the heatmap using pcolor:
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    # This heatmap is made using a range of colors, which can be a range of anything to anything, and the color scale
    # is generated from the cmap that we use.
    # You can find all of the options for color maps here. We're going to use RdYlGn, which is a colormap that goes
    # from red on the low side, yellow for
    # the middle, and green for the higher part of the scale, which will give us red for negative correlations,
    # green for positive correlations,
    # and yellow for no-correlations. We'll add a side-bar that is a colorbar as a sort of "scale" for us:
    fig1.colorbar(heatmap1)
    # Next, we're going to set our x and y axis ticks so we know which companies are which, since right now we've
    # only just plotted the data:
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    # What this does is simply create tick markers for us. We don't yet have any labels.
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    # This will flip our yaxis, so that the graph is a little easier to read, since there will be some space between
    # the x's and y's.
    # Generally matplotlib leaves room on the extreme ends of your graph since this tends to make graphs easier to
    # read, but, in our case,
    # it doesn't. Then we also flip the xaxis to be at the top of the graph, rather than the traditional bottom,
    # again to just make this
    # more like a correlation table should be. Now we're actually going to add the company names to the
    # currently-nameless ticks:
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    # In this case, we could have used the exact same list from both sides, since column_labels and row_lables
    # should be identical lists.
    # This wont always be true for all heatmaps, however, so I decided to show this as the proper method for
    # just about any heatmap from a dataframe. Finally:
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    # plt.savefig("correlations.png", dpi = (300))
    plt.show()
    # We rotate the xticks, which are the actual tickers themselves, since normally they'll be written out
    # horizontally.
    # We've got over 500 labels here, so we're going to rotate them 90 degrees so they're vertical. It's
    # still a graph that's going to
    # be far too large to really see everything zoomed out, but that's fine. The line that says
    # heatmap1.set_clim(-1,1) just tells
    # the colormap that our range is going to be from -1 to positive 1. It should already be the case,
    # but we want to be certain.
    # Without this line, it should still be the min and max of your dataset, so it would have been pretty close anyway.


def get_sp500_correlation():
    compile_sp500_data()
    visualize_data()


"""
Preprocessing data to prepare for Machine Learning with stock data - Python Programming for Finance p.9 - NOT WORKING
"""


def process_data_for_labels(ticker):
    """
    This function will take one parameter: the ticker in question. Each model will be trained on a single company.
    Next, we want to know how many days into the future we need prices for. We're choosing 7 here. Now, we'll read in
    the data for the close prices for all companies that we've saved in the past, grab a list of the existing tickers,
    and we'll fill any missing with 0 for now. This might be something you want to change in the future, but we'll go
    with 0 for now. Now, we want to grab the % change values for the next 7 days:
    :param ticker:
    :return: tickers, df
    """
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    """
    The capital X contains our feature sets (daily % changes for every company in the S&P 500). The lowercase y is our 
    "target" or our "label." Basically what we're trying to map our feature sets to.
    """
    df.fillna(0, inplace=True)
    return df, tickers


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


# Creating targets for machine learning labels Python Programming for Finance p.10 and 11
def extract_featuresets(ticker):
    """
    This function will take any ticker, create the needed dataset, and create our "target" column, which is our label.
    The target column will have either a -1, 0, or 1 for each row, based on our function and the columns we feed through
    Now, we can get the distribution:

    """

    df, tickers = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    # This function will take any ticker, create the needed dataset, and create our "target" column, which is our label.
    # The target column will have either a -1, 0, or 1 for each row, based on our function and the columns we feed
    # through.
    # Now, we can get the distribution:
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    """
    We probably have some totally missing data, which we'll replace with 0. Next we probably have some infinite data,
    especially if we did a percent change from 0 to anything. We're going to convert infinite values to NaNs, then
    we're going to drop NaNs. We're *almost* ready to rumble, but right now our "features" are that day's prices for
    stocks. Just static numbers, really nothing telling at all. Instead, a better metric would be every company's
    percent change that day. The idea here being that some companies will change in price before others, and we can
    profit maybe on the laggards. We'll convert the stock prices to % changes:
    """

    """ 
    We probably have some totally missing data, which we'll replace with 0. Next we probably have some infinite data,
    especially if we did a percent change from 0 to anything. We're going to convert infinite values to NaNs, then
    we're going to drop NaNs. We're *almost* ready to rumble, but right now our "features" are that day's prices for
    stocks. Just static numbers, really nothing telling at all. Instead, a better metric would be every company's
    percent change that day. The idea here being that some companies will change in price before others, and we can
    profit maybe on the laggards. We'll convert the stock prices to % changes:
    """
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    # Again, being careful about infinite numbers, and then filling any other missing data, and, now, finally,
    # we are ready to create our features and labels:
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    # The capital X contains our feature sets (daily % changes for every company in the S&P 500). The lowercase y is
    # our "target" or our "label." Basically what we're trying to map our feature sets to.
    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    return confidence


if __name__ == "__main__":
    # Individual tickers
    # tesla = get_df_sticker("TSLA", start, end)
    # plot_df_with_moving_averages(tesla, [50, 100])
    # plot_df_candlesticks(tesla, 10)

    # Getting all SP500 companies data
    get_data_from_sp500()
# get_sp500_correlation()
# do_ml('XOM')
# do_ml('AAPL')
# do_ml('ABT')
