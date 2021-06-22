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


class Data_handler:
    def __init__(self, start, end):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def get_df_sticker(self,ticker, start, end):
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

    def save_sp500_tickers(self):
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

    # For each ticker, get the complete data in a csv file. If there is existing data in the csv file,
    # it gets only the remaining data until today, and append it into the csv file
    def get_data_from_sp500(self,reload_sp500=True):
        global start, end
        if reload_sp500:
            tickers = self.save_sp500_tickers()
        else:
            with open("sp500tickers.pickle", "rb") as f:
                tickers = pickle.load(f)

        for ticker in tickers:
            ticker = str(ticker).rstrip('\n')
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = self.get_df_sticker(ticker, start, end)
            else:
                df = pd.read_csv('stock_dfs/' + ticker + '.csv')
                df.set_index("Date", inplace=True, drop=False)
                last_date_in_csv = df['Date'].iloc[-1]
                csv_date = str(last_date_in_csv).split('-')

                if int(csv_date[0]) == dt.datetime.today().year and int(
                        csv_date[1]) == dt.datetime.today().month and int(
                        csv_date[2]) == dt.datetime.today().day:
                    print('stock_dfs/' + ticker + '.csv already updated')
                else:
                    start2 = dt.datetime(int(csv_date[0]), int(csv_date[1]), int(csv_date[2]))
                    start2 = start2 + timedelta(days=1)
                    end2 = dt.datetime.now()
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


if __name__ == "__main__":
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()
    # Individual tickers
    tesla = Data.get_df_sticker("TSLA", start, end)

# Getting all SP500 companies data
# get_data_from_sp500()
# get_sp500_correlation()
