import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas_datareader.data as web
import datetime as dt
import matplotlib.dates as mdates

import pickle


class SVM_Predict():

    def __init__(self, ticker, exchange='yahoo', days_forecast=10):
        """
        Just instantiating this class, will predict the price into the future automatically
        :param ticker:
        :param exchange:
        :param days_forecast: 0 means 1% of the entire history
        """
        self.ticker = ticker
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()
        self.forecast_out = days_forecast

        try:
            self.stock = web.DataReader(self.ticker, exchange, start, end)

        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return

        self.prepare_data(self.stock)

        self.X_test, self.X_train, self.y_test, self.y_train = self.train_data(self.X, self.y)

        self.get_bestSVMKernel(self.X_test, self.X_train, self.y_test, self.y_train)

        self.predict()

    def predict(self):
        self.X_lately = self.X[-self.forecast_out:]
        self.X = self.X[:-self.forecast_out]

        self.build_algorithm()

        self.build_forecast()

        self.plot_forecast()

    def plot_forecast(self):
        self.stock['Adj Close'].plot()
        self.stock['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.ylabel(self.ticker + ' Price')
        plt.show()

    def build_forecast(self):
        self.forecast_set = self.clf.predict(self.X_lately)
        print("Confidence {:.2f}%".format(self.confidence * 100))
        print("Days forecasted {}".format(self.forecast_out))

        self.stock['Forecast'] = np.nan
        last_date = self.stock.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400  # seconds in day
        next_unix = last_unix + one_day
        for i in self.forecast_set:
            next_date = dt.datetime.fromtimestamp(next_unix)
            next_unix += 86400  # seconds in day
            self.stock.loc[next_date] = [np.nan for _ in range(len(self.stock.columns) - 1)] + [i]

    def build_algorithm(self):
        # try:
        #     pickle_in = open('linearregression.pickle', 'rb')
        #     self.clf = pickle.load(pickle_in)
        #
        # except Exception as e:
        #     self.clf = svm.SVR(kernel=self.kernel)
        #     self.clf.fit(X_train, y_train)
        #     with open('linearregression.pickle', 'wb') as f:
        #         pickle.dump(self.clf, f)
        #
        #     return

        self.clf = svm.SVR(kernel=self.kernel)
        self.clf.fit(self.X_train, self.y_train)

    def get_bestSVMKernel(self, X_test, X_train, y_test, y_train):
        self.confidence = 0
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
            clf = svm.SVR(kernel=k)
            clf.fit(X_train, y_train)
            self.analyze_confidence(X_test, clf, k, y_test)

    def train_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_test, X_train, y_test, y_train

    def prepare_data(self, stock):
        stock = stock[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        stock['HL_PCT'] = (stock['High'] - stock['Low']) / stock['Adj Close'] * 100.0
        stock['PCT_change'] = (stock['Adj Close'] - stock['Open']) / stock['Open'] * 100.0
        stock = stock[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
        forecast_col = 'Adj Close'
        stock.fillna(value=-99999, inplace=True)
        if self.forecast_out == 0:
            self.forecast_out = int(math.ceil(0.01 * len(stock)))  # 1%
        # stock['label'] = stock[forecast_col].shift(-self.forecast_out)
        stock['label'] = stock[forecast_col].shift(self.forecast_out)

        stock.dropna(inplace=True)
        self.stock = stock.copy()
        self.X = np.array(self.stock.drop(['label'], 1))
        self.y = np.array(self.stock['label'])
        self.X = preprocessing.scale(self.X)
        self.y = np.array(self.stock['label'])

    def analyze_confidence(self, X_test, clf, k, y_test):
        confidence = clf.score(X_test, y_test)
        if self.confidence < confidence:
            self.kernel = k
            self.confidence = confidence


if __name__ == '__main__':
    NIO = SVM_Predict(ticker='NIO', days_forecast=10)
