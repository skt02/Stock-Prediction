import os
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
from collections import Counter
# import matplotlib.dates as mdates
import plotly.graph_objects as go
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
import pydotplus



from keras.utils.vis_utils import plot_model

from keras.utils.vis_utils import plot_model

style.use('ggplot')


def get_data(symbol, index):
    # company_df = pd.DataFrame()
    start = dt.datetime(2015, 1, 1)
    end = dt.date.today()-dt.timedelta(hours=24)
    try:
        company_df = web.DataReader(symbol, 'yahoo', start, end)
        company_df.to_csv('{}.csv'.format(symbol))
    except:
        try:
            os.remove('{}.csv'.format(symbol))
        except:
            pass
        df.drop(labels=index, axis=0, inplace=True)


def getMovingAverage(symbol):
    df = pd.read_csv('{}.csv'.format(symbol), parse_dates=True, index_col=0)
    df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
    df.dropna(inplace=True)
    return df


def plotMovingAverage(df):
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.plot(df.index, df['Adj Close'])
    ax1.plot(df.index, df['100ma'])
    ax2.bar(df.index, df['Volume'])
    plt.show()


def plt_ohlc(df):
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']))
    fig.show()
    # ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax1.xaxis_date()
    # mpf.plot(data, type='candle', style='yahoo')
    # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    # plt.show()


def resample_data(df):
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    df_ohlc.reset_index(inplace=True)
    return df_ohlc

# df = pd.read_csv("constituents_csv.csv")
# for index, row in df.iterrows():
#     get_data(row['Symbol'], index)
#
# df = df.to_csv("constituents_csv.csv")

def compile_data():
    main_df = pd.DataFrame()
    df = pd.read_csv("constituents_csv.csv")
    for index, row in df.iterrows():
        try:
            df = pd.read_csv('{}.csv'.format(row['Symbol']))
        except:
            continue
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close':row['Symbol']}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


def correlation_table():
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    data1 = df_corr.values

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    heatmap = ax1.pcolor(data1, cmap = plt.cm.RdYlGn)
    ax1.set_xticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


def process_data_for_labels(Symbol):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    Symbols = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(Symbol, i)] = (df[Symbol].shift(-i) - df[Symbol]) / df[Symbol]

    df.fillna(0, inplace=True)
    return  Symbols, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)  # replacing infinite values with 0
    df_vals.fillna(0, inplace=True)  # replacing na with 0

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence


def lstm(ticker):
    df = pd.read_csv('{}.csv'.format(ticker), index_col='Date', parse_dates=True, infer_datetime_format=True)
    # print(df.columns)

    # outputVariable and Input Variable
    output_var = pd.DataFrame(df['Adj Close'])
    features = ['Open', 'High', 'Low', 'Volume']

    #Scaling
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index) + len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index) + len(test_index))].values.ravel()

    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences = False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer ='adam')
    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)
    y_pred = lstm.predict(X_test)
    plt.plot(y_test, label='True Value')
    plt.plot(y_pred, label='LSTM Value')
    plt.title("Prediction by LSTM")
    plt.xlabel("Time Scale")
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.show()


df = getMovingAverage('MMM')
df_ohlc = resample_data(df)
# print(df_ohlc)
plt_ohlc(df_ohlc)
correlation_table()
plotMovingAverage(df)
df.drop(df.columns[len(df.columns)-1], axis = 1, inplace=True)
company_df = web.DataReader('TSLA', 'yahoo', start, end)
# print(company_df.head())
# print(df.columns)
do_ml('MMM')
lstm('MMM')

main.py9 KB
