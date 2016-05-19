"""Bollinger Bands."""

#from pykalman import KalmanFilter
import sys
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import pandas_datareader.data as pdr

adj_close_key = 'Adj Close'
default_port = ["SPY", "QQQ", "IWM", "DIA", "XOP", "XLE", "SBUX", "AAPL", "BRK-B", "GLD", "SLV", "TLT", "MU"]

def get_adj_close_data(symbols, start, end):
    """Read stock data (adjusted close) for given symbols from Yahoo Finance."""
    df = pdr.DataReader('SPY', 'yahoo', start, end)[[adj_close_key]]
    df = df.rename(columns={adj_close_key: 'SPY'})

    for symbol in symbols:
        if symbol == 'SPY':
            continue
        df_temp = pdr.DataReader(symbol, 'yahoo', start, end)[[adj_close_key]]
        df_temp = df_temp.rename(columns={adj_close_key: symbol})
        df = df.join(df_temp)

    return df

def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def get_bollinger_bands(rm, rstd, factor):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + (factor * rstd)
    lower_band = rm - (factor * rstd)
    return upper_band, lower_band

def plot_bollinger_bands(symbols, lookback_days):
    price = get_adj_close_data(symbols, '2014-01-01', dt.datetime.today())

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    roll = price.rolling(window=lookback_days, center=False)
    rm = roll.mean()

    # 2. Compute rolling standard deviation
    rstd = roll.std()

    """kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

    km, cov = kf.filter(price.values)"""
    #print(km)
    #print(rm.tail())
    #t = np.diag(cov)
    #rstd = np.sqrt(np.diag(cov))

    #arma = sm.tsa.ARMA(price).fit()

    #print(arma)
    #print(rstd.tail())

    # Compute 1 std move
    upper1, lower1 = get_bollinger_bands(rm, rstd, 1.0)

    # Compute upper and lower bands at 2 std
    upper2, lower2 = get_bollinger_bands(rm, rstd, 2.0)

    # Plot raw values, rolling mean and Bollinger Bands
    #ax = price.plot(title="Bollinger Bands")\
    for i, s in enumerate(symbols):
        plt.figure(i)
        ax = price[s].plot(title="Bollinger Bands", color='b')
        rm[s].plot(label='Rolling mean', ax=ax, color='g')
        #plt.plot(price[s].index, price[s])
        #plt.plot(rm[s].index, rm[s])
        #plt.fill_between(rstd[s].index, upper2[s].index, lower2[s].index, color='b', alpha=0.2)
        #arma[s].plot(label='kalman', ax=ax, color='m')
        upper1[s].plot(label='1 std upper', ax=ax, color='y')
        lower1[s].plot(label='1 std lower', ax=ax, color='y')
        upper2[s].plot(label='2 std upper', ax=ax, color='r')
        lower2[s].plot(label='2 std lower', ax=ax, color='r')
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend(loc='upper left')

    # Add axis labels and legend
    #ax.set_xlabel("Date")
    #ax.set_ylabel("Price")
    #ax.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    moving_avg_days = int(sys.argv[1])
    portfolio = []

    if len(sys.argv) > 2:
        portfolio = [sys.argv[2]]
    else:
        portfolio = default_port

    plot_bollinger_bands(portfolio, moving_avg_days)
