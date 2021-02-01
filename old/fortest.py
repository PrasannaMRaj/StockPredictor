import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

start=dt.datetime(2000,1,1)
end=dt.datetime(2016,12,31)

df=web.DataReader('AAN','yahoo',start,end)
#df.to_csv('tsla.csv')