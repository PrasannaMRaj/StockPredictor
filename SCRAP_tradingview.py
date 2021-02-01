
from bs4 import BeautifulSoup
import requests
import lxml.html as lh
import pandas as pd
import datetime
import pandas as pd
import csv
import json
import os
import time
from ta import add_all_ta_features
from ta.momentum import StochasticOscillator,StochRSIIndicator,RSIIndicator,WilliamsRIndicator
from ta.volatility import AverageTrueRange,BollingerBands,KeltnerChannel
from ta.trend import MACD
from ta.utils import dropna
import numpy as np


ticker='NLICL'

def start_requests(ticker):

        Todate_origin = datetime.datetime(int(time.strftime("%Y")), int(time.strftime("%m")), int(time.strftime("%d")), 23, 59).timestamp()
        Fromdate_origin=datetime.datetime(2000, 1, 1, 0, 0).timestamp()
        #print("fromdate: ",Fromdate_origin)
        #print("TO date: ", Todate_origin)

        #url = "https://da.merolagani.com/handlers/TechnicalChartHandler.ashx"
        #payload = {"symbol": ticker, "resolution": "1D", "rangeStartDate": Fromdate_origin, "rangeEndDate": Todate_origin,
        #                   "isAdjust":0 ,"currencyCode": "NRS","type":"get_advanced_chart"}



        url = "https://nepsealpha.com/trading/0/history"
        payload = {"symbol":ticker , "resolution":"1D" , "from":Fromdate_origin , "to":Todate_origin,"currencyCode":"NRS"}


        page=requests.get(url,params=payload)
        #print(page.json())
        df = pd.DataFrame(page.json())

        df.columns=['date','adjclose','open','high','low','volume','other']


        #df['date'] = datetime.datetime.fromtimestamp(df['date']).strftime('%Y-%m-%d')
        #df['date']= df['date'].date()
        #df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
        del df['date']
        #<-df['date'] = df['date'].map(lambda val: datetime.datetime.fromtimestamp(val).strftime('%Y-%m-%d'))
        #<-df['date'] = pd.to_datetime(df['date'])
        #df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y-%m-%d')
        timewindow=14
        df = df.replace(0, np.nan)


        #df['RSI'] = CalcRSI(df['adjclose'],timewindow)

        RSI_data=RSIIndicator(close=df['adjclose'], window=timewindow)
        df['RSI']=RSI_data.rsi()

        StochOsc_data = StochasticOscillator(high=df['high'], low=df['low'], close=df['adjclose'])
        df['stoch']=StochOsc_data.stoch()
        #df['stochsignal']=StochOsc_data.stoch_signal()

        StochRSI_data = StochRSIIndicator(close=df['adjclose'])
        df['momentum_stoch_rsi']=StochRSI_data.stochrsi()
        df['momentum_stoch_rsi_k'] = StochRSI_data.stochrsi_k()
        df['momentum_stoch_rsi_d'] = StochRSI_data.stochrsi_d()

        MACD_data = MACD(window_slow=17, window_fast=8,close=df['adjclose'])
        df['macd']=MACD_data.macd()
        df['macddiff'] = MACD_data.macd_diff()

        Volatility_data=AverageTrueRange(close=df['adjclose'], high=df['high'], low=df['low'], window=10)
        df['volatility_atr']=Volatility_data.average_true_range()

        BollingerData=BollingerBands(close=df['adjclose'], window=20)
        df['volatility_bbm']=BollingerData.bollinger_mavg()
        df['volatility_bbl']=BollingerData.bollinger_lband()
        df['volatility_bbh']=BollingerData.bollinger_hband()
        df['volatility_bbw']=BollingerData.bollinger_wband()

        Keltner_data=KeltnerChannel(close=df['adjclose'], high=df['high'], low=df['low'], window=10)
        df['volatility_kch']=Keltner_data.keltner_channel_hband()
        df['volatility_kcl']=Keltner_data.keltner_channel_lband()
        df['volatility_kcw']=Keltner_data.keltner_channel_wband()

        WilliamsR_data=WilliamsRIndicator(high=df['high'], low=df['low'], close=df['adjclose'], lbp=14)
        df['WilliamR']=WilliamsR_data.williams_r()

        #ALLTechnical_data = add_all_ta_features(df, open="open", high="high", low="low", close="adjclose", volume="volume")




        df.to_csv(f'{ticker}.csv', index=True, encoding="utf-8")

def AddTechnicalIndicators(wholedata):
        #wholedata is the dataframe containing csv data of stock
        mom_data = add_all_ta_features(wholedata, open="open", high="high", low="low", close="adjclose", volume="volume")
        #print(mom_data.columns)

def AddMomentumIndicators(wholedata):
        #wholedata is the dataframe containing csv data of stock
        mom_data = add_all_ta_features(wholedata, open="open", high="high", low="low", close="adjclose", volume="volume")
        #print(mom_data.columns)


def CalcRSI(data,time_window):
        diff = data.diff(1).dropna()  # diff in one field(one day)

        # this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[diff > 0]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[diff < 0]

        # check pandas documentation for ewm
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

        rs = abs(up_chg_avg / down_chg_avg)
        rsi = 100 - 100 / (1 + rs)
        return rsi

start_requests(ticker)





