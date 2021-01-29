
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
from ta.utils import dropna


ticker='NABIL'

def start_requests(ticker):

        Todate_origin = datetime.datetime(int(time.strftime("%Y")), int(time.strftime("%m")), int(time.strftime("%d")), 23, 59).timestamp()
        Fromdate_origin=datetime.datetime(1994, 1, 1, 0, 0).timestamp()
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
        df.columns = df.columns.str.replace("[t]", "date")
        df.columns = df.columns.str.replace("[o]", "open")
        df.columns = df.columns.str.replace("[h]", "high")
        df.columns = df.columns.str.replace("[l]", "low")
        df.columns = df.columns.str.replace("[c]", "adjclose")
        df.columns = df.columns.str.replace("[v]", "volume")

        df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
        #df['RSI'] = RSICalculation(df)
        timewindow=14
        df['RSI'] = CalcRSI(df['adjclose'],timewindow)
        #mom_data = add_all_ta_features(df, open="open", high="high", low="low", close="adjclose", volume="volume")

        df.to_csv(f'{ticker}.csv', index=False, encoding="utf-8")

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




