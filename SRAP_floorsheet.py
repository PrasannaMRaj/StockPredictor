
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
import matplotlib.pyplot as plt
import numpy as np


ticker='NMB'
floorsheetfile='floorsheet'
plottype='kde'


def start_requests(ticker):

        Todate_origin = datetime.datetime(int(time.strftime("%Y")), int(time.strftime("%m")), int(time.strftime("%d")), 23, 59).timestamp()
        Fromdate_origin=datetime.datetime(2000, 1, 1, 0, 0).timestamp()
        print("fromdate: ",Fromdate_origin)
        print("TO date: ", Todate_origin)

        url = "https://newweb.nepalstock.com/api/nots/nepse-data/floorsheet?&size=1000000&sort=contractId,desc"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'}

        page=requests.get(url,headers=headers)

        df = pd.DataFrame(page.json())

        df2=df['floorsheets']['content']
        df_final = pd.DataFrame(df2)


        df_TotalPage=df['floorsheets']['totalPages']
        print(df_TotalPage)

        df_final.to_csv(f'{floorsheetfile}.csv', index=True, encoding="utf-8")

        for pagenumber in range(0,df_TotalPage):
                url="https://newweb.nepalstock.com/api/nots/nepse-data/floorsheet?&size=1000000&sort=contractId,desc"
                payload={'page':(pagenumber+1) }
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'}
                page_url = requests.get(url, headers=headers, params=payload)
                df_append= pd.DataFrame(page_url.json())
                df4 = df_append['floorsheets']['content']
                df_final_append = pd.DataFrame(df4)
                df_final_append.to_csv(f'{floorsheetfile}.csv', index=True, encoding="utf-8", mode='a')

                #df_final.append(df_final_append)

                #df_final_append=df_final_append[1:]



        broker_analysis()


def broker_analysis():
        data=pd.read_csv(f'{floorsheetfile}.csv')
        brokerid=1
        df_ticker=data[(data == ticker).any(axis=1)]
        print(df_ticker)
        df_ticker.to_csv(f'{ticker}_floorsheet.csv', index=True, encoding="utf-8")
        df_ticker['contractRate']=df_ticker['contractRate'].astype(float)
        df_ticker['contractQuantity'] = df_ticker['contractQuantity'].astype(float)

        df_ticker.plot(x='contractRate', y='contractQuantity', kind=plottype,title=ticker)
        df_ticker.plot(x='contractQuantity', y='contractRate', kind=plottype, title=ticker)
        plt.show()


broker_analysis()

#start_requests(ticker)





