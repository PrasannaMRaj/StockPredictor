
from bs4 import BeautifulSoup
import requests
import lxml.html as lh
import pandas as pd
import datetime as dt
import pandas as pd
import csv
import datetime
import os

ticker='NABIL'
def write_header():
        Head_data = open(f'{ticker}.csv', 'w', newline='')
        csvwriter = csv.writer(Head_data)
        tableheaddata = []
        openrange = 'open'
        high = 'high'
        low = 'low'
        close = 'adjclose'
        volume = 'volume'
        trdate = 'date'
        tableheaddata.append(trdate)
        tableheaddata.append(openrange)
        tableheaddata.append(high)
        tableheaddata.append(low)
        tableheaddata.append(close)
        tableheaddata.append(volume)
        csvwriter.writerow(tableheaddata)
        Head_data.close()



def start_requests(ticker):


        url = "http://www.nepalipaisa.com/Modules/CompanyProfile/webservices/CompanyService.asmx/GetCompanyPriceHistory"
        payload = {"StockSymbol": ticker, "FromDate": "1994-01-01", "ToDate": "2021-01-28", "Offset": 1, "Limit": 10000}


        page=requests.get(url,params=payload)
        #soup = BeautifulSoup(page.content, 'html.parser')
        soup = BeautifulSoup(page.content, 'lxml')
        #print(soup)

        # open a file for writing
        Resident_data = open(f'{ticker}.csv', 'a', newline='')

        # create the csv writer object
        csvwriter = csv.writer(Resident_data)


        table = soup.find("body")



        for table_row in table.findAll('todayshareprice'):
                dfdata=[]
                trdate = table_row.find('asofdate').text
                date_time_obj = datetime.datetime.strptime(trdate, '%Y-%m-%dT%H:%M:%S')
                #print('Date:', date_time_obj.date())
                dfdata.append(date_time_obj.date())
                openrange = table_row.find('previousclosing').text
                dfdata.append(openrange)
                high=table_row.find('maxprice').text
                dfdata.append(high)
                low = table_row.find('minprice').text
                dfdata.append(low)
                close = table_row.find('closingprice').text
                dfdata.append(close)
                volume = table_row.find('tradedshares').text
                dfdata.append(volume)


                csvwriter.writerow(dfdata)
        Resident_data.close()


def converttodataframe():
        df1=pd.read_csv(f'{ticker}.csv')
        df1['RSI'] = CalcRSI(df1['adjclose'], 14)

        df1.to_csv(f'{ticker}.csv', index=False, encoding="utf-8")


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

if os.path.isfile(f'{ticker}.csv'):
        print("Appending csv file")
else:
        write_header()
start_requests(ticker)
converttodataframe()





