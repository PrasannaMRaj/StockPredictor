
from bs4 import BeautifulSoup
import requests
import lxml.html as lh
import pandas as pd
import datetime as dt
import pandas as pd
import csv
import datetime
import os

ticker='NMB'
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
        payload = {"StockSymbol": ticker, "FromDate": "1994-01-01", "ToDate": "2021-01-24", "Offset": 1, "Limit": 10000}


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


if os.path.isfile(f'{ticker}.csv'):
        print("Appending csv file")
else:
        write_header()
start_requests(ticker)





