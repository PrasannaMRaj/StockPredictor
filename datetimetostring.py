import datetime
import requests
import lxml.html as lh
from bs4 import BeautifulSoup

start="19940101"



fromdate=1270882773
todate=1297321172
#https://www.sharesansar.com/company-chart/history?symbol=NABIL&resolution=1D&from=1270882773&to=1611555929&currencyCode=NPR

s_datetime=datetime.datetime.fromtimestamp(todate).strftime('%c') #from epoch to datetime
print(datetime.datetime(2011,2,10,12,44,32).timestamp()) #from time to epoch


print(s_datetime)

fromdate=datetime.datetime(1994,1,1).timestamp()
todate=datetime.datetime(2021,1,25).timestamp()
url = "https://www.sharesansar.com/company-chart/history"
payload = {"symbol": "NABIL", "from": fromdate, "to": todate, "resolution": "1D", "currencyCode": "NPR"}


page=requests.get(url,params=payload)
        #soup = BeautifulSoup(page.content, 'html.parser')
soup = BeautifulSoup(page.content, 'lxml')
print(soup)
table = soup.find("body")



for table_row in table.findAll('p'):
                dfdata=[]
                trdate = table_row.find('t').text
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
