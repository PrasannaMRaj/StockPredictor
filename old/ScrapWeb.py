from bs4 import BeautifulSoup
import requests
import lxml.html as lh
import pandas as pd
import datetime as dt
import pandas as pd


#page = requests.get("https://money.cnn.com/data/us_markets/")
page = requests.get("http://www.nepalipaisa.com/Modules/CompanyProfile/webservices/CompanyService.asmx/GetCompanyPriceHistory")
soup = BeautifulSoup(page.content, 'html.parser')
print (soup)

