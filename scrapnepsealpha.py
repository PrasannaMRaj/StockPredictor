
import requests
import csv
import pandas as pd

url="https://backendtradingview.systemxlite.com/tradingViewSystemxLite/history?symbol=ADBL&resolution=1D&from=1580292997&to=1614507457"


print(requests.get(url).json())

page=requests.get(url)
output = page.json()
print (output)

df = pd.DataFrame(output)
df_final =pd.DataFrame()
df_final["open"]=df["o"]
df.to_csv("filename.csv", index=False, encoding="utf-8")






