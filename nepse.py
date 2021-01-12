import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

df=pd.read_csv('nabil.csv',parse_dates=True,index_col=0)
df['100ma']=df['Closing Price'].rolling(window=100, min_periods=0).mean()
print(df.head())

ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)

ax1.plot(df.index, df['Closing Price'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Traded Shares'])

plt.show()