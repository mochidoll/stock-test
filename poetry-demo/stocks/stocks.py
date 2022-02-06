import streamlit as st
import pandas as pd
import datetime
import numpy as np
import pandas_datareader .data as web
from pandas.plotting import scatter_matrix

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

# plotly
import plotly.express as px

st.header("hello")

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()

## Checking the 5 first and 5 last observation of data
st.write(df.head())
st.write(df.tail())

## Moving Average k days
close_px = df['Adj Close']
st.line_chart(close_px)

mavg = close_px.rolling(window=100).mean()

"Mean average of the last 100 days"
st.line_chart(mavg)

f"first mean is {np.mean(close_px[0:100])}, {mavg[99]} is given by"
st.latex(r'''\bar{x}^{1}=\frac{1}{n}\sum_{i=0}^{100}{x_i}''')

"Plotting the closing adjusted value against moving average"
fig, ax = plt.subplots()
ax.plot(close_px, label='AAPL')
ax.plot(mavg, label='mavg')
ax.legend()
ax.grid(False)
ax.set_facecolor('xkcd:white')
st.pyplot(fig)

"Return of the stocks"
rets = close_px / close_px.shift(1) - 1
fig_ret, ax_ret = plt.subplots()
ax_ret.plot(rets, label='return')
ax_ret.legend()
ax_ret.grid(False)
ax_ret.set_facecolor('xkcd:white')
st.pyplot(fig_ret)

## Other stock prices
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
st.write(dfcomp.tail())


## Correlation
retscomp = dfcomp.pct_change()
corr = retscomp.corr()


## plot of correlation
fig_corr, ax_corr = plt.subplots()
ax_corr.scatter(retscomp.AAPL, retscomp.GE)
ax_corr.set_xlabel("Returns AAPL")
ax_corr.set_ylabel("Returns GE")
st.write(fig_corr) #  MAGIC!!!!


## Correlations
fig_cheat = px.imshow(corr)
st.write(fig_cheat)


figg, axx = plt.subplots()
axx.scatter(retscomp.mean(), retscomp.std())
axx.set_xlabel('Expected returns')
axx.set_ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    axx.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
z = np.polyfit(retscomp.mean(), retscomp.std(), 1)
p = np.poly1d(z)
axx.plot(retscomp.mean(), p(retscomp.mean()), "r")

st.write(figg)


dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

st.write(dfreg)