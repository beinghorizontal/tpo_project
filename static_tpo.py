# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 05:17:43 2020

@author: alex1
"""

import pandas as pd
import plotly.graph_objects as go
from tpo_helper import get_ticksize, abc, get_mean, get_rf, get_context, get_contextnow
import numpy as np
from datetime import timedelta
from plotly.offline import plot

# from transform import transform_live, transform_hist
# from alpha_dataframe import get_data

# refresh_int = 1  # refresh interval in seconds for live updates
freq = 30
avglen = 10  # num days mean to get values
days_to_display = 10  # Number of last n days you want on the screen to display
mode = 'tpo'  # for volume --> 'vol'

# 1 min historical data in symbol,datetime,open,high,low,close,volume
dfhist = pd.read_csv('history.txt')

# Check the sample file. Match the format exactly else code will not run.

dfhist.iloc[:, 2:] = dfhist.iloc[:, 2:].apply(pd.to_numeric)

# # It calculates tick size for TPO based on mean and standard deviation.
ticksz = get_ticksize(dfhist, freq=freq)
symbol = dfhist.symbol[0]


def datetime(dfhist):
    """
    dfhist : pandas series
    Convert date time to pandas date time
    Returns dataframe with datetime index
    """
    dfhist['datetime2'] = pd.to_datetime(dfhist['datetime'], format='%Y%m%d %H:%M:%S')
    dfhist = dfhist.set_index(dfhist['datetime2'], drop=True, inplace=False)
    return(dfhist)


dfhist = datetime(dfhist)
# Get mean values for context and also get daily trading hours
mean_val = get_mean(dfhist, avglen=avglen, freq=freq)
trading_hr = mean_val['session_hr']
# !!! get rotational factor
dfhist = get_rf(dfhist.copy())
# !!! resample to desire time frequency. For TPO charts 30 min is optimal
dfhist = dfhist.resample(str(freq)+'min').agg({'symbol': 'last', 'datetime': 'first', 'Open': 'first', 'High': 'max',
                                               'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
dfhist = dfhist.dropna()

# slice df based on days_to_display parameter
dt1 = dfhist.index[-1]
sday1 = dt1 - timedelta(days_to_display)
dfhist = dfhist[(dfhist.index.date > sday1.date())]
# !!! concat current data to avoid insufficient bar num error


def live_merge(dfli):
    """
    dfli: pandas dataframe with live quotes.

    This is the live data, and will continue to refresh. Since it merges with historical data keep the format same though source
    can be different..
    For this we only need small sample and if there are duplicate quotes duplicate values will get droped keeping the original value.
    """

    dflive = datetime(dfli)

    dflive = get_rf(dflive.copy())
    dflive = dflive.resample(str(freq)+'min').agg({'symbol': 'last', 'datetime': 'first', 'Open': 'first', 'High': 'max',
                                                   'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})

    df_final = pd.concat([dfhist, dflive])
    df_final = df_final.drop_duplicates()

    return (df_final)


# get live data from external source, it is not inside loop so it gets called 1 time only to make sure historical data is in sync
# provided the sample file. To check live updates are working, add data in live.txt
dfli = pd.read_csv('live.txt')
df_final = live_merge(dfli)
df_final.iloc[:, 2:] = df_final.iloc[:, 2:].apply(pd.to_numeric)

dfli = pd.read_csv('live.txt')  # This is live file in loop to check updates every n seconds
df = live_merge(dfli)
df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric)

# !!! split the dataframe with new date
DFList = [group[1] for group in df.groupby(df.index.date)]
# !!! for context based bubbles at the top with text hovers
dfcontext = get_context(df, freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr)
#  get market profile DataFrame and ranking as a series for each day.
# @todo: IN next version, display the ranking DataFrame with drop-down menu
dfmp_list = dfcontext[0]
ranking = dfcontext[1]
# !!! get context based on IB It is predictive value caculated by using various IB stats and previous day's value area
# IB is 1st 1 hour of the session. Not useful for scrips with global 24 x 7 session
context_ibdf = get_contextnow(mean_val, ranking)
ibpower = context_ibdf.power  # Non-normalised IB strength
ibpower1 = context_ibdf.power1  # Normalised IB strength for dynamic shape size for markers at bottom


fig = go.Figure()

fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     showlegend=True,
                                     name=symbol, opacity=0.3)])  # To make candlesticks more prominent increase the opacity

# !!! get TPO for each day
for i in range(len(dfmp_list)):  # test the loop with i=1

    # df1 is used for datetime axis, other dataframe we have is df_mp but it is not a timeseries
    df1 = DFList[i].copy()
    df_mp = dfmp_list[i]
    irank = ranking.iloc[i]
    # df_mp['i_date'] = df1['datetime'][0]
    df_mp['i_date'] = irank.date
    # # @todo: background color for text
    df_mp['color'] = np.where(np.logical_and(
        df_mp['close'] > irank.vallist, df_mp['close'] < irank.vahlist), 'green', 'white')

    df_mp = df_mp.set_index('i_date', inplace=False)

    fig.add_trace(go.Scatter(x=df_mp.index, y=df_mp.close, mode="text", name=str(df_mp.index[0]), text=df_mp.alphabets,
                             showlegend=False, textposition="top right", textfont=dict(family="verdana", size=6, color=df_mp.color)))
    power = int(irank['power1'])
    if power < 0:
        my_rgb = 'rgba({power}, 3, 252, 0.5)'.format(power=abs(165))
    else:
        my_rgb = 'rgba(23, {power}, 3, 0.5)'.format(power=abs(252))

    fig.add_trace(go.Scatter(
        # x=[df1.iloc[4]['datetime']],
        x=[irank.date],
        y=[df['High'].max()],
        mode="markers",
        marker=dict(color=my_rgb, size=0.40*abs(power),
                    line=dict(color='rgb(17, 17, 17)', width=2)),
        # marker_symbol='square',
        hovertext=['VAH:{}, POC:{}, VAL:{}, Balance Target:{}, Day Type:{}'.format(irank.vahlist, irank.poclist, irank.vallist,
                                                                                   irank.btlist, irank.daytype)], showlegend=False
    ))
    # !!! we will use this for hover text at bottom for developing day
    if ibpower1[i] < 0:
        ib_rgb = 'rgba(165, 3, 252, 0.5)'
    else:
        ib_rgb = 'rgba(23, 252, 3, 0.5)'

    fig.add_trace(go.Scatter(
        # x=[df1.iloc[4]['datetime']],
        x=[irank.date],
        y=[df['Low'].min()],
        mode="markers",
        marker=dict(color=ib_rgb, size=0.40 * \
                    abs(ibpower[i]), line=dict(color='rgb(17, 17, 17)', width=2)),
        marker_symbol='square',
        hovertext=['Vol_mean:{}, Vol_Daily:{}, RF_mean:{}, RF_daily:{}, IBvol_mean:{}, IBvol_day:{}, IB_RFmean:{}, IB_RFday:{}'.format(round(mean_val['volume_mean'], 2),
                                                                                                                                       round(irank.volumed, 2), round(mean_val['rf_mean'], 2), round(
                                                                                                                                           irank.rfd, 2), round(mean_val['volib_mean'], 2),
                                                                                                                                       round(irank.ibvol, 2), round(mean_val['ibrf_mean'], 2), round(irank.ibrf, 2))], showlegend=False
    ))

    lvns = irank.lvnlist

    for lvn in lvns:
        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=df1.iloc[0]['datetime'],
            y0=lvn,
            x1=df1.iloc[5]['datetime'],
            y1=lvn,
            line=dict(
                color="darksalmon",
                width=2,
                dash="dashdot",),)

    excess = irank.excesslist
    if excess > 0:
        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=df1.iloc[0]['datetime'],
            y0=excess,
            x1=df1.iloc[5]['datetime'],
            y1=excess,
            line=dict(
                color="cyan",
                width=2,
                dash="dashdot",),)

# @todo: last price marker. Color code as per close above poc or below
ltp = df1.iloc[-1]['Close']
if ltp >= irank.poclist:
    ltp_color = 'green'
else:
    ltp_color = 'red'

fig.add_trace(go.Scatter(
    x=[df1.iloc[-1]['datetime']],
    y=[df1.iloc[-1]['Close']],
    mode="text",
    name="last traded price",
    text=['last '+str(df1.iloc[-1]['Close'])],
    textposition="bottom right",
    textfont=dict(size=11, color=ltp_color),
    showlegend=False
))

fig.layout.xaxis.color = 'white'
fig.layout.yaxis.color = 'white'
fig.layout.autosize = True
fig["layout"]["height"] = 800
# fig.layout.hovermode = 'x'

fig.update_xaxes(title_text='Time', title_font=dict(size=18, color='white'),
                 tickangle=45, tickfont=dict(size=8, color='white'), showgrid=False, dtick=len(dfmp_list))

fig.update_yaxes(title_text=symbol, title_font=dict(size=18, color='white'),
                 tickfont=dict(size=12, color='white'), showgrid=False)
fig.layout.update(template="plotly_dark", title="@"+abc()[1], autosize=True,
                  xaxis=dict(showline=True, color='white'), yaxis=dict(showline=True, color='white'))

fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
fig["layout"]["xaxis"]["tickformat"] = "%H:%M:%S"

plot(fig, auto_open=True)
fig.show()
