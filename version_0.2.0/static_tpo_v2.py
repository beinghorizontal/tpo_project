# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 07:02:43 2020

@author: alex1
twitter.com/beinghorizontal

"""

import pandas as pd
import plotly.graph_objects as go
from tpo_helper2 import get_ticksize, abc, get_mean, get_rf, get_context, get_dayrank, get_ibrank
import numpy as np
from datetime import timedelta
from plotly.offline import plot

# read file and set datetime index
dfhist = pd.read_csv(r'C:\Users\alex1\Dropbox\scripts\tpo_v2\history.txt')
dfhist.iloc[:, 2:] = dfhist.iloc[:, 2:].apply(pd.to_numeric)
def datetime(data):
    """
    dfhist : pandas series
    Convert date time to pandas date time
    Returns dataframe with datetime index
    """
    dfhist = data.copy()
    dfhist['datetime2'] = pd.to_datetime(dfhist['datetime'], format='%Y%m%d %H:%M:%S')
    dfhist = dfhist.set_index(dfhist['datetime2'], drop=True, inplace=False)
    return dfhist
dfhist = datetime(dfhist)

# manual parameters
freq = 30
avglen = 10  # num days mean to get values
days_to_display = 40  # Number of last n days you want on the screen to display
mode = 'tpo'  # for volume --> 'vol'
# dynamic parameters based on data & mean values for volume and Rotational Factor (RF)
ticksz = get_ticksize(dfhist, freq=freq)
symbol = dfhist.symbol[0]
mean_val = get_mean(dfhist, avglen=avglen, freq=freq)
trading_hr = mean_val['session_hr']

# !!! get rotational factor again for 30 min resampled data
dfhist = get_rf(dfhist.copy())
# !!! resample to desire time frequency. For TPO charts 30 min is optimal
dfresample = dfhist.copy()  # create seperate resampled data frame and preserve old 1 min file
dfresample = dfresample.resample(str(freq)+'min').agg({'symbol': 'last', 'datetime': 'first', 'Open': 'first', 'High': 'max',
                                                       'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
dfresample = dfresample.dropna()

# slice df based on days_to_display parameter
dt1 = dfresample.index[-1]
sday1 = dt1 - timedelta(days_to_display)
dfresample = dfresample[(dfresample.index.date > sday1.date())]

# !!! split the dataframe with new date
DFList = [group[1] for group in dfresample.groupby(dfresample.index.date)]
# !!! for context based bubbles at the top with text hovers
dfcontext = get_context(dfresample, freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr)
#  get market profile DataFrame and ranking as a series for each day.
# @todo: IN next version, display the ranking DataFrame with drop-down menu
dfmp_list = dfcontext[0]
df_distribution = dfcontext[1]
df_ranking = get_dayrank(df_distribution.copy(), mean_val)
ranking = df_ranking[0]
power1 = ranking.power1  # Non-normalised IB strength
power = ranking.power  # Normalised IB strength for dynamic shape size for markers at bottom
breakdown = df_ranking[1]
dh_list = ranking.highd
dl_list = ranking.lowd
# !!! get context based on IB It is predictive value caculated by using various IB stats and previous day's value area
# IB is 1st 1 hour of the session. Not useful for scrips with global 24 x 7 session
context_ibdf = get_ibrank(mean_val, ranking)
ibpower1 = context_ibdf[0].ibpower1  # Non-normalised IB strength
ibpower = context_ibdf[0].IB_power  # Normalised IB strength for dynamic shape size for markers at bottom
ibbreakdown = context_ibdf[1]
ib_high_list = context_ibdf[0].ibh
ib_low_list = context_ibdf[0].ibl

fig = go.Figure(data=[go.Candlestick(x=dfresample['datetime'],

                                     open=dfresample['Open'],
                                     high=dfresample['High'],
                                     low=dfresample['Low'],
                                     close=dfresample['Close'],
                                     showlegend=True,
                                     name=symbol, opacity=0.3)])  # To make candlesticks more prominent increase the opacity

# !!! get TPO for each day
for i in range(len(dfmp_list)):  # test the loop with i=1

    # df1 is used for datetime axis, other dataframe we have is df_mp but it is not a timeseries
    df1 = DFList[i].copy()
    df_mp = dfmp_list[i]
    irank = ranking.iloc[i]  # select single row from ranking df
    # df_mp['i_date'] = df1['datetime'][0]
    df_mp['i_date'] = irank.date
    # # @todo: background color for text
    df_mp['color'] = np.where(np.logical_and(
        df_mp['close'] > irank.vallist, df_mp['close'] < irank.vahlist), 'green', 'white')

    df_mp = df_mp.set_index('i_date', inplace=False)

    fig.add_trace(go.Scattergl(x=df_mp.index, y=df_mp.close, mode="text", name=str(df_mp.index[0]), text=df_mp.alphabets,
                             showlegend=False, textposition="top right", textfont=dict(family="verdana", size=6, color=df_mp.color)))

    #power1 = int(irank['power1']) # non normalized strength
    #power = int(irank['power'])
    if power1[i] < 0:
        my_rgb = 'rgba({power}, 3, 252, 0.5)'.format(power=abs(165))
    else:
        my_rgb = 'rgba(23, {power}, 3, 0.5)'.format(power=abs(252))


    brk_f_list_maj = []
    f = 0
    for f in range(len(breakdown.columns)):
        brk_f_list_min=[]
        for index, rows in breakdown.iterrows():
            brk_f_list_min.append(index+str(': ')+str(rows[f])+'<br />')
        brk_f_list_maj.append(brk_f_list_min)

    breakdown_values =''  # for bubbles
    for st in brk_f_list_maj[i]:
            breakdown_values += st


    # .........................
    ibrk_f_list_maj = []
    g = 0
    for g in range(len(ibbreakdown.columns)):
        ibrk_f_list_min=[]
        for index, rows in ibbreakdown.iterrows():
            ibrk_f_list_min.append(index+str(': ')+str(rows[g])+'<br />')
        ibrk_f_list_maj.append(ibrk_f_list_min)

    ibreakdown_values = ''  # for squares
    for ist in ibrk_f_list_maj[i]:
            ibreakdown_values += ist
    # irank.power1
    # ..................................
    fig.add_trace(go.Scattergl(
        # x=[df1.iloc[4]['datetime']],
        x=[irank.date],
        y=[dfresample['High'].max()],
        mode="markers",
        marker=dict(color=my_rgb, size=0.90*power[i],
                    line=dict(color='rgb(17, 17, 17)', width=2)),
        # marker_symbol='square',
        hovertext=['<br />Insights:<br />VAH:  {}<br /> POC:  {}<br /> VAL:  {}<br /> Balance Target:  {}<br /> Day Type:  {}<br />strength: {}<br />BreakDown:  {}<br />{}<br />{}'.format(irank.vahlist,
                                                                                                                             irank.poclist, irank.vallist,irank.btlist, irank.daytype, irank.power,'','-------------------',breakdown_values)], showlegend=False))
    # !!! we will use this for hover text at bottom for developing day
    if ibpower1[i] < 0:
        ib_rgb = 'rgba(165, 3, 252, 0.5)'
    else:
        ib_rgb = 'rgba(23, 252, 3, 0.5)'

    fig.add_trace(go.Scattergl(
        # x=[df1.iloc[4]['datetime']],
        x=[irank.date],
        y=[dfresample['Low'].min()],
        mode="markers",
        marker=dict(color=ib_rgb, size=0.40 * ibpower[i], line=dict(color='rgb(17, 17, 17)', width=2)),
        marker_symbol='square',
        hovertext=['<br />Insights:<br />Vol_mean:  {}<br /> Vol_Daily:  {}<br /> RF_mean:  {}<br /> RF_daily:  {}<br /> IBvol_mean:  {}<br /> IBvol_day:  {}<br /> IB_RFmean:  {}<br /> IB_RFday:  {}<br />strength: {}<br />BreakDown:  {}<br />{}<br />{}'.format(mean_val['volume_mean'],irank.volumed, mean_val['rf_mean'],irank.rfd,
                   mean_val['volib_mean'], irank.ibvol, mean_val['ibrf_mean'],irank.ibrf, ibpower[i],'','......................',ibreakdown_values)],showlegend=False))

    # @todo: add ib high, low, hd, hl as vertical line at start of each day's start just before above TPOs ib_high_list[i],ib_low_list[i],dh_list[i], dl_list[i]


    lvns = irank.lvnlist

    for lvn in lvns:
        if lvn > irank.vallist and lvn < irank.vahlist:

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

    fig.add_shape(
        # Line Horizontal
        type="line",
        x0=df1.iloc[0]['datetime'],
        y0=ib_low_list[i],
        x1=df1.iloc[0]['datetime'],
        y1=ib_high_list[i],
        line=dict(
            color="cyan",
            width=3,
            ),)
    # day high and low
    fig.add_shape(
        # Line Horizontal
        type="line",
        x0=df1.iloc[0]['datetime'],
        y0=dl_list[i],
        x1=df1.iloc[0]['datetime'],
        y1=dh_list[i],
        line=dict(
            color="gray",
            width=1,
            dash="dashdot",),)



ltp = dfresample.iloc[-1]['Close']
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
fig["layout"]["height"] = 1000
fig["layout"]["width"] = 3000
# fig.layout.hovermode = 'x'

fig.update_xaxes(title_text='Time', title_font=dict(size=18, color='white'),
                 tickangle=45, tickfont=dict(size=8, color='white'), showgrid=False, dtick=len(dfmp_list))

fig.update_yaxes(title_text=symbol, title_font=dict(size=18, color='white'),
                 tickfont=dict(size=12, color='white'), showgrid=False)
fig.layout.update(template="plotly_dark", title="@"+abc()[1], autosize=True,
                  xaxis=dict(showline=True, color='white'), yaxis=dict(showline=True, color='white',autorange= True,fixedrange=False))

fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
fig["layout"]["xaxis"]["tickformat"] = "%H:%M:%S"

# fig.write_html('tpo.html')  # uncomment to save as html

# To save as png
# from kaleido.scopes.plotly import PlotlyScope  # pip install kaleido
# scope = PlotlyScope()
# with open("figure.png", "wb") as f:
#     f.write(scope.transform(fig, format="png"))

plot(fig, auto_open=True)
fig.show()
