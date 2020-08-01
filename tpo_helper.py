# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:26:04 2020

@author: alex1
"""
import pandas as pd
import numpy as np
import math
# import itertools


def get_ticksize(data, freq=30):
    # data = df
    numlen = int(len(data)/2)
    # sample size for calculating ticksize = 50% of most recent data
    tztail = data.tail(numlen).copy()
    tztail['tz'] = tztail.Close.rolling(freq).std()  # std. dev of 30 period rolling
    tztail = tztail.dropna()
    ticksize = np.ceil(tztail['tz'].mean()*0.25)  # 1/4 th of mean std. dev is our ticksize

    if ticksize < 0.2:
        ticksize = 0.2  # minimum ticksize limit

    return int(ticksize)


def abc(session_hr=6.5, freq=30):

    caps = [' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' J', ' K', ' L', ' M',
            ' N', ' O', ' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z']
    abc_lw = [x.lower() for x in caps]
    Aa = caps + abc_lw
    alimit = math.ceil(session_hr * (60 / freq)) + 3
    if alimit > 52:
        alphabets = Aa * int(
            (np.ceil((alimit - 52) / 52)) + 1)  # if bar frequency is less than 30 minutes then multiply list
    else:
        alphabets = Aa[0:alimit]
    bk = [28, 31, 35, 40, 33, 34, 41, 44, 35, 52, 41, 40, 46, 27, 38]
    ti = []
    for s1 in bk:
        ti.append(Aa[s1 - 1])
    tt = (''.join(ti))

    return (alphabets, tt)


def tpo(dft_rs, freq=30, ticksize=10, style='tpo', session_hr=6.5):

    if len(dft_rs) > int(60/freq):
        dft_rs = dft_rs.drop_duplicates('datetime')
        dft_rs = dft_rs.reset_index(inplace=False, drop=True)
        dft_rs['rol_mx'] = dft_rs['High'].cummax()
        dft_rs['rol_mn'] = dft_rs['Low'].cummin()
        dft_rs['ext_up'] = dft_rs['rol_mn'] > dft_rs['rol_mx'].shift(2)
        dft_rs['ext_dn'] = dft_rs['rol_mx'] < dft_rs['rol_mn'].shift(2)

        alphabets = abc(session_hr, freq)[0]
        alphabets = alphabets[0:len(dft_rs)]
        hh = dft_rs['High'].max()
        ll = dft_rs['Low'].min()
        day_range = hh - ll
        dft_rs['abc'] = alphabets
        # place represents total number of steps to take to compare the TPO count
        place = int(np.ceil((hh - ll) / ticksize))
        # kk = 0
        abl_bg = []
        tpo_countbg = []
        pricel = []
        volcountbg = []
        # datel = []
        for u in range(place):
            abl = []
            tpoc = []
            volcount = []
            p = ll + (u*ticksize)
            for lenrs in range(len(dft_rs)):
                if p >= dft_rs['Low'][lenrs] and p < dft_rs['High'][lenrs]:
                    abl.append(dft_rs['abc'][lenrs])
                    tpoc.append(1)
                    volcount.append((dft_rs['Volume'][lenrs]) / freq)
            abl_bg.append(''.join(abl))
            tpo_countbg.append(sum(tpoc))
            volcountbg.append(sum(volcount))
            pricel.append(p)

        dftpo = pd.DataFrame({'close': pricel, 'alphabets': abl_bg,
                              'tpocount': tpo_countbg, 'volsum': volcountbg})
        # drop empty rows
        dftpo['alphabets'].replace('', np.nan, inplace=True)
        dftpo = dftpo.dropna()
        dftpo = dftpo.reset_index(inplace=False, drop=True)
        dftpo = dftpo.sort_index(ascending=False)
        dftpo = dftpo.reset_index(inplace=False, drop=True)

        if style == 'tpo':
            column = 'tpocount'
        else:
            column = 'volsum'

        dfmx = dftpo[dftpo[column] == dftpo[column].max()]

        mid = ll + ((hh - ll) / 2)
        dfmax = dfmx.copy()
        dfmax['poc-mid'] = abs(dfmax['close'] - mid)
        pocidx = dfmax['poc-mid'].idxmin()
        poc = dfmax['close'][pocidx]
        poctpo = dftpo[column].max()
        tpo_updf = dftpo[dftpo['close'] > poc]
        tpo_updf = tpo_updf.sort_index(ascending=False)
        tpo_updf = tpo_updf.reset_index(inplace=False, drop=True)

        tpo_dndf = dftpo[dftpo['close'] < poc]
        tpo_dndf = tpo_dndf.reset_index(inplace=False, drop=True)

        valtpo = (dftpo[column].sum()) * 0.70

        abovepoc = tpo_updf[column].to_list()
        belowpoc = tpo_dndf[column].to_list()


        if (len(abovepoc)/2).is_integer() is False:
            abovepoc = abovepoc+[0]

        if (len(belowpoc)/2).is_integer() is False:
            belowpoc = belowpoc+[0]

        bel2 = np.array(belowpoc).reshape(-1, 2)
        bel3 = bel2.sum(axis=1)
        bel4 = list(bel3)
        abv2 = np.array(abovepoc).reshape(-1, 2)
        abv3 = abv2.sum(axis=1)
        abv4 = list(abv3)
        # cum = poctpo
        # up_i = 0
        # dn_i = 0
        df_va = pd.DataFrame({'abv': pd.Series(abv4), 'bel': pd.Series(bel4)})
        df_va = df_va.fillna(0)
        df_va['abv_idx'] = np.where(df_va.abv > df_va.bel, 1, 0)
        df_va['bel_idx'] = np.where(df_va.bel > df_va.abv, 1, 0)
        df_va['cum_tpo'] = np.where(df_va.abv > df_va.bel, df_va.abv, 0)
        df_va['cum_tpo'] = np.where(df_va.bel > df_va.abv, df_va.bel, df_va.cum_tpo)

        df_va['cum_tpo'] = np.where(df_va.abv == df_va.bel, df_va.abv+df_va.bel, df_va.cum_tpo)
        df_va['abv_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.abv_idx)
        df_va['bel_idx'] = np.where(df_va.abv == df_va.bel, 1, df_va.bel_idx)
        df_va['cum_tpo_cumsum'] = df_va.cum_tpo.cumsum()
        # haven't add poc tpo because loop cuts off way before 70% so it gives same effect
        df_va_cut = df_va[df_va.cum_tpo_cumsum + poctpo <= valtpo]
        vah_idx = (df_va_cut.abv_idx.sum())*2
        val_idx = (df_va_cut.bel_idx.sum())*2

        if vah_idx >= len(tpo_updf) and vah_idx != 0:
            vah_idx = vah_idx - 2

        if val_idx >= len(tpo_dndf) and val_idx != 0:
            val_idx = val_idx - 2

        vah = tpo_updf.close[vah_idx]
        val = tpo_dndf.close[val_idx]


        tpoval = dftpo[ticksize * 2:-(ticksize * 2)]['tpocount']  # take mid section
        exhandle_index = np.where(tpoval <= 2, tpoval.index, None)  # get index where TPOs are 2
        exhandle_index = list(filter(None, exhandle_index))
        distance = ticksize * 3  # distance b/w two ex handles / lvn
        lvn_list = []
        for ex in exhandle_index[0:-1:distance]:
            lvn_list.append(dftpo['close'][ex])

        excess_h = dftpo[0:ticksize]['tpocount'].sum() / ticksize  # take top tail
        excess_l = dftpo[-(ticksize):]['tpocount'].sum() / ticksize  # take lower tail
        excess = 0
        if excess_h == 1 and dftpo.iloc[-1]['close'] < poc:
            excess = dftpo['close'][ticksize]

        if excess_l == 1 and dftpo.iloc[-1]['close'] >= poc:
            excess = dftpo.iloc[-ticksize]['close']


        area_above_poc = dft_rs.High.max() - poc
        area_below_poc = poc - dft_rs.Low.min()
        if area_above_poc == 0:
            area_above_poc = 1
        if area_below_poc == 0:
            area_below_poc = 1
        balance = area_above_poc/area_below_poc

        if balance >= 0:
            bal_target = poc - area_above_poc
        else:
            bal_target = poc + area_below_poc

        mp = {'df': dftpo, 'vah': round(vah, 2), 'poc': round(poc, 2), 'val': round(val, 2), 'lvn': lvn_list, 'excess': round(excess, 2),
              'bal_target': round(bal_target, 2)}

    else:
        print('not enough bars for date {}'.format(dft_rs['datetime'][0]))
        mp = {}

    return mp

# !!! fetch all MP derived results here with date and do extra context analysis


def get_context(df_hi, freq=30, ticksize=5, style='tpo', session_hr=6.5):
    #    df_hi=df.copy()
    try:

        DFcontext = [group[1] for group in df_hi.groupby(df_hi.index.date)]
        dfmp_l = []
        i_poctpo_l = []
        i_tposum = []
        vah_l = []
        poc_l = []
        val_l = []
        bt_l = []
        lvn_l = []
        excess_l = []
        date_l = []
        volume_l = []
        rf_l = []
        ibv_l = []
        ibrf_l = []
        ibh_l = []
        ib_l = []
        close_l = []
        hh_l = []
        ll_l = []
        range_l = []

        for c in range(len(DFcontext)):  # c=1 for testing
            dfc1 = DFcontext[c].copy()
            dfc1.iloc[:, 2:6] = dfc1.iloc[:, 2:6].apply(pd.to_numeric)

            dfc1 = dfc1.reset_index(inplace=False, drop=True)
            mpc = tpo(dfc1, freq, ticksize, style, session_hr)
            dftmp = mpc['df']
            dfmp_l.append(dftmp)
            # for day types
            i_poctpo_l.append(dftmp['tpocount'].max())
            i_tposum.append(dftmp['tpocount'].sum())
            # !!! get value areas
            vah_l.append(mpc['vah'])
            poc_l.append(mpc['poc'])
            val_l.append(mpc['val'])

            bt_l.append(mpc['bal_target'])
            lvn_l.append(mpc['lvn'])
            excess_l.append(mpc['excess'])

            # !!! operatio of non profile stats
            date_l.append(dfc1.datetime[0])
            close_l.append(dfc1.iloc[-1]['Close'])
            ll_l.append(dfc1.High.max())
            hh_l.append(dfc1.Low.min())
            range_l.append(dfc1.High.max() - dfc1.Low.min())

            volume_l.append(dfc1.Volume.sum())
            rf_l.append(dfc1.rf.sum())
            # !!! get IB
            dfc1['cumsumvol'] = dfc1.Volume.cumsum()
            dfc1['cumsumrf'] = dfc1.rf.cumsum()
            dfc1['cumsumhigh'] = dfc1.High.cummax()
            dfc1['cumsummin'] = dfc1.Low.cummin()
            # !!! append ib values
            # 60 min = 1 hr divide by time frame to get number of bars
            ibv_l.append(dfc1.cumsumvol[int(60/freq)])
            ibrf_l.append(dfc1.cumsumrf[int(60/freq)])
            ib_l.append(dfc1.cumsummin[int(60/freq)])
            ibh_l.append(dfc1.cumsumhigh[int(60/freq)])

        # dffin = pd.concat(dfcon_l)
        max_po = max(i_poctpo_l)
        min_po = min(i_poctpo_l)

        dist_df = pd.DataFrame({'date': date_l, 'maxtpo': i_poctpo_l, 'tpocount': i_tposum, 'vahlist': vah_l,
                                'poclist': poc_l, 'vallist': val_l, 'btlist': bt_l, 'lvnlist': lvn_l, 'excesslist': excess_l,
                                'volumed': volume_l, 'rfd': rf_l, 'highd': hh_l, 'lowd': ll_l, 'ranged': range_l, 'ibh': ibh_l,
                                'ibl': ib_l, 'ibvol': ibv_l, 'ibrf': ibrf_l, 'close': close_l})

        dist_df['distr'] = dist_df.tpocount/dist_df.maxtpo
        dismean = math.floor(dist_df.distr.mean())
        dissig = math.floor(dist_df.distr.std())

        dist_df['daytype'] = np.where(np.logical_and(dist_df.distr >= dismean,
                                                     dist_df.distr < dismean + (dissig)), 'Trend Distribution Day', '')

        dist_df['daytype'] = np.where(np.logical_and(dist_df.distr < dismean,
                                                     dist_df.distr >= dismean - (dissig)), 'Normal Variation Day', dist_df['daytype'])

        dist_df['daytype'] = np.where(dist_df.distr < dismean - (dissig),
                                      'Neutral Day', dist_df['daytype'])

        dist_df['daytype'] = np.where(dist_df.distr > dismean + (dissig),
                                      'Trend Day', dist_df['daytype'])
        daytypes = dist_df['daytype'].to_list()

        # !!! get ranking based on distribution data frame aka dist_df
        ranking_df = dist_df.copy()
        ranking_df['vahtrend'] = np.where(ranking_df.vahlist >= ranking_df.vahlist.shift(), 1, -1)
        ranking_df['valtrend'] = np.where(ranking_df.vallist >= ranking_df.vallist.shift(), 1, -1)
        ranking_df['poctrend'] = np.where(ranking_df.poclist >= ranking_df.poclist.shift(), 1, -1)
        ranking_df['hhtrend'] = np.where(ranking_df.highd >= ranking_df.highd.shift(), 1, -1)
        ranking_df['lltrend'] = np.where(ranking_df.lowd >= ranking_df.lowd.shift(), 1, -1)
        ranking_df['closetrend'] = np.where(ranking_df.close >= ranking_df.close.shift(), 1, -1)
        ranking_df['cl_poc'] = np.where(ranking_df.close >= ranking_df.poclist, 1, -1)
        ranking_df['cl_vah'] = np.where(ranking_df.close >= ranking_df.vahlist, 2, 0)  # Max is 2
        ranking_df['cl_val'] = np.where(ranking_df.close <= ranking_df.vallist, -2, 0)  # Min is -2
        # !!! total 9 rankings, even though 2 of them have max score of +2 and -2 their else score set to 0 so wont exceed 100%
        ranking_df['power1'] = 100*((ranking_df.vahtrend + ranking_df.valtrend+ranking_df.poctrend+ranking_df.hhtrend +
                                     ranking_df.lltrend+ranking_df['closetrend']+ranking_df['cl_poc']+ranking_df['cl_vah']+ranking_df['cl_val'])/9)

        a, b = 70, 100
        x, y = ranking_df.power1.min(), ranking_df.power1.max()
        ranking_df['power'] = (ranking_df.power1 - x) / (y - x) * (b - a) + a

    except Exception as e:
        print(str(e))
        ranking_df = []
        dfmp_l = []

    return(dfmp_l, ranking_df)


def get_contextnow(mean_val, ranking):
    ibrankdf = ranking.copy()
    ibvol_mean = mean_val['volib_mean']
    ibrf_mean = mean_val['ibrf_mean']
    rf_mean = mean_val['rf_mean']
    vol_mean = mean_val['volume_mean']

    ibrankdf['ibmid'] = ibrankdf.ibl+((ibrankdf.ibh-ibrankdf.ibl)/2)

    ibrankdf['ib_poc'] = np.where(ibrankdf.ibmid >= ibrankdf.poclist.shift(), 1, -1)
    ibrankdf['ib_vah'] = np.where(ibrankdf.ibmid >= ibrankdf.vahlist.shift(), 2, 0)
    ibrankdf['ib_val'] = np.where(ibrankdf.ibmid <= ibrankdf.vallist.shift(), -2, 0)

    ibrankdf['ibvol_rise'] = ibrankdf.ibvol/ibvol_mean
    ibrankdf['ibvol_rise'] = ibrankdf.ibvol_rise * ibrankdf.ib_poc

    ibrankdf['power1'] = 100*((ibrankdf.ib_poc+ibrankdf.ib_vah +
                               ibrankdf.ib_val+ibrankdf.ibvol_rise)/4)

    #  normalize manually instead of sklearn minmax scaler to avoid dependency

    a, b = 50, 100
    x, y = ibrankdf.power1.min(), ibrankdf.power1.max()
    ibrankdf['power'] = (ibrankdf.power1 - x) / (y - x) * (b - a) + a

    return ibrankdf


def get_rf(df):
    df['cup'] = np.where(df['Close'] >= df['Close'].shift(), 1, -1)
    df['hup'] = np.where(df['High'] >= df['High'].shift(), 1, -1)
    df['lup'] = np.where(df['Low'] >= df['Low'].shift(), 1, -1)

    df['rf'] = df['cup'] + df['hup'] + df['lup']
    df = df.drop(['cup', 'lup', 'hup'], axis=1)
    return df


def get_mean(dfhist, avglen=30, freq=30):
    dfhist = get_rf(dfhist.copy())
    dfhistd = dfhist.resample("D").agg(
        {'symbol': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
         'rf': 'sum', })
    dfhistd = dfhistd.dropna()
    comp_days = len(dfhistd)

    vm30 = dfhistd['Volume'].rolling(avglen).mean()
    volume_mean = vm30[len(vm30) - 1]
    rf30 = (dfhistd['rf']).rolling(avglen).mean()
    rf_mean = rf30[len(rf30) - 1]

    date2 = dfhistd.index[1].date()
    mask = dfhist.index.date < date2
    dfsession = dfhist.loc[mask]
    session_hr = math.ceil(len(dfsession)/60)
    "get IB volume mean"
    ib_start = dfhist.index.time[0]
    ib_end = dfhist.index.time[int(freq*(60/freq))]
    dfib = dfhist.between_time(ib_start, ib_end)
    # dfib = df.head(int(60/freq))
    # dfib['Volume'].plot()
    dfib = dfib.resample("D").agg(
        {'symbol': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
         'rf': 'sum', })
    dfib = dfib.dropna()
    vib = dfib['Volume'].rolling(avglen).mean()
    volib_mean = vib[len(vib) - 1]
    ibrf30 = (dfib['rf']).rolling(avglen).mean()
    ibrf_mean = ibrf30[len(ibrf30) - 1]

    all_val = dict(volume_mean=volume_mean, rf_mean=rf_mean, volib_mean=volib_mean,
                   ibrf_mean=ibrf_mean, session_hr=session_hr)

    return all_val
