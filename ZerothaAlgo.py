from joblib import Parallel, delayed

from kiteconnect import KiteConnect
import pandas as pd
import datetime
import numpy as np
import sys
from math import floor
import os
import time
import ticker
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

dirpath = os.getcwd()
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 20)
print("\nRun Started.......... : ", datetime.datetime.now())
risk_per_trade = 100  # if stoploss gets triggers, you loss will be this, trade quantity will be calculated based on this
tickerlist = ticker.tickerlist
NSELTPformate = ['NSE:{}'.format(i) for i in tickerlist]
nums_buy = 5  # number of stock to buy
nums_sell = 5  # number of stock to sell
target = 2  # as %
stoploss = 1  # as %
trailing_stoploss = 0.5  # as %
filterstock_lowpricelimit = 100
filterstock_highpricelimit = 2000
supertrend_period = 30
supertrend_multiplier = 2
orderslist = [0, 0]
# keep token of stocks that you want to skip
candlesize = '5minute'


def login():
    try:
        kites = KiteConnect(api_key="nx5fvm3iodrq4ix5")
        data = kites.generate_session("kDWQ1n26zi4qhH8XtGz4sftLMR0M4feB", api_secret="s7jpa4hrrpievt66oxqivewy25pc7qf2")
        kites.set_access_token(data["access_token"])
    except Exception as e:
        print(" ERROR in api_key", e, datetime.datetime.now())
    print("user data loaded..........", datetime.datetime.now())
    return kites


def calculateEMA(df, column, period):
    df[column] = df['close'].ewm(span=period).mean()
    return df


def calculateVwap(df):
    df['dates'] = df['date'].apply(lambda a: pd.to_datetime(a).date())
    day = datetime.datetime.now().date()
    df = df[df['dates'] >= day]
    tp = (df['high'] + df['low'] + df['close']) / 3
    vp = tp * df['volume']
    df['vp'] = vp.cumsum()
    df['cvp'] = df['volume'].cumsum()
    df['VWAP'] = df['vp'] / df['cvp']
    return df


def gethistoricaldata(token):
    enddate = datetime.datetime.today()
    startdate = enddate - datetime.timedelta(10)
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    try:
        data = kites.historical_data(token, startdate, enddate, interval=candlesize)
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None)
        # print(df)
        if not df.empty:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = df['date'].astype(str).str[:-6]
            df['date'] = pd.to_datetime(df['date'])
            df = SuperTrend(df)

    except Exception as e:
        print("         error in gethistoricaldata", token, e)
    return df


def SuperTrend(df, period=supertrend_period, multiplier=supertrend_multiplier, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute SuperTrend

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """
    calculateEMA(df, 'EMA21', 21)
    calculateEMA(df, 'EMA9', 9)

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST'  # + str(period) + '_' + str(multiplier)
    stx = 'STX'  # + str(period) + '_' + str(multiplier)

    """
    SuperTrend Algorithm :

        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """
    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]
    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
            df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
            df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)
    df = calculateVwap(df)

    # Remove basic and final bands from the columns
    # df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    return df


def ATR(df, period, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute Average True Range (ATR)

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=False)
    return df


def EMA(df, base, target, period, alpha):
    """
    Function to compute Exponential Moving Average (EMA)

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    df[target].fillna(0, inplace=True)
    return df


def run_trategy(i):
    try:
        pd.options.mode.chained_assignment = None

        if combinetoken[i] in shared_list:
            print("Already in orderlist")
        else:

            your_custom_conditions = 'None'
            histdata = gethistoricaldata(combinetoken[i])
            super_trend_value = histdata.ST.values[-1]
            vwap = histdata.VWAP.values[-1]
            EMA21 = histdata.EMA21.values[-1]
            EMA9 = histdata.EMA9.values[-1]
            lastclose = histdata.close.values[-1]
            if (combinetoken[i] in tickerstobuy_token):
                if super_trend_value < lastclose and EMA21 < lastclose and EMA9 < lastclose and vwap < lastclose:
                    your_custom_conditions = 'BUY'
            if (combinetoken[i] in tickerstosell_token):
                if super_trend_value > lastclose and EMA21 > lastclose and EMA9 > lastclose and vwap > lastclose:
                    your_custom_conditions = 'SELL'  # apply your strategy

            if your_custom_conditions == 'BUY':
                shared_list.append(combinetoken[i])
                price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
                stoploss_buy = lastclose * 0.01
                stoploss_buy = int(100 * (floor(stoploss_buy / 0.05) * 0.05)) / 100
                quantity = floor(max(1, (risk_per_trade / stoploss_buy)))
                target = lastclose * 0.02
                quantity = int(quantity)
                target = int(100 * (floor(target / 0.05) * 0.05)) / 100
                try:
                    print("\n\nBUY Order",
                          "\ntradingsymbol", combineStock[i],
                          "\nquantity:", quantity,
                          "\nbuyprice:", price,
                          "\nbuy_target :", target,
                          "\nbuy_stoploss:", stoploss_buy, " Time : ", datetime.datetime.now())

                    orderid_b = kites.place_order(variety=kites.VARIETY_REGULAR,
                                                  tradingsymbol=combineStock[i],
                                                  quantity=int(1),
                                                  exchange=kites.EXCHANGE_NSE,
                                                  order_type=kites.ORDER_TYPE_MARKET,
                                                  product=kites.PRODUCT_MIS,
                                                  transaction_type=kites.TRANSACTION_TYPE_BUY,
                                                  validity=kites.VALIDITY_DAY)

                    print("BUY Order is placed : orderid ", orderid_b)


                except Exception as e:
                    print("BUY ORDER FAILED : RESPONSE FROM ZERODHA : ", e)

            if your_custom_conditions == 'SELL':
                shared_list.append(combinetoken[i])

                price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
                stoploss_buy = lastclose * 0.01  # 0.01
                stoploss_buy = int(100 * (floor(stoploss_buy / 0.05) * 0.05)) / 100
                quantity = floor(max(1, (risk_per_trade / stoploss_buy)))
                target = lastclose * 0.02
                quantity = int(quantity)
                target = int(100 * (floor(target / 0.05) * 0.05)) / 100

                try:
                    print("\n\nSELL Order",
                          "\ntradingsymbol", combineStock[i],
                          "\nquantity:", quantity,
                          "\nsell price:", price,
                          "\nsell_target :", target,
                          "\nsell_stoploss:", stoploss_buy, " Time : ", datetime.datetime.now())
                    orderid_b = kites.place_order(variety=kites.VARIETY_REGULAR,
                                              tradingsymbol=combineStock[i],
                                              quantity=int(1),
                                              exchange=kites.EXCHANGE_NSE,
                                              order_type=kites.ORDER_TYPE_MARKET,
                                              product=kites.PRODUCT_MIS,
                                              transaction_type=kites.TRANSACTION_TYPE_SELL,
                                              validity=kites.VALIDITY_DAY)

                    print("BUY Order is placed : orderid ", orderid_b)
                except Exception as e:
                    print("SELL ORDER FAILED : RESPONSE FROM ZERODHA : ", e)
    except Exception as e:
        print("Exceptions: RESPONSE FROM ZERODHA : ", e)


if __name__ == '__main__':
    global kites
    kites = login()
    runcount = 0
    global OHLCdf_buy
    global OHLCdf_sell
    global combinetoken
    global combineStock
    global tickerstosell_token
    global tickerstobuy_token
    global shared_list
    day = int(0)
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    start_time = int(9) * 60 + int(20)  # specify in int (hr) and int (min) foramte
    end_time = int(15) * 60 + int(10)  # do not place fresh order
    stop_time = int(15) * 60 + int(15)  # square off all open positions
    algoruntime = int(9) * 60 + int(10)
    print("pd.options.mode.chained_assignment :", pd.options.mode.chained_assignment)

    while True:
        try:
            timenow = (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute)
            if timenow >= algoruntime and day == 0:
                OHLCdict = kites.ohlc(NSELTPformate)
                OHLCdf = pd.DataFrame(
                    columns=['tradingsymbol', 'instrument_token', 'last_price', 'open', 'high', 'low', 'close',
                             'change',
                             'pchange'])

                for key, value in OHLCdict.items():
                    try:
                        c1 = key.split(":")[1]
                        c2 = value['instrument_token']
                        c3 = value['last_price']
                        value2 = value['ohlc']
                        c4 = value2['open']
                        c5 = value2['high']
                        c6 = value2['low']
                        c7 = value2['close']
                        # print(c1,c2,c3,c4,c5,c6,c7)
                        if c3 > filterstock_lowpricelimit and c3 < filterstock_highpricelimit:
                            OHLCdf.loc[len(OHLCdf)] = [c1, c2, c3, c4, c5, c6, c7, c3 - c7, 100 * (c3 - c7) / c7]
                    except Exception as e:
                        print(e)
                OHLCdf = OHLCdf.sort_values(by=['pchange'], ascending=False)
                OHLCdf_buy = OHLCdf.head(nums_buy)
                OHLCdf_sell = OHLCdf.tail(nums_sell)
                OHLCdf_buy = OHLCdf_buy.drop(OHLCdf_buy[OHLCdf_buy.pchange <= 0.5].index)
                OHLCdf_sell = OHLCdf_sell.drop(OHLCdf_sell[OHLCdf_sell.pchange >= -0.5].index)
                print("\n\nTOP gainer\n", OHLCdf_buy)
                print("\n\nTop loser\n", OHLCdf_sell)
                day = 1
        except Exception as e:
            print("ERROR in RUN ", e)

        if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= end_time:
            print(end_time)
            if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= stop_time:
                print(sys._getframe().f_lineno, "Trading day closed, time is above stop_time")
                day == 0
                shared_list *= 0
                break
        elif (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= start_time:
            if runcount >= 0:
                print("\n\n {} Run Count : Time - {} ".format(runcount, datetime.datetime.now()))
                tickerstobuy = OHLCdf_buy.tradingsymbol.values
                tickerstobuy_ltp = OHLCdf_buy.last_price.values
                tickerstobuy_token = OHLCdf_buy.instrument_token.values
                tickerstosell = OHLCdf_sell.tradingsymbol.values
                tickerstosell_ltp = OHLCdf_sell.last_price.values
                tickerstosell_token = OHLCdf_sell.instrument_token.values
                arr1 = np.array(tickerstobuy)
                arr2 = np.array(tickerstosell)
                combineStock = np.concatenate((arr1, arr2))
                arr3 = np.array(tickerstobuy_token)
                arr4 = np.array(tickerstosell_token)
                combinetoken = np.concatenate((arr3, arr4))
                inputs = range(0, len(combineStock))
                num_cores = multiprocessing.cpu_count()
                print("number of CPU cores " + str(num_cores))
                results = Parallel(n_jobs=num_cores)(delayed(run_trategy)(i) for i in inputs)
                runcount = runcount + 1
                time.sleep(300)

        else:
            print(' Waiting... ', datetime.datetime.now())
