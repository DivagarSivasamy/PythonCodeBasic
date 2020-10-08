from kiteconnect import KiteConnect
import datetime
import pandas as pd
import time
import numpy as np

import ticker

tickerlist = ticker.tickerlist
pd.options.mode.chained_assignment = None


def ATR(df, period, ohlc=['open', 'high', 'low', 'close']):
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

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


def login():
    kite = KiteConnect(api_key="nx5fvm3iodrq4ix5")
    data = kite.generate_session("FhXfeRsN2tWnt5iYv0uM6ImJKOmm1lZI", api_secret="s7jpa4hrrpievt66oxqivewy25pc7qf2")
    kite.set_access_token(data["access_token"])
    return kite


def downloading():
    df = pd.read_csv("https://api.kite.trade/instruments")
    writer = pd.ExcelWriter("C:\\Users\\Administrator\\Documents\\testig\\"+ 'bdadad.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()
    df = df[df['segment'] == "NFO-FUT"]
    df = df[~df['tradingsymbol'].str.startswith("{}".format("NIFTYIT"))]
    df = df[df['tradingsymbol'].str.startswith("{}".format("NIFTY"))]
    df['expiry'] = pd.to_datetime(df['expiry'])
    tday = pd.Timestamp.today() - pd.DateOffset(0)
    df = df[~(df['expiry'] <= tday)]
    df = df[df.expiry == df.expiry.min()]
    print("Instrument tokens Downloaded..........", datetime.datetime.now())
    return df


candlesize = '15minute'


def gethistoricaldata(token):
    enddate = datetime.datetime.today()
    startdate = enddate - datetime.timedelta(10)
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    try:
        data = kite.historical_data(token, startdate, enddate, interval=candlesize)
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None)
        if not df.empty:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = df['date'].astype(str).str[:-6]
            df['date'] = pd.to_datetime(df['date'])
            df = SuperTrend(df)

    except Exception as e:
        print("         error in gethistoricaldata", token, e)
    return df


supertrend_period = 30
supertrend_multiplier = 2


def SuperTrend(df, period=supertrend_period, multiplier=supertrend_multiplier, ohlc=['open', 'high', 'low', 'close']):
    calculateEMA(df, 'EMA21', 21)
    calculateEMA(df, 'EMA9', 9)

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST'  # + str(period) + '_' + str(multiplier)
    stx = 'STX'  # + str(period) + '_' + str(multiplier)
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


def calculateEMA(df, column, period):
    df[column] = df['close'].ewm(span=period).mean()
    return df


def plcaeFutureNiftyOrder(next_print, buyTradeCount, sellTradeCount, orderTradedCondition):
    if not orderTradedCondition:

        timenow = (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute)

        if timenow >= closetime:
            print("Market Closed.........", datetime.datetime.now())
        try:
            histdata = gethistoricaldata(token_nifty_future)
            your_custom_conditions = 'None'
            super_trend_value = histdata.ST.values[-1]
            vwap = histdata.VWAP.values[-1]
            EMA21 = histdata.EMA21.values[-1]
            EMA9 = histdata.EMA9.values[-1]
            lastclose = histdata.close.values[-1]
            open = histdata.open.values[-1]
            if EMA9>vwap and lastclose > super_trend_value and  open < EMA9 and  lastclose > EMA9 and buyTradeCount <= 2:
                your_custom_conditions = kite.TRANSACTION_TYPE_BUY
                buyTradeCount = buyTradeCount + 1
            elif EMA9<vwap and lastclose < super_trend_value and  open > EMA9 and  lastclose < EMA9 and sellTradeCount <= 2:
                your_custom_conditions = kite.TRANSACTION_TYPE_SELL
                sellTradeCount = sellTradeCount + 1
            if your_custom_conditions == kite.TRANSACTION_TYPE_BUY or your_custom_conditions == kite.TRANSACTION_TYPE_SELL:
                try:
                    ##orderTradedCondition = True
                    orderid_b = kite.place_order(exchange=kite.EXCHANGE_NFO,
                                                 tradingsymbol=tradingsymbol_nifty_future,
                                                 transaction_type=your_custom_conditions,
                                                 quantity=int(75),
                                                 product=kite.PRODUCT_MIS,
                                                 order_type=kite.ORDER_TYPE_MARKET,
                                                 validity=kite.VALIDITY_DAY,
                                                 variety=kite.VARIETY_REGULAR
                                                 )

                    print("BUY Order is placed : orderid ", orderid_b)
                except Exception as es:
                    print("BUY ORDER FAILED : RESPONSE FROM ZERODHA : ", es)
                    orderTradedCondition = False
        except Exception as e:
            print(e)
    return orderTradedCondition, buyTradeCount, sellTradeCount


def sellNiftyOrders():
    df = pd.DataFrame(kite.positions()["day"])[
        ["tradingsymbol", "exchange", "average_price", "buy_m2m", "buy_price", "buy_quantity", "buy_value",
         "close_price", "sell_m2m", "sell_price", "sell_quantity", "sell_value", "unrealised", "value",
         "day_buy_price",
         "day_buy_quantity", "day_buy_value", "day_sell_price", "product", "quantity", "realised", "last_price"]]
    for i in range(len(df)):
        tradingsymbol = df.loc[i, "tradingsymbol"]
        exchange = df.loc[i, "exchange"]
        buy_price = df.loc[i, "buy_price"]
        last_price = df.loc[i, "last_price"]
        sell_price = df.loc[i, "sell_price"]
        quantity = df.loc[i, "quantity"]
        if quantity > 0 and exchange == 'NFO':
            Buytarget = buy_price + 40
            SLorBuy = buy_price - 20
            if last_price >= Buytarget or last_price <= SLorBuy:
                try:
                    orderTradedCondition = True
                    orderid_b = kite.place_order(exchange=kite.EXCHANGE_NFO,
                                                 tradingsymbol=tradingsymbol,
                                                 transaction_type=kite.TRANSACTION_TYPE_SELL,
                                                 quantity=quantity,
                                                 product=kite.PRODUCT_MIS,
                                                 order_type=kite.ORDER_TYPE_MARKET,
                                                 validity=kite.VALIDITY_DAY,
                                                 variety=kite.VARIETY_REGULAR
                                                 )

                    print("Sell Order is placed : orderid ", orderid_b)
                except Exception as e:
                    print(e)
        elif quantity < 0 and exchange == 'NFO':
            selltarget = sell_price - 40
            SLorBuy = sell_price + 20
            if last_price <= selltarget or last_price >= SLorBuy:
                try:
                    orderTradedCondition = True
                    orderid_b = kite.place_order(exchange=kite.EXCHANGE_NFO,
                                                 tradingsymbol=tradingsymbol,
                                                 transaction_type=kite.TRANSACTION_TYPE_BUY,
                                                 quantity=quantity,
                                                 product=kite.PRODUCT_MIS,
                                                 order_type=kite.ORDER_TYPE_MARKET,
                                                 validity=kite.VALIDITY_DAY,
                                                 variety=kite.VARIETY_REGULAR
                                                 )

                    print("Sell Order is placed : orderid ", orderid_b)
                except Exception as e:
                    print(e)


print_interval = datetime.timedelta(minutes=15)
global sellTradeCount
global buyTradeCount
buyTradeCount = 0
sellTradeCount = 0

if __name__ == '__main__':
    global kite
    global data
    global closetime
    global token_nifty_future
    global tradingsymbol_nifty_future
    global orderTradedCondition
    global next_print
    kite = login()
    orderopentime = int(9) * 60 + int(30)
    closetime = int(14) * 60 + int(30)
    orderTradedCondition = False
    next_print = datetime.datetime.now()
    df = downloading()
    token_nifty_future = df.instrument_token.values[0]
    tradingsymbol_nifty_future = df.tradingsymbol.values[0]
    print("\ntoken_nifty_future", token_nifty_future)
    print("tradingsymbol_nifty_future", tradingsymbol_nifty_future)
    while True:
        if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= orderopentime:
            now = datetime.datetime.now()
            if now >= next_print:
                print(now)
                next_print = now + print_interval
                orderTradedCondition, buyTradeCount, sellTradeCount = plcaeFutureNiftyOrder(next_print, buyTradeCount,
                                                                                           sellTradeCount,
                                                                                           orderTradedCondition)
                print(next_print)
            # if orderTradedCondition:
            #   sellNiftyOrders()
        else:
            print(' Waiting... ', datetime.datetime.now())

