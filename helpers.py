import ta
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")


class LongStrategies(object):
    def __init__(self, df, starting_capital, stop_loss, take_profit,
                 mav1, mav2, period, low):

        self.SMA_TDF = self.MA(df, initial_capital=starting_capital,
                               take_profit=take_profit, stop_loss=stop_loss,
                               ma_type="SMA", mav1=mav1, mav2=mav2)

        self.EMA_TDF = self.MA(df, initial_capital=starting_capital,
                               take_profit=take_profit, stop_loss=stop_loss,
                               ma_type="EMA", mav1=mav1, mav2=mav2)

        self.DEMA_TDF = self.MA(df, initial_capital=starting_capital,
                                take_profit=take_profit, stop_loss=stop_loss,
                                ma_type="DEMA", mav1=mav1, mav2=mav2)

        self.BB_TDF = self.BBANDS(df, initial_capital=starting_capital,
                                  take_profit=take_profit, stop_loss=stop_loss,
                                  period=period)

        self.RSI_TDF = self.RSI(df,  initial_capital=starting_capital,
                                take_profit=take_profit, stop_loss=stop_loss,
                                period=period, Low=low)

        self.MFI_TDF = self.MFI(df,  initial_capital=starting_capital,
                                take_profit=take_profit, stop_loss=stop_loss,
                                period=period, Low=low)

    def RETURN_FRAMES(self):
        return self.SMA_TDF, self.EMA_TDF, self.DEMA_TDF, self.BB_TDF, self.RSI_TDF, self.MFI_TDF

    def MA(self, df, *args, **kwargs):
        buy_signal = []
        sell_signal = []

        cum_sum = []
        profits = []

        profit_dates = []
        percent_change = []

        position = False

        IC = kwargs['initial_capital']
        TP = kwargs['take_profit']
        SL = kwargs['stop_loss']
        take_profit_percent = (float(TP) / 100) + 1
        stop_loss_percent = 1 - (float(SL) / 100)

        if kwargs["ma_type"] == "SMA":
            df[f"SMA{kwargs['mav1']}"] = df.Close.rolling(kwargs['mav1']).mean()
            df[f"SMA{kwargs['mav2']}"] = df.Close.rolling(kwargs['mav2']).mean()
            if not position:
                for i in range(0, len(df.index)):
                    close = df.Close[i]
                    date = df.iloc[:, 0][i]
                    if df[f"SMA{kwargs['mav1']}"][i] > df[f"SMA{kwargs['mav2']}"][i] and position is False:
                        buy_price = close
                        qty_shares = IC / buy_price
                        buy_value = buy_price * qty_shares
                        position = True

                    if position:
                        if df.Close[i] > take_profit_percent * buy_price \
                                or df.Close[i] < stop_loss_percent * buy_price and position is True:

                            sell_price = close
                            sell_value = (sell_price * qty_shares)
                            profit = round(float(sell_value - buy_value), 2)
                            IC = IC + profit
                            change = (sell_price / buy_price - 1) * 100

                            buy_signal.append(buy_price)
                            sell_signal.append(sell_price)
                            profits.append(profit)
                            profit_dates.append(date)
                            cum_sum.append(IC)
                            percent_change.append(change)
                            position = False

            df_trade_final = pd.DataFrame({"Date": profit_dates,
                                           "Value": cum_sum,
                                           "Profit": profits,
                                           "Change": percent_change,
                                           "Buy Price": buy_signal,
                                           "Sell Price": sell_signal})

            return df_trade_final

        elif kwargs["ma_type"] == "EMA":
            df[f"EMA{kwargs['mav1']}"] = df.Close.ewm(span=kwargs['mav1'], adjust=False).mean()
            df[f"EMA{kwargs['mav2']}"] = df.Close.ewm(span=kwargs['mav2'], adjust=False).mean()

            if not position:
                for i in range(0, len(df.index)):
                    close = df.Close[i]
                    date = df.iloc[:, 0][i]
                    if df[f"EMA{kwargs['mav1']}"][i] > df[f"EMA{kwargs['mav2']}"][i] and position is False:
                        buy_price = close
                        qty_shares = IC / buy_price
                        buy_value = buy_price * qty_shares
                        position = True

                    if position:
                        if df.Close[i] > take_profit_percent * buy_price \
                                or df.Close[i] < stop_loss_percent * buy_price and position is True:

                            sell_price = close
                            sell_value = (sell_price * qty_shares)
                            profit = round(float(sell_value - buy_value), 2)
                            IC = IC + profit
                            change = (sell_price / buy_price - 1) * 100

                            buy_signal.append(buy_price)
                            sell_signal.append(sell_price)
                            profits.append(profit)
                            profit_dates.append(date)
                            cum_sum.append(IC)
                            percent_change.append(change)
                            position = False

            df_trade_final = pd.DataFrame({"Date": profit_dates,
                                           "Value": cum_sum,
                                           "Profit": profits,
                                           "Change": percent_change,
                                           "Buy Price": buy_signal,
                                           "Sell Price": sell_signal})

            return df_trade_final

        elif kwargs["ma_type"] == "DEMA":
            EMA_1 = df.Close.ewm(span=kwargs['mav1'], adjust=False).mean()
            EMA_2 = df.Close.ewm(span=kwargs['mav2'], adjust=False).mean()

            df[f"DEMA{str(kwargs['mav1'])}"] = (2 * EMA_1 - EMA_1.ewm(span=kwargs['mav1'], adjust=False).mean())
            df[f"DEMA{str(kwargs['mav2'])}"] = (2 * EMA_2 - EMA_2.ewm(span=kwargs['mav2'], adjust=False).mean())

            if not position:
                for i in range(0, len(df.index)):
                    close = df.Close[i]
                    date = df.iloc[:, 0][i]
                    if df[f"DEMA{kwargs['mav1']}"][i] > df[f"DEMA{kwargs['mav2']}"][i] and position is False:
                        buy_price = close
                        qty_shares = IC / buy_price
                        buy_value = buy_price * qty_shares
                        position = True

                    if position:
                        if df.Close[i] > take_profit_percent * buy_price \
                                or df.Close[i] < stop_loss_percent * buy_price and position is True:
                            sell_price = close
                            sell_value = (sell_price * qty_shares)
                            profit = round(float(sell_value - buy_value), 2)
                            IC = IC + profit
                            change = (sell_price / buy_price - 1) * 100

                            buy_signal.append(buy_price)
                            sell_signal.append(sell_price)
                            profits.append(profit)
                            profit_dates.append(date)
                            cum_sum.append(IC)
                            percent_change.append(change)
                            position = False

            df_trade_final = pd.DataFrame({"Date": profit_dates,
                                           "Value": cum_sum,
                                           "Profit": profits,
                                           "Change": percent_change,
                                           "Buy Price": buy_signal,
                                           "Sell Price": sell_signal})

            return df_trade_final

    def BBANDS(self, df, *args, **kwargs):
        buy_signal = []
        sell_signal = []
        cum_sum = []
        profits = []
        profit_dates = []
        percent_change = []
        position = False

        IC = kwargs['initial_capital']
        TP = kwargs['take_profit']
        SL = kwargs['stop_loss']
        take_profit_percent = (float(TP) / 100) + 1
        stop_loss_percent = 1 - (float(SL) / 100)

        STD = df['Close'].rolling(window=kwargs['period']).std()
        EMA = df['Close'].ewm(span=kwargs['period'], adjust=False).mean()
        df['Lower'] = EMA - (STD * 2)
        df['Lower'] = df["Lower"].replace(np.nan, 0)

        if not position:
            for i in range(0, len(df.index)):
                close = df.Close[i]
                date = df.iloc[:, 0][i]

                if df.Close[i] < df["Lower"][i] and position is False:
                    buy_price = close
                    qty_shares = IC / buy_price
                    buy_value = buy_price * qty_shares
                    position = True

                if position:
                    if df.Close[i] > take_profit_percent * buy_price \
                            or df.Close[i] < stop_loss_percent * buy_price and position is True:
                        sell_price = close
                        sell_value = (sell_price * qty_shares)
                        profit = round(float(sell_value - buy_value), 2)
                        IC = IC + profit
                        change = (sell_price / buy_price - 1) * 100

                        buy_signal.append(buy_price)
                        sell_signal.append(sell_price)
                        profits.append(profit)
                        profit_dates.append(date)
                        cum_sum.append(IC)
                        percent_change.append(change)
                        position = False

        df_trade_final = pd.DataFrame({"Date": profit_dates,
                                       "Value": cum_sum,
                                       "Profit": profits,
                                       "Change": percent_change,
                                       "Buy Price": buy_signal,
                                       "Sell Price": sell_signal})

        return df_trade_final

    def RSI(self, df, *args, **kwargs):
        buy_signal = []
        sell_signal = []
        cum_sum = []
        profits = []
        profit_dates = []
        percent_change = []
        position = False

        IC = kwargs['initial_capital']
        TP = kwargs['take_profit']
        SL = kwargs['stop_loss']
        take_profit_percent = (float(TP) / 100) + 1
        stop_loss_percent = 1 - (float(SL) / 100)

        rsi = ta.momentum.rsi(df.Close, window=kwargs['period'], fillna=True)
        df['RSI'] = rsi

        if not position:
            for i in range(0, len(df["RSI"])):
                close = df.Close[i]
                date = df.iloc[:, 0][i]
                if df["RSI"][i] < kwargs["Low"] and position is False:
                    buy_price = close
                    qty_shares = IC / buy_price
                    buy_value = buy_price * qty_shares
                    position = True

                if position:
                    if df.Close[i] > take_profit_percent * buy_price \
                            or df.Close[i] < stop_loss_percent * buy_price and position is True:
                        sell_price = close
                        sell_value = (sell_price * qty_shares)
                        profit = round(float(sell_value - buy_value), 2)
                        IC = IC + profit
                        change = (sell_price / buy_price - 1) * 100

                        buy_signal.append(buy_price)
                        sell_signal.append(sell_price)
                        profits.append(profit)
                        profit_dates.append(date)
                        cum_sum.append(IC)
                        percent_change.append(change)
                        position = False

        df_trade_final = pd.DataFrame({"Date": profit_dates,
                                       "Value": cum_sum,
                                       "Profit": profits,
                                       "Change": percent_change,
                                       "Buy Price": buy_signal,
                                       "Sell Price": sell_signal})

        return df_trade_final

    def MFI(self, df, *args, **kwargs):
        buy_signal = []
        sell_signal = []
        cum_sum = []
        profits = []
        profit_dates = []
        percent_change = []
        position = False

        IC = kwargs['initial_capital']
        TP = kwargs['take_profit']
        SL = kwargs['stop_loss']
        take_profit_percent = (float(TP) / 100) + 1
        stop_loss_percent = 1 - (float(SL) / 100)

        df["MFI"] = ta.volume.money_flow_index(df.High, df.Low, df.Close, df.Volume,
                                               window=kwargs["period"], fillna=True)

        if not position:
            for i in range(0, len(df["MFI"])):
                close = df.Close[i]
                date = df.iloc[:, 0][i]
                if df["MFI"][i] < kwargs["Low"] and position is False:
                    buy_price = close
                    qty_shares = IC / buy_price
                    buy_value = buy_price * qty_shares
                    position = True

                if position:
                    if df.Close[i] > take_profit_percent * buy_price \
                            or df.Close[i] < stop_loss_percent * buy_price and position is True:

                        sell_price = close
                        sell_value = (sell_price * qty_shares)
                        profit = round(float(sell_value - buy_value), 2)
                        IC = IC + profit
                        change = (sell_price / buy_price - 1) * 100

                        buy_signal.append(buy_price)
                        sell_signal.append(sell_price)
                        profits.append(profit)
                        profit_dates.append(date)
                        cum_sum.append(IC)
                        percent_change.append(change)
                        position = False

        df_trade_final = pd.DataFrame({"Date": profit_dates,
                                       "Value": cum_sum,
                                       "Profit": profits,
                                       "Change": percent_change,
                                       "Buy Price": buy_signal,
                                       "Sell Price": sell_signal})
        return df_trade_final

    def MA_ADV_CROSS(self, df, *args, **kwargs):  # TODO 4 MOVING AVERAGE CROSS
        buy_signal = []
        sell_signal = []

        cum_sum = []
        profits = []

        profit_dates = []
        percent_change = []

        position = False
        flag_long = False
        flag_short = False

        IC = kwargs['initial_capital']
        TP = kwargs['take_profit']
        SL = kwargs['stop_loss']
        take_profit_percent = (float(TP) / 100) + 1
        stop_loss_percent = 1 - (float(SL) / 100)

        if kwargs["ma_type"] == "EMA" and kwargs["ma_type_2"] == "DEMA":

            df[f"EMA{kwargs['ema_mav1']}"] = df.Close.rolling(kwargs['ema_mav1']).mean()
            df[f"EMA{kwargs['ema_mav2']}"] = df.Close.rolling(kwargs['ema_mav2']).mean()

            df[f"DEMA{kwargs['dema_mav2']}"] = df.Close.ewm(span=kwargs['dema_mav2'], adjust=False).mean()
            df[f"DEMA{kwargs['dema_mav2']}"] = df.Close.ewm(span=kwargs['dema_mav2'], adjust=False).mean()

            if not position:
                for i in range(0, len(df.index)):
                    close = df.Close[i]
                    date = df.iloc[:, 0][i]

                    EMA_MA_1 = df[f"EMA{kwargs['ema_mav1']}"][i]
                    EMA_MA_2 = df[f"EMA{kwargs['ema_mav2']}"][i]

                    DEMA_MA_1 = df[f"DEMA{kwargs['dema_mav1']}"][i]
                    DEMA_MA_2 = df[f"DEMA{kwargs['dema_mav2']}"][i]

                    # Long Conditions
                    if DEMA_MA_1 and EMA_MA_1 > DEMA_MA_2 and EMA_MA_2:
                        flag_long = True
                        flag_short = False

                    # Short Conditions
                    elif DEMA_MA_2 and EMA_MA_2 > DEMA_MA_1 and EMA_MA_1:
                        flag_long = False
                        flag_short = True

                    # Long Crossover Position
                    elif DEMA_MA_1 > DEMA_MA_2 and EMA_MA_1 > EMA_MA_2 \
                        and flag_long == True \
                            and flag_short == False \
                            and position == False:

                        buy_price = close
                        qty_shares = IC / buy_price
                        buy_value = buy_price * qty_shares

                        position = True
                        flag_long = False
                        flag_short = False

                    if position:
                        if close > take_profit_percent * buy_price \
                                or close < stop_loss_percent * buy_price:

                            sell_price = close
                            sell_value = (sell_price * qty_shares)
                            profit = round(float(sell_value - buy_value), 2)
                            IC = IC + profit
                            change = (sell_price / buy_price - 1) * 100

                            buy_signal.append(buy_price)
                            sell_signal.append(sell_price)
                            profits.append(profit)
                            profit_dates.append(date)
                            cum_sum.append(IC)
                            percent_change.append(change)
                            position = False

                    # Short Crossover position
                    elif DEMA_MA_2 > DEMA_MA_1 and EMA_MA_2 > EMA_MA_1 \
                        and flag_long == False \
                            and flag_short == True \
                            and position == False:

                        buy_price = close
                        qty_shares = IC / buy_price
                        buy_value = buy_price * qty_shares

                        position = True
                        flag_long = False
                        flag_short = False

                    if position:
                        if close > take_profit_percent * buy_price \
                                or close < stop_loss_percent * buy_price:

                            sell_price = close
                            sell_value = (sell_price * qty_shares)
                            profit = round(float(buy_value - sell_value), 2)
                            IC = IC + profit
                            change = (sell_price / buy_price - 1) * 100

                            buy_signal.append(buy_price)
                            sell_signal.append(sell_price)
                            profits.append(profit)
                            profit_dates.append(date)
                            cum_sum.append(IC)
                            percent_change.append(change)
                            position = False

            df_trade_final = pd.DataFrame({"Date": profit_dates,
                                           "Value": cum_sum,
                                           "Profit": profits,
                                           "Change": percent_change,
                                           "Buy Price": buy_signal,
                                           "Sell Price": sell_signal})

            return df_trade_final #


def parseTrades(df, logs, **kwargs):
    buy_signal = []
    sell_signal = []

    cum_sum = []
    profits = []

    profit_dates = []
    percent_change = []

    position = False
    flag_long = False
    flag_short = False

    IC = kwargs['initial_capital']
    TP = kwargs['take_profit']
    SL = kwargs['stop_loss']

    close = df.Close
    date = df.iloc[:, 0]

    buy_price = close
    qty_shares = IC / buy_price
    buy_value = buy_price * qty_shares
    position = True

    sell_price = close
    sell_value = (sell_price * qty_shares)
    profit = round(float(sell_value - buy_value), 2)
    IC = IC + profit
    change = (sell_price / buy_price - 1) * 100

    buy_signal.append(buy_price)
    sell_signal.append(sell_price)
    profits.append(profit)
    profit_dates.append(date)
    cum_sum.append(IC)
    percent_change.append(change)
    position = False

    df_trade_final = pd.DataFrame({"Date": profit_dates,
                                   "Value": cum_sum,
                                   "Profit": profits,
                                   "Change": percent_change,
                                   "Buy Price": buy_signal,
                                   "Sell Price": sell_signal})

    return df_trade_final


def calculateReturns(df, **kwargs):
    gains = 0
    number_gain = 0
    losses = 0
    number_loss = 0
    total_return = 1

    for i in df["Change"]:

        if i > 0:
            gains += i
            number_gain += 1

        else:
            losses += i
            number_loss += 1

        total_return = total_return * ((i / 100) + 1)

    total_return = round((total_return - 1) * 100, 2)

    if number_gain > 0:
        avg_gain = gains / number_gain
        max_return = str(max(df["Change"]))

    else:
        avg_gain = 0
        max_return = "undefined"

    if number_loss > 0:
        avg_loss = losses / number_loss
        max_loss = str(min(df["Change"]))
        ratio = str(-avg_gain / avg_loss)

    else:
        avg_loss = 0
        max_loss = "undefined"
        ratio = "inf"

    if number_gain > 0 or number_loss > 0:
        batting_avg = number_gain / (number_gain + number_loss)

    else:
        batting_avg = 0

    fees = len(df['Profit']) * (0.025 * kwargs['initial_capital'] / 100)
    BACKTEST_DATE   = f"BACKTEST_DATE:   {str(pd.to_datetime(time.time(), unit='s'))}"
    STRATEGY        = f"STRATEGY:        {kwargs['strategy']}"
    ASSET           = f"ASSET:           {kwargs['asset']}"
    DATE_RANGE      = f"DATE_RANGE:      {str(df.iloc[:, 0].iloc[0])} - {str(df.iloc[:, 0].iloc[-1])}"
    SAMPLE_SIZE     = f"SAMPLE_SIZE:     {number_gain + number_loss} TRADES"
    BATTING_AVG     = f"BATTING_AVG:     {batting_avg}"
    GAIN_LOSS_RATIO = f"GAIN_LOSS_RATIO: {ratio}"
    AVERAGE_GAIN    = f"AVERAGE_GAIN:    {avg_gain}"
    AVERAGE_LOSS    = f"AVERAGE_LOSS:    {avg_loss}"
    MAX_RETURN      = f"MAX_RETURN:      {max_return}"
    MAX_LOSS        = f"MAX_LOSS:        {max_loss}"
    TOTAL_RETURN    = f"TOTAL_RETURN:    {total_return}%"

    FEES            = f"FEES: (0.025%)    ${fees:,.2f}"
    INITIAL_CAPITAL = f"INITIAL_CAPITAL: ${kwargs['initial_capital']:,.2f}"
    FINAL_CAPITAL   = f"FINAL_CAPITAL:   ${kwargs['initial_capital'] + df['Profit'].sum() + (-fees):,.2f}"
    TOTAL_PNL       = f"TOTAL_PNL:       ${kwargs['initial_capital'] + df['Profit'].sum() + (-fees) - kwargs['initial_capital']:,.2f} \n"

    TRADING_PROFIT  = f"TRADING_PROFIT:  ${round(df['Profit'].sum(), 2):,.2f} "
    BEST_TRADE      = f"BEST_TRADE:      ${df.max()['Profit']:,.2f} : {df.max()['Date']}"
    WORST_TRADE     = f"WORST_TRADE:     ${df.min()['Profit']:,.2f} : {df.min()['Date']}"

    details = (BACKTEST_DATE, STRATEGY, ASSET, DATE_RANGE, SAMPLE_SIZE, BATTING_AVG, GAIN_LOSS_RATIO,
               AVERAGE_GAIN, AVERAGE_LOSS, MAX_RETURN, MAX_LOSS, TOTAL_RETURN, FEES, INITIAL_CAPITAL,
               FINAL_CAPITAL, TRADING_PROFIT, BEST_TRADE, WORST_TRADE)

    info = (
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"    
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
        "{}\n"
    ).format(*details)

    return info


def plotData(df, df_2, *args, **kwargs):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df.iloc[:, 0], df.Close, color="red")

    # ax1.plot(df.iloc[:, 0], df[f"EMA{kwargs['mav1']}"], color="orange", alpha=0.35, label=f"EMA{kwargs['mav1']}")
    # ax1.plot(df.iloc[:, 0], df[f"EMA{kwargs['mav2']}"], color="blue", alpha=0.35, label=f"EMA{kwargs['mav2']}")
    ax1.scatter(df_2["Date"], df_2["Buy Price"], color="green", marker="^", alpha=1)
    ax1.scatter(df_2["Date"], df_2["Sell Price"], color="red", marker="v", alpha=1)
    line_labels = [kwargs["asset"], f"TP: {kwargs['take_profit']}%", f"SL: {kwargs['stop_loss']}%"]
    ax1.legend(labels=line_labels, loc="upper left", borderaxespad=0.1)
    ax2.plot(df_2["Date"], df_2["Value"])

    # ax2.bar(df_trade_final["Date"], df_trade_final["Value"], linewidth=0,  width=2/len(df_trade_final["Value"]))
    line_labels2 = ["Portfolio Value"]
    ax2.legend(labels=line_labels2, loc="upper left", borderaxespad=0.1)

    plt.tight_layout()
    fig.suptitle(f"{kwargs['asset']} Close Price | {str(df.iloc[:, 0].iloc[0])} - {str(df.iloc[:, 0].iloc[-1])}|")
    plt.savefig(f"{kwargs['file_path']}{kwargs['asset']}_{kwargs['strategy']}_{kwargs['time_span']}_{kwargs['time_interval']}.png")
    plt.close()

def setPaths(token, strategy=None):
    ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
    BO_PATH = os.path.join(ROOT_PATH, "BacktestOutput/")
    if not os.path.exists(BO_PATH):
        os.mkdir(BO_PATH)
        print("MKDIR: {}".format(BO_PATH))

    ASSET_OUT_PATH = os.path.join(BO_PATH, f"{token}/")
    if not os.path.exists(ASSET_OUT_PATH):
        os.mkdir(ASSET_OUT_PATH)
        print("MKDIR: {}".format(ASSET_OUT_PATH))

    DATA_PATH = os.path.join(ASSET_OUT_PATH, f"Data/")
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        print("MKDIR: {}".format(DATA_PATH))

    if strategy is not None:

        STRAT_OUT_PATH = os.path.join(ASSET_OUT_PATH, f"{strategy}/")
        if not os.path.exists(STRAT_OUT_PATH):
            os.mkdir(STRAT_OUT_PATH)
            print("MKDIR: {}".format(STRAT_OUT_PATH))

        LOG_PATH = os.path.join(STRAT_OUT_PATH, f"Logs/")
        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
            print("MKDIR: {}".format(LOG_PATH))

        CHART_PATH = os.path.join(STRAT_OUT_PATH, f"Charts/")
        if not os.path.exists(CHART_PATH):
            os.mkdir(CHART_PATH)
            print("MKDIR: {}".format(CHART_PATH))

        path_params = {
            "ROOT": ROOT_PATH,
            "BACKTEST_PATH": BO_PATH,
            "ASSET_PATH": ASSET_OUT_PATH,
            "DATA_PATH": DATA_PATH,
            "STRATEGY_PATH": STRAT_OUT_PATH,
            "LOG_PATH": LOG_PATH,
            "CHART_PATH": CHART_PATH}

        return path_params

    if strategy is None:

        path_params = {
            "ROOT": ROOT_PATH,
            "BACKTEST_PATH": BO_PATH,
            "ASSET_PATH": ASSET_OUT_PATH,
            "DATA_PATH": DATA_PATH}

        return path_params


def network_helper():
    #  Use this if you start to get rate limited by yahoo finance.
    #  Import it as a function in emacross.py and pass it as the proxy parameter in the getPriceData function in main()

    proxy_list = ["191.81.48.144:999", "186.13.56.97:8080", "172.105.184.208:8001",
                  "93.84.70.211:3128", "186.248.89.6:5005", "190.26.201.194:8080",
                  "181.129.2.90:8081", "177.93.48.165:999", "181.233.49.14:999",
                  "181.233.49.146:999" "186.5.94.203:999", "157.100.12.138:999"]

    random_proxy = f"{random.choice(proxy_list)}"
    return random_proxy

