import time
import yfinance as yf
from helpers import LongStrategies, calculateReturns, plotData, setPaths
from parameters import *


def string_clean(text):
    for x in ["-", "=", ".", "^"]:
        if x in text:
            text = text.replace(x, "_")
    else:
        return text


def getPriceData(proxy=None, **kwargs):

    if proxy is not None:
        df = yf.download(tickers=kwargs['asset'], period=kwargs['time_span'], interval=kwargs['time_interval'],
                         auto_adjust=True, progress=False, proxy=proxy).reset_index()

    if proxy is None:
        df = yf.download(tickers=kwargs['asset'], period=kwargs['time_span'], interval=kwargs['time_interval'],
                         auto_adjust=True, progress=False).reset_index()

    if df.empty:
        print("COULD NOT GET: {}".format(kwargs["asset"]))
        pass

    else:
        print("FINISHED DOWNLOADING: {}:{}:{}".format(kwargs["asset"], kwargs['time_span'], kwargs['time_interval']))
    return df


def timed(function):

    def wrapper(*args, **kwargs):
        before = time.time()
        value = function(*args, **kwargs)
        after = time.time()
        func_name = function.__name__
        print(f"\n\n{func_name} TOOK {after-before} SECONDS TO COMPLETE \n")
        return value
    return wrapper()


@timed
def main():
    periods = ["1d", "5d"]
    intervals = ["1m", "5m", "15m", "30m", "60m"]

    for token in token_list:
        print(f"\n\nSTARTING BACKTEST FOR {token} \n")
        for period in periods:
            for interval in intervals:
                df = getPriceData(asset=token, time_span=period, time_interval=interval)
                temp_file_name = f"{string_clean(token)}_{period}_{interval}"

                paths = setPaths(token)
                df.to_csv(paths.get("DATA_PATH")+temp_file_name + ".csv", index=0)
                STRATS = LongStrategies(df, starting_capital=initial_investment, stop_loss=stop_loss, take_profit=take_profit,
                                        mav1=mav1, mav2=mav2, period=bband_period, low=rsi_mfi_low)

                SMA_DF, EMA_DF, DEMA_DF, BB_DF, RSI_DF, MFI_DF = STRATS.RETURN_FRAMES()
                strategy_return_dict = {
                    "SMA_CROSS": SMA_DF,
                    "EMA_CROSS": EMA_DF,
                    "DEMA_CROSS": DEMA_DF,
                    "BB_CROSS": BB_DF,
                    "RSI_CROSS": RSI_DF,
                    "MFI_CROSS": MFI_DF,
                }
                for strategy, frame in strategy_return_dict.items():
                    if not frame.empty:

                        information = calculateReturns(frame, asset=token, initial_capital=initial_investment, strategy=strategy)
                        paths = setPaths(token, strategy)

                        with open(paths.get("LOG_PATH")+temp_file_name + ".txt", "w") as f:
                            f.write("")

                        with open(paths.get("LOG_PATH")+temp_file_name + ".txt", "+a") as f:
                            f.write(information)

                        plotData(df, df_2=frame, file_path=paths.get("CHART_PATH"), asset=token, take_profit=take_profit,
                                 stop_loss=stop_loss, strategy=strategy, time_span=period, time_interval=interval)


if __name__ == "__main__":
    main()
