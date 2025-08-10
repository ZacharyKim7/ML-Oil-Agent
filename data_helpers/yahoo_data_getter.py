import yfinance as yf
import pandas as pd
import os

def get_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo")

def get_closing_prices(stocks="SLB HAL BKR WFRD RIG FTI", start_date="1985-01-01", live_read=False, save=True):
    df = get_stock_prices(stocks, start_date, live_read, save)
    tickers = [t.strip() for t in "WFRD,FTI,SLB,BKR,RIG,HAL".split(",")]

    new_df = pd.DataFrame(index=df.index)   # dates preserved/aligned

    df_close = df.xs("Close", axis=1, level=1)        # keep only Close across all tickers
    new_df = df_close[tickers]

    new_df = new_df.dropna()

    return new_df

def get_stock_prices(stocks="SLB HAL BKR WFRD RIG FTI", start_date="1985-01-01", live_read=False, save=True):
    data_folder = get_data_folder()
    df = None
    if live_read:
        os.makedirs(data_folder, exist_ok=True)
        df = yf.download(stocks, start=start_date, group_by="ticker")
    else:
        df = pd.read_csv(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo/stock_prices.csv"),
            header=[0, 1],
            index_col=0,           # first column is Date
            parse_dates=True       # actually parse it into datetime
        )

    return df

def get_exchange_rates(currencies="USDEUR=X USDGBP=X USDJPY=X", start_date="1985-01-01", save=True):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    data = yf.download(currencies, start=start_date, group_by="ticker")

    if save:
        data.to_csv(os.path.join(data_folder, "exchange_rates.csv"))

    return data
