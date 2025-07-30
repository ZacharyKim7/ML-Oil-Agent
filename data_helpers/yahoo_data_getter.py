import yfinance as yf
import pandas as pd
import os

# Ensure the folder exists

def get_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo")

def get_stock_prices(stocks="SLB HAL BKR WFRD RIG FTI", start_date="1985-01-01", live_read=False, save=True):
    data_folder = get_data_folder()

    if live_read:
        os.makedirs(data_folder, exist_ok=True)
        df = yf.download(stocks, start=start_date, group_by="ticker")
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo/exchange_rates.csv"), header=[0, 1], index_col=2)

    # Select only the 'Close' values for each ticker
    df_close = df.xs('Close', level=1, axis=1)

    # Optional: Convert index to datetime
    df_close.index = pd.to_datetime(df_close.index)

    # Check result
    # print(df_close.head())

    return df_close

def get_exchange_rates(currencies="USDEUR=X USDGBP=X USDJPY=X", start_date="1985-01-01", save=True):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    data = yf.download(currencies, start=start_date, group_by="ticker")

    if save:
        data.to_csv(os.path.join(data_folder, "exchange_rates.csv"))

    return data

# Example usage
# stock_prices = get_stock_prices("HAL BKR WFRD RIG FTI", "1985-01-01", live_read=False, save=False)
# print(stock_prices)
# exchange_rates = get_exchange_rates("USDEUR=X USDGBP=X USDJPY=X", "1985-01-01")
# print(exchange_rates)
