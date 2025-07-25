import yfinance as yf
import pandas as pd
import os

# Ensure the folder exists

def get_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo")

def get_stock_prices(stocks="SLB HAL BKR WFRD RIG FTI", start_date="1985-01-01"):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    data = yf.download(stocks, start=start_date, group_by="ticker")
    data.to_csv(os.path.join(data_folder, "stock_prices.csv"))
    return data

def get_exchange_rates(currencies="USDEUR=X USDGBP=X USDJPY=X", start_date="1985-01-01"):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    data = yf.download(currencies, start=start_date, group_by="ticker")
    data.to_csv(os.path.join(data_folder, "exchange_rates.csv"))
    return data

# Example usage
# stock_prices = get_stock_prices("SLB HAL BKR WFRD RIG FTI", "1985-01-01")
# exchange_rates = get_exchange_rates("USDEUR=X USDGBP=X USDJPY=X", "1985-01-01")
# print(exchange_rates)
