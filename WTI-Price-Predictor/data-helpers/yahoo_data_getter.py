import yfinance as yf
import pandas as pd
import os

# Ensure the folder exists
os.makedirs("data/Yahoo", exist_ok=True)

def get_stock_prices(stocks="SLB HAL BKR WFRD RIG FTI", start_date="1985-01-01"):
    data = yf.download(stocks, start=start_date, group_by="ticker")
    data.to_csv("data/Yahoo/stock_prices.csv")
    return data

def get_exchange_rates(currencies="USDEUR=X USDGBP=X USDJPY=X", start_date="1985-01-01"):
    data = yf.download(currencies, start=start_date, group_by="ticker")
    data.to_csv("data/Yahoo/exchange_rates.csv")
    return data

# Example usage
# stock_prices = get_stock_prices("SLB HAL BKR WFRD RIG FTI", "1985-01-01")
exchange_rates = get_exchange_rates("USDEUR=X USDGBP=X USDJPY=X", "2025-01-01")
