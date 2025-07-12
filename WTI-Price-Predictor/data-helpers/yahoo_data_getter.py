import yfinance as yf, pandas as pd

# SLB
# HAL
# BKR
# WFRD
# RIG
# FTI

def get_stock_prices(stocks, start_date):
    return yf.download(stocks, start=start_date, group_by="ticker")
    # stock_prices = yf.download("SLB HAL BKR WFRD RIG FTI", start="1985-01-01")
    # print(len(data))

def get_exchange_rates(currencies, start_date):
    return yf.download(currencies, start=start_date, group_by="ticker")

# get_stock_prices("SLB HAL BKR WFRD RIG FTI", "1985-01-01")
exchange_rates = get_exchange_rates("USDEUR=X USDGBP=X USDJPY=X", "2025-01-01")
