import yfinance as yf
import pandas as pd
import os

def get_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/Yahoo")

def get_stock_prices(stocks="SLB HAL BKR WFRD RIG FTI",
                     start_date="1985-01-01",
                     live_read=False,
                     save=True):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)

    if live_read:
        # Pull from Yahoo Finance, progress=False prevents console logging
        # df = yf.download(stocks, start=start_date, group_by="ticker", progress=False)
        df = yf.download(stocks, start=start_date, group_by="ticker")

        # If user passes a single ticker, yfinance can return single-level columns.
        # Normalize to MultiIndex so the same slicing works:
        if not isinstance(df.columns, pd.MultiIndex):
            df = pd.concat({stocks.strip(): df}, axis=1)

        # Save the raw multiindex CSV if you want it around
        raw_path = os.path.join(data_folder, "stock_prices.csv")
        df.to_csv(raw_path)
    else:
        # Read the raw multiindex CSV that was written by df.to_csv above
        raw_path = os.path.join(data_folder, "stock_prices.csv")
        df = pd.read_csv(
            raw_path,
            header=[0, 1],        # reconstruct MultiIndex columns
            index_col=0,          # 'Date' column
            parse_dates=[0]
        )

    # Slice just the Close price across tickers
    close_df = df.xs("Close", axis=1, level=1)

    # df_clean = close_df.copy()
    
    # # Ensure the index is datetime
    # if not isinstance(df_clean.index, pd.DatetimeIndex):
    #     df_clean.index = pd.to_datetime(df_clean.index)
    
    # # Ensure the index is named 'Date'
    # df_clean.index.name = 'Date'
    
    # # Convert all columns to numeric, handling any non-numeric values
    # for col in df_clean.columns:
    #     df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean up names so the display matches your "clean" CSV (no extra 'Ticker' header line)
    # close_df.index.name = "Date"
    # close_df.columns.name = None

    if save:
        clean_path = os.path.join(data_folder, "clean_stock_prices.csv")
        close_df.to_csv(clean_path)

    return close_df

def get_exchange_rates(currencies="USDEUR=X USDGBP=X USDJPY=X", start_date="1985-01-01", save=True):
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)
    data = yf.download(currencies, start=start_date, group_by="ticker")

    if save:
        data.to_csv(os.path.join(data_folder, "exchange_rates.csv"))

    return data

# print(get_closing_prices())
# print(get_stock_prices(live_read=False).head())