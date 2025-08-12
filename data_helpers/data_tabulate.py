from . import EIA_data_getter as EIA
from . import FRED_data_getter as FRED
from . import WB_data_getter as WB
from . import yahoo_data_getter as YF
import pandas as pd, os, json
from pandas.tseries.offsets import MonthEnd, QuarterEnd

"""
This file contains the data pipeline that converts and adds the API response data
into a single Pandas data frame, df. The functions also include a "live_read" parameter
that can switch between pulling the data and processing it directly in memory, or if 
the pipeline stores the data as a JSON first and then reads from the JSONs.
"""


def get_EIA_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/EIA')

def get_WB_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/WB')

def get_FRED_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/FRED')

def get_Yahoo_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/Yahoo')

"""
Begin EIA data integration
Dataframe is built around WTI spot prices
"""
def tabulate_WTI_price(live_read=False):
    data = None
    if live_read:
        data = EIA.WTI_price()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"WTI_price.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    df = pd.DataFrame([
    {
        "Date": item["period"],
        "WTI ($/bbl)": float(item["value"])
    }
    for item in data["response"]["data"]
    ])

    # Set 'Date' as the index and convert it to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df

def add_Brent_to_df(df, live_read=False):
    data = None
    if live_read:
        data = EIA.brent_price()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"brent_price.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    new_df = pd.DataFrame([
    {
        "Date": item["period"],
        "Brent ($/bbl)": float(item["value"])  # different column name
    }
    for item in data["response"]["data"]
    ])
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    new_df.set_index("Date", inplace=True)

    # Merge and forward-fill missing USD-GBP values
    merged = df.merge(new_df, how='left', left_index=True, right_index=True)
    merged["Brent ($/bbl)"] = merged["Brent ($/bbl)"].ffill()

    return merged

def add_OPEC_production_to_df(df, live_read=False, lag_days=30):
    """
    Adds a lagged monthly cumulative time series to a daily-indexed DataFrame.
    """
    data = None
    if live_read:
        data = EIA.opec_production()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"opec_production.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "OPEC P (tbbl/d)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",  # turn "2025-03" into "2025-03-01"
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_NOPEC_production_to_df(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.non_opec_production()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"non_opec_production.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "NOPEC P (tbbl/d)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",  # turn "2025-03" into "2025-03-01"
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_OECD_Consumption_to_df(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.oecd_consumption()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"oecd_consumption.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "OECD C (tbbl/d)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_China_Consumption_to_df(df, live_read=False, lag_days=365):
    data = None
    """
    Normally, the live_read function would allow for a live read.
    However, China's data is annually with a delay > 365 days. 
    The current data in the JSON is hard coded from recent reports,
    and will be valid for the next few years. Thus, live_read has been disabled.
    """
    # if live_read:
    #     data = EIA.china_consumption()
    # else:
    data_folder = get_EIA_data_folder()
    filepath = os.path.join(data_folder, f"china_consumption.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    column_name = "China C (tbbl/d)"
    annual_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    annual_df["Date"] = pd.to_datetime(annual_df["Date"])
    annual_df.set_index("Date", inplace=True)

    # Fill entire year with same value, and shift by lag_days
    daily_entries = []
    for year_start, row in annual_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        year_end = year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        days = pd.date_range(start=year_start, end=year_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_India_Consumption_to_df(df, live_read=False, lag_days=365):
    data = None
    """
    Normally, the live_read function would allow for a live read.
    However, India's data is annually with a delay > 365 days. 
    The current data in the JSON is hard coded from recent reports,
    and will be valid for the next few years. Thus, live_read has been disabled.
    """
    # if live_read:
    #     data = EIA.china_consumption()
    # else:
    data_folder = get_EIA_data_folder()
    filepath = os.path.join(data_folder, f"india_consumption.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    column_name = "India C (tbbl/d)"
    annual_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    annual_df["Date"] = pd.to_datetime(annual_df["Date"])
    annual_df.set_index("Date", inplace=True)

    # Fill entire year with same value, and shift by lag_days
    daily_entries = []
    for year_start, row in annual_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        year_end = year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        days = pd.date_range(start=year_start, end=year_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_OECD_stocks_to_df(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.oecd_stocks()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"oecd_stocks.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "OECD S (mbbl)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_USA_stocks_to_df(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.usa_stocks()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"usa_stocks.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "USA S (mbbl)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_USA_from_OPEC(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.usa_from_opec()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"usa_from_opec.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "USA <- OPEC (tbbl/d)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_USA_from_NOPEC(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.usa_from_nopec()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"usa_from_nopec.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "USA <- NOPEC (tbbl/m)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else ""
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_USA_net_imports(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.usa_net_imports()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"usa_net_imports.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "US net imports (tbbl/d)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else ""
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_USA_rig_count(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = EIA.usa_rig_count()
    else:
        data_folder = get_EIA_data_folder()
        filepath = os.path.join(data_folder, f"US_rig_count.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "USA rigs (m)"
    monthly_df = pd.DataFrame([
        {
            "Date": item["period"] + "-01",
            column_name: float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["response"]["data"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

"""
Begin FRED data integration
"""
def add_CPI_to_df(df, live_read=False, lag_days=30):
    data = None
    if live_read:
        data = FRED.get_CPI()
    else:
        data_folder = get_FRED_data_folder()
        filepath = os.path.join(data_folder, f"CPI.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "CPI"
    monthly_df = pd.DataFrame([
        {
            "Date": obs["date"],
            column_name: float(obs["value"]) if obs["value"] not in ('', '.', None) else None
        }
        for obs in data["observations"]
    ])
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df.set_index("Date", inplace=True)

    # Fill entire month with same value
    daily_entries = []
    for month_start, row in monthly_df.iterrows():
        if pd.isna(row[column_name]):
            continue
        month_end = month_start + MonthEnd(0)
        days = pd.date_range(start=month_start, end=month_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),  # Apply 30-day lag
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how='left', left_index=True, right_index=True)

def add_GDP_growth_to_df(df, live_read=False, lag_quarters=1):
    data = None
    if live_read:
        data = FRED.get_GDP_growth()
    else:
        data_folder = get_FRED_data_folder()
        filepath = os.path.join(data_folder, "GDP_growth.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    column_name = "GDP (yoy%)"

    # Load GDP data
    if live_read:
        data = FRED.get_GDP_growth()
    else:
        data_folder = get_FRED_data_folder()
        filepath = os.path.join(data_folder, "GDP_growth.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    # Parse and clean
    observations = data["observations"]
    gdp_rows = []
    for obs in observations:
        val = obs["value"]
        if val not in ("", ".", None):
            gdp_rows.append({
                "QuarterStart": pd.to_datetime(obs["date"]),
                column_name: float(val)
            })

    quarterly_df = pd.DataFrame(gdp_rows)
    quarterly_df.set_index("QuarterStart", inplace=True)

    # Generate daily values for lagged quarter
    daily_entries = []
    for quarter_start, row in quarterly_df.iterrows():
        gdp_value = row[column_name]

        # Define original quarter range
        quarter_end = quarter_start + QuarterEnd(0)

        # Shift entire quarter forward by `lag_quarters`
        lagged_start = quarter_start + pd.DateOffset(months=3 * lag_quarters)
        lagged_end = quarter_end + pd.DateOffset(months=3 * lag_quarters)

        # Fill business days only (like your df)
        bdays = pd.bdate_range(start=lagged_start, end=lagged_end)
        for bday in bdays:
            daily_entries.append({
                "Date": bday,
                column_name: gdp_value
            })

    # Build lagged daily GDP DataFrame
    daily_df = pd.DataFrame(daily_entries)
    daily_df.drop_duplicates(subset="Date", keep="last", inplace=True)
    daily_df.set_index("Date", inplace=True)

    # Align and merge
    daily_df = daily_df.reindex(df.index)  # Now safe

    # Merge and forward-fill missing GDP values
    merged = df.merge(daily_df, how='left', left_index=True, right_index=True)
    merged[column_name] = merged[column_name].ffill()

    return merged

def add_USD_GBP(df, live_read=False):
    data = None
    if live_read:
        data = FRED.get_USD_GBP()
    else:
        data_folder = get_FRED_data_folder()
        filepath = os.path.join(data_folder, f"USD-GBP.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    # Convert JSON observations into a DataFrame
    new_df = pd.DataFrame([
        {
            "Date": item["date"],
            "USD-GBP": float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["observations"]
    ])
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    new_df.set_index("Date", inplace=True)

    # Merge and forward-fill missing USD-GBP values
    merged = df.merge(new_df, how='left', left_index=True, right_index=True)
    merged["USD-GBP"] = merged["USD-GBP"].ffill()

    return merged

def add_USD_YEN(df, live_read=False):
    data = None
    if live_read:
        data = FRED.get_USD_YEN()
    else:
        data_folder = get_FRED_data_folder()
        filepath = os.path.join(data_folder, f"USD-YEN.json")
        with open(filepath, 'r') as f:
            data = json.load(f)

    # Convert JSON observations into a DataFrame
    new_df = pd.DataFrame([
        {
            "Date": item["date"],
            "USD-YEN": float(item["value"]) if item["value"] not in ('', '.', None) else None
        }
        for item in data["observations"]
    ])
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    new_df.set_index("Date", inplace=True)

    # Merge and forward-fill missing USD-YEN values
    merged = df.merge(new_df, how='left', left_index=True, right_index=True)
    merged["USD-YEN"] = merged["USD-YEN"].ffill()

    return merged

def fix_USD_YEN_type():
    # Get the CSV path
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/combined_oil_df.csv")

    # Load the DataFrame
    df = pd.read_csv(csv_path)

    # Convert USD-YEN to float64
    df["USD-YEN"] = pd.to_numeric(df["USD-YEN"], errors="coerce")

    # (Optional) Confirm the dtype is now float64
    print(df["USD-YEN"].dtype)  # Should print 'float64'

    # Write the updated DataFrame back to the same CSV
    df.to_csv(csv_path, index=False)

"""
Begin World Bank data integration
"""
def add_world_population(df, live_read=False, lag_days=365):
    population = WB.parse_population_data(live_read)

    population["Year"] = pd.to_datetime(population["Year"])
    population.set_index("Year", inplace=True)

    column_name = "Population"
    # Fill entire year with same value, and shift by lag_days
    daily_entries = []
    for year_start, row in population.iterrows():
        if pd.isna(row[column_name]):
            continue
        year_end = year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        days = pd.date_range(start=year_start, end=year_end)
        for day in days:
            daily_entries.append({
                "Date": day + pd.Timedelta(days=lag_days),
                column_name: row[column_name]
            })

    daily_df = pd.DataFrame(daily_entries)
    daily_df.set_index("Date", inplace=True)

    # Trim to match master DataFrame's date range
    daily_df = daily_df.reindex(df.index)

    # Merge with the master DataFrame
    return df.merge(daily_df, how="left", left_index=True, right_index=True)

# Controls whether API data is stored as JSON and read, or pulled and processed directly in memory
handle_in_memory = False
def get_combined_oil_df(save=True):
    master = (
        tabulate_WTI_price(handle_in_memory)
        .pipe(add_Brent_to_df, live_read=handle_in_memory)
        .pipe(add_OPEC_production_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_NOPEC_production_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_OECD_Consumption_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_China_Consumption_to_df, live_read=handle_in_memory, lag_days=365)
        .pipe(add_India_Consumption_to_df, live_read=handle_in_memory, lag_days=365)
        .pipe(add_OECD_stocks_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_USA_stocks_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_USA_net_imports, live_read=True, lag_days=30)
        .pipe(add_USA_rig_count, live_read=handle_in_memory, lag_days=30)
        .pipe(add_CPI_to_df, live_read=handle_in_memory, lag_days=30)
        .pipe(add_GDP_growth_to_df, live_read=handle_in_memory, lag_quarters=1)
        .pipe(add_USD_GBP, live_read=handle_in_memory)
        .pipe(add_USD_YEN, live_read=handle_in_memory)
        .pipe(add_world_population, live_read=handle_in_memory, lag_days=365)

        # OPEC and non-OPEC oil imports have been excluded in favor of USA net imports.
        # .pipe(add_USA_from_OPEC, live_read=handle_in_memory, lag_days=30)
        # .pipe(add_USA_from_NOPEC, live_read=handle_in_memory, lag_days=30)
    )

    if save:
        # Save the DataFrame to data/combined_oil_df.csv, overwriting if it exists
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        output_path = os.path.join(output_dir, 'combined_oil_df.csv')
        master.to_csv(output_path)
        print("Successfully saved combined oil data to data/combined_oil_df.csv")

    return master

"""
Function to get data stored in the combined_oil_df.csv rather than from the API.
For use in development and testing.
"""
def get_data_from_csv():
    return pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/combined_oil_df.csv"))

"""
Gets Yahoo stock data and appends it to a master data frame
Used for secondary training of predictive models for stock prices.
"""
def add_stocks_to_df(df):
    stocks = YF.get_stock_prices(stocks="SLB HAL", start_date="1985-01-01", live_read=False, save=True)
    master = df.copy()
    master = df.set_index("Date")

    # Trim to match master DataFrame's date range
    stocks = stocks.reindex(master.index)

    # Merge with the master DataFrame
    return master.merge(stocks, how='left', left_index=True, right_index=True)
