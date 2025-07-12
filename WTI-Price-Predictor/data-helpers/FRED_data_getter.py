import os, dotenv, pandas as pd, requests, json
from datetime import date
dotenv.load_dotenv()

API_KEY = os.getenv("FRED_API")

# https://api.stlouisfed.org/fred/category?category_id=125&api_key=abcdefghijklmnopqrstuvwxyz123456&file_type=json


def get_CPI():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()

    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/FRED')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"CPI_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def get_inflation_rates():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10YIE&api_key={API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()

    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/FRED')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"CPI_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def get_GDP_growth():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=A191RL1Q225SBEA&api_key={API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()

    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/FRED')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"CPI_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

# CPIAUCSL
# T10YIE
# A191RL1Q225SBEA

get_CPI()