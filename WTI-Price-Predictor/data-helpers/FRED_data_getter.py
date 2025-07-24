import os, dotenv, requests, json
from datetime import date
dotenv.load_dotenv()

API_KEY = os.getenv("FRED_API")

def save_fred_data(series_id: str, filename: str):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()

    # Create target folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/FRED')
    os.makedirs(data_folder, exist_ok=True)

    # Use static filename (no date)
    filepath = os.path.join(data_folder, f"{filename}.json")

    # Always overwrite the file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"FRED data saved to: {filepath}")
    return data

def get_CPI():
    return save_fred_data("CPIAUCSL", "CPI")

def get_inflation_rates():
    return save_fred_data("T10YIE", "inflation_rates")

def get_GDP_growth():
    return save_fred_data("A191RL1Q225SBEA", "GDP_growth")

# starts from 1999, only include if needed.
def get_USD_EUD():
    return save_fred_data("DEXUSEU", "USD-EUD")

def get_USD_GBP():
    return save_fred_data("DEXUSUK", "USD-GBP")

def get_USD_YEN():
    return save_fred_data("DEXJPUS", "USD-YEN")

# get_CPI()
# get_inflation_rates()
# get_GDP_growth()
# get_USD_EUD()
# get_USD_GBP()
# get_USD_YEN()
