import os
import dotenv
import requests
import json

dotenv.load_dotenv()
EIA = os.getenv("EIA_KEY")

def get_data_folder():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/EIA")

def fetch_and_store_json(url: str, filename: str) -> dict:
    all_data = []
    offset = 0
    limit = 5000

    while True:
        paged_url = f"{url}&offset={offset}&length={limit}"
        response = requests.get(paged_url)
        response.raise_for_status()
        page = response.json()

        if "response" not in page or "data" not in page["response"]:
            break

        data_batch = page["response"]["data"]
        if not data_batch:
            break

        all_data.extend(data_batch)
        print(f"Fetched {len(data_batch)} records (offset: {offset})")

        if len(data_batch) < limit:
            break

        offset += limit

    # Replace the original 'data' field with the full result
    result = {"response": {"data": all_data}}

    # Ensure data directory exists
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)

    # Save to a fixed file (overwrites each time)
    filepath = os.path.join(data_folder, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"EIA data saved to: {filepath}")
    return result

def brent_price():
    url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA}&frequency=daily&data[0]=value&facets[series][]=RBRTE&sort[0][column]=period&sort[0][direction]=desc"
    return fetch_and_store_json(url, "brent_price")

def WTI_price():
    url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA}&frequency=daily&data[0]=value&facets[series][]=RWTC&sort[0][column]=period&sort[0][direction]=desc"
    return fetch_and_store_json(url, "WTI_price")

def opec_production():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=57&facets[countryRegionId][]=OPEC&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "opec_production")

def non_opec_production():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=57&facets[countryRegionId][]=OPNO&facets[unit][]=TBPD&offset=0&length=5000"
    return fetch_and_store_json(url, "non_opec_production")

def oecd_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=2&facets[productId][]=54&facets[countryRegionId][]=OECD&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "oecd_consumption")

def china_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=annual&data[0]=value&facets[activityId][]=2&facets[productId][]=5&facets[countryRegionId][]=CHN&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "china_consumption")

def india_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=annual&data[0]=value&facets[activityId][]=2&facets[productId][]=5&facets[countryRegionId][]=IND&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "india_consumption")

def oecd_stocks():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=5&facets[productId][]=5&facets[countryRegionId][]=OECD&facets[unit][]=MBBL&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "oecd_stocks")

def usa_stocks():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=5&facets[productId][]=5&facets[countryRegionId][]=USA&facets[unit][]=MBBL&offset=0&length=5000"
    return fetch_and_store_json(url, "usa_stocks")

def usa_from_opec():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=3&facets[productId][]=787&facets[countryRegionId][]=OPEC&facets[unit][]=TBPD&offset=0&length=5000"
    return fetch_and_store_json(url, "usa_from_opec")

def usa_from_nopec():
    url = f"https://api.eia.gov/v2/petroleum/move/impcus/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=MTTIMUSVV1&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "usa_from_nopec")

def usa_rig_count():
    url = f"https://api.eia.gov/v2/natural-gas/enr/drill/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=E_ERTRR0_XR0_NUS_C&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    return fetch_and_store_json(url, "US_rig_count")
