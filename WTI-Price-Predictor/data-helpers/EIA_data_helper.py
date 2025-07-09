import os, dotenv, requests, json
from datetime import date
dotenv.load_dotenv()

EIA = os.getenv("EIA_KEY")

def opec_production():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=57&facets[countryRegionId][]=OPEC&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"opec_production_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def non_opec_production():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=57&facets[countryRegionId][]=OPNO&facets[unit][]=TBPD&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"non_opec_production_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data
    
def oecd_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=2&facets[productId][]=54&facets[countryRegionId][]=OECD&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"oecd_consumption_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def china_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=annual&data[0]=value&facets[activityId][]=2&facets[productId][]=5&facets[countryRegionId][]=CHN&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"china_consumption_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def india_consumption():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=annual&data[0]=value&facets[activityId][]=2&facets[productId][]=5&facets[countryRegionId][]=IND&facets[unit][]=TBPD&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"india_consumption_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def oecd_stocks():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=5&facets[productId][]=5&facets[countryRegionId][]=OECD&facets[unit][]=MBBL&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"oecd_stocks_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def usa_stocks():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=5&facets[productId][]=5&facets[countryRegionId][]=USA&facets[unit][]=MBBL&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"usa_stocks_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def usa_from_opec():
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[activityId][]=3&facets[productId][]=787&facets[countryRegionId][]=OPEC&facets[unit][]=TBPD&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"usa_from_opec_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data 

def usa_from_nopec():
    url = f"https://api.eia.gov/v2/petroleum/move/impcus/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=MTTIMUSVV1&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"usa_from_nopec_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data

def WTI_price():
    url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=RWTC&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"WTI_price_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data 

def brent_price():
    url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=RBRTE&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"brent_price_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data 

def US_rig_count():
    url = f"https://api.eia.gov/v2/natural-gas/enr/drill/data/?api_key={EIA}&frequency=monthly&data[0]=value&facets[series][]=E_ERTRR0_XR0_NUS_C&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(url)
    data = response.json()
    
    # Save the JSON data to the data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create filename with timestamp
    today = date.today().strftime("%Y-%m-%d")
    filename = f"US_rig_count_{today}.json"
    filepath = os.path.join(data_folder, filename)
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"EIA data saved to: {filepath}")
    return data 


# opec_production()
# non_opec_production()
# oecd_consumption()
# china_consumption()
# india_consumption()
# oecd_stocks()
# usa_stocks()
# usa_from_opec()
# usa_from_nopec()
# WTI_price()
# brent_price()
US_rig_count()