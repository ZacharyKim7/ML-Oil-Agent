import os
import json
import pandas as pd

# Path to the data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# List all JSON files in the data directory
json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]

all_rows = []

for filename in json_files:
    category = filename.split('_')[0]  # e.g., 'oecd_stocks' from 'oecd_stocks_2025-07-06.json'
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
        # Try to find the data list
        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
        elif 'data' in data:
            records = data['data']
        else:
            continue  # skip files with unexpected structure
        for entry in records:
            row = {
                'period': entry.get('period'),
                'productName': entry.get('productName'),
                'activityName': entry.get('activityName'),
                'countryRegionName': entry.get('countryRegionName'),
                'unitName': entry.get('unitName'),
                'value': entry.get('value'),
                'unit': entry.get('unit'),
                'category': category
            }
            all_rows.append(row)

# Create DataFrame
combined_df = pd.DataFrame(all_rows)

# Save to CSV for convenience
combined_df.to_csv(os.path.join(DATA_DIR, 'combined_eia_data.csv'), index=False)

print(f"Combined DataFrame shape: {combined_df.shape}")
print(combined_df.head()) 