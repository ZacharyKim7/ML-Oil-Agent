import os, requests, zipfile, io, pandas as pd

def get_population_data():
    url = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"

    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/WB")
    os.makedirs(data_folder, exist_ok=True)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")

    # Extract ZIP in-memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the actual population CSV file
        csv_filename = next((f for f in z.namelist() if f.startswith("API_SP.POP.TOTL") and f.endswith(".csv")), None)
        if not csv_filename:
            raise Exception("Population CSV not found in ZIP file.")

        # Read the CSV file directly into pandas
        with z.open(csv_filename) as csvfile:
            df = pd.read_csv(csvfile, skiprows=4)  # Skip the World Bank header rows

    # Save to your target directory
    df.to_csv(os.path.join(data_folder, "world_population.csv"), index=False)
    print(f"Saved population data to: {data_folder}")

    return df

    # combined_df.to_csv(os.path.join(DATA_DIR, 'combined_eia_data.csv'), index=False)
    # DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# add individual population into cumulative total for each year
def parse_population_data(live_read=False):
    # Read the population data
    df = None
    if live_read:
        df = get_population_data()
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/WB/world_population.csv"))
    
    # Get the year columns (excluding metadata columns)
    year_columns = [col for col in df.columns if col.isdigit()]
    
    # Calculate total population for each year by summing across all countries
    yearly_totals = {}
    for year in year_columns:
        # Sum all non-null population values for this year
        total = df[year].sum()
        yearly_totals[str(year) + "-01-01"] = total
    
    # Create new dataframe with year and total population
    total_population_df = pd.DataFrame({
        'Year': list(yearly_totals.keys()),
        'Population': list(yearly_totals.values())
    })
    
    # Sort by year
    total_population_df = total_population_df.sort_values('Year')
    
    # Save to CSV only if pulling new data.
    if live_read:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/WB/world_total_population.csv")
        total_population_df.to_csv(output_path, index=False)
    # print(f"Saved total population data to: {output_path}")
    # print(f"Data shape: {total_population_df.shape}")
    # print(f"Year range: {total_population_df['Year'].min()} - {total_population_df['Year'].max()}")
    
    return total_population_df

# get_population_data()
# parse_population_data()