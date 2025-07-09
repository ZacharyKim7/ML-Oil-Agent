import requests, pandas as pd, os

URL = "https://rigcount.bakerhughes.com/"     # ← use the real URL
# HEADERS = {"User-Agent": "Mozilla/5.0"}   # polite, avoids some blocks

html = requests.get(URL, timeout=30).text

# Find the specific table by its class to avoid picking up unrelated tables
tables = pd.read_html(html, attrs={"class": "nirtable"})
if not tables:
    raise RuntimeError("No tables matched; check the class name")

df = tables[0]          # only one match on this page

# Optional cleanup ---------------------------------------------------------
# 1. Flatten the <br> tags in the date columns from "03 July<br>2025" → "03 July 2025"
df = df.replace(r"\s*<br>\s*", " ", regex=True)

# 2. Parse the two date columns into proper datetime objects
for col in ["Last Count", "Date of Prior Count", "Date of Last Year's Count"]:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

# Save to CSV

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

out_file = "rig_counts_2025-07-08.csv"
df.to_csv(os.path.join(DATA_DIR, 'rig_counts.csv'), index=False)
print(f"Wrote {len(df)} rows to {out_file}")