import finnhub
import os, dotenv, requests, json
from datetime import date
dotenv.load_dotenv()

API_KEY = os.getenv("FINHUB_API")

finnhub_client = finnhub.Client(api_key=API_KEY)

print(finnhub_client.company_basic_financials('AAPL', 'all'))
