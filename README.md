# ML-Oil-Agent Overview
A two-stage machine learning pipeline that first predicts the price of WTI oil futures 21 business days out, then attempts to predict the average stock prices of Schlumberger SLB and Haliburton (HAL) across the same 21 business day window. This model was constructed on the hypothesis that oil service companies look for specific stable oil prices before increasing oil service operations. The training and test data ranges from 1986 to present.

## How to run:

Install the dependencies (recomended to use a virtual environment)
```
ML-Oil-Agent$ pip install -r requirements.txt
```

Run the pipeline (note that main should be run as a package, not as a file).
```
ML-Oil-Agent$ python -m main
```

## Data Sources 

* EIA

* FRED

* Yahoo Finance

* World Bank

## Historical Data Collected on Crude Oil

* WTI Price (Daily)

* Brent Price (Daily)

* OECD Consumption (monthly)

* India Consumption (monthly)

* China Consumption (monthly)

* OPEC Production (monthly)

* Non-OPEC Production (Monthly)

* USA import from OPEC (Monthly)

* USA import from Non-OPEC (Monthly)

* USA Stocks (Monthly)

* OECD Stocks (Monthly)

* US Rig Counts (Monthly)

# Key Features