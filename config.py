"""
This model performs a 2-stage transfer learning protocol. To avoid removing a large number of rows,
training is split into two stages to maximize training data, when available.

    TICKERS: primary traning targets, preferably with a long stock history
    EXTRA_TICKERS: secondary targets with a shorter stock history.

    NEW_DATA: determines if new data is pulled from the APIs (True), or read from data saved in the CSVs (False)

    SAVE_MODEL: True saves the trained model as a .pk1 file with the serialized model parameters. Saves in models/.

    TRAIN_NEW_MODEL: True trains a new model, False reads the saved model parameters, if available.
"""

TICKERS = ["SLB", "HAL"]
EXTRA_TICKERS = ["BKR", "FTI", "NOV", "VAL", "CLB", "NINE", "PUMP", "RES", "LBRT"]
NEW_DATA = True
SAVE_MODEL = True
TRAIN_NEW_MODEL = False
