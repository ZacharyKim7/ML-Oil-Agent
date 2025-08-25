"""
This model performs a 2-stage transfer learning protocol. To avoid removing a large number of rows,
training is split into two stages to maximize training data, when available.

    TICKERS: primary traning targets, preferably with a long stock history
    EXTRA_TICKERS: secondary targets with a shorter stock history.

"""
TICKERS = ["SLB", "HAL"]
EXTRA_TICKERS = ["BKR", "FTI", "NOV", "VAL", "CLB", "NINE", "PUMP", "RES", "LBRT"]

"""
If READ_LIVE is True, the training pipeline pulls the current data from the API (recommended during production).
IF READ_LIVE is False, the training pipeline pulls data saved in the CSVs to train (recommended during development).
"""
