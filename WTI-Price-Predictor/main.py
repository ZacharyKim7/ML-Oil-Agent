from data_helpers import data_tabulate as data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_random_forest_forecast_model(df, target_column='WTI ($/bbl)', horizon=21, test_ratio=0.2, scale_features=False, plot_importance=True):
    """
    Trains a RandomForest model to predict the target_column `horizon` days into the future.

    Parameters:
        df (pd.DataFrame): Daily-indexed data with exogenous features.
        target_column (str): Name of the column to forecast.
        horizon (int): Days to forecast ahead.
        test_ratio (float): Fraction of data to reserve for testing.
        scale_features (bool): Whether to apply StandardScaler.
        plot_importance (bool): Whether to show feature importance plot.

    Returns:
        model: Trained RandomForestRegressor
        predictions: np.ndarray of test predictions
        y_test: True values
    """

    df = df.copy()
    df['WTI_target'] = df[target_column].shift(-horizon)
    df.dropna(inplace=True)

    # Ensure 'Date' is datetime and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    features = df.drop(columns=['WTI_target'])
    target = df['WTI_target']

    # Time-aware train/test split
    split_index = int(len(df) * (1 - test_ratio))
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    # Optional scaling
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Evaluation Metrics:")
    print(f"  Mean Squared Error:      {mse:.2f}")
    print(f"  Mean Absolute Error:     {mae:.2f}")
    print(f"  R^2 Score:                {r2:.4f}")

    # Plot feature importances
    if plot_importance:
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.barh(X_train.columns[sorted_indices], importances[sorted_indices])
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()

    return model, predictions, y_test


def main():
    df = data.get_data_from_csv()
    df.dropna(inplace=True)
    # df.drop(columns=["Brent ($/bbl)"], inplace=True)
    # print(df.isna().any(axis=1).sum())

    model, preds, y_true = train_random_forest_forecast_model(df)

    return 1

if __name__ == "__main__":
    main()
