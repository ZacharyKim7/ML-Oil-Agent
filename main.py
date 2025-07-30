from data_helpers import data_tabulate as data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

    # Ensure 'Date' is datetime and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    spot_predictor = True
    if spot_predictor:
        df['WTI_target'] = df[target_column].shift(-horizon)
        df.dropna(inplace=True)
        features = df.drop(columns=['WTI_target'])
        target = df['WTI_target']

    rolling_mean = False
    if rolling_mean:
        # Step 1: Create rolling target from future values
        df['WTI_target'] = df[target_column].shift(-1).rolling(window=horizon).mean()

        # Step 2: Drop NaNs after rolling
        df.dropna(inplace=True)

        # Step 3: Drop original column (to avoid data leakage)
        features = df.drop(columns=['WTI_target', target_column])

        # Step 4: Assign the target variable
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

def train_quantile_forecast_model(df, target_column='WTI ($/bbl)', horizon=21, test_ratio=0.2, quantiles=[0.05, 0.5, 0.95], scale_features=False):
    """
    Trains multiple GradientBoostingRegressor models to predict lower, median, and upper quantile forecasts.

    Returns:
        models: dict of trained models by quantile
        predictions: dict of predicted arrays by quantile
        y_test: actual target values
    """
    df = df.copy()

    # Ensure 'Date' is datetime and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Target: average WTI over next `horizon` days
    df['WTI_target'] = df[target_column].shift(-1).rolling(window=horizon).mean()
    df.dropna(inplace=True)

    df = df.sort_index()

    # Drop current WTI to avoid leakage
    features = df.drop(columns=['WTI_target', target_column])
    target = df['WTI_target']

    # Split
    split_index = int(len(df) * (1 - test_ratio))
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    # Optional scaling
    if scale_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    models = {}
    predictions = {}

    for q in quantiles:
        model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[q] = model
        predictions[q] = y_pred

        if q == 0.5:
            # Evaluate only on median model
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("📊 Median (50%) Forecast Evaluation:")
            print(f"  MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    within_bounds = (y_test >= predictions[0.05]) & (y_test <= predictions[0.95])
    coverage = within_bounds.mean() * 100
    print(f"Coverage: {coverage:.2f}% of actual values fall within the 90% prediction interval.")

    if all(q in predictions for q in [0.05, 0.5, 0.95]):
        plt.figure(figsize=(12, 5))
        plt.plot(y_test.index, y_test.values, label='Actual', color='black')
        plt.plot(y_test.index, predictions[0.5], label='Median Forecast', color='blue')
        plt.fill_between(y_test.index, predictions[0.05], predictions[0.95], color='blue', alpha=0.2,
                         label='90% Prediction Interval')
        plt.legend()
        plt.title("WTI Forecast with Quantile Intervals")
        plt.xlabel("Date")
        plt.ylabel("WTI Price ($/bbl)")
        plt.text(
            x=0.99, y=0.02,  # relative position in axes coordinates
            s=f"Coverage: {coverage:.2f}% within 90% prediction interval",
            transform=plt.gca().transAxes,  # so x/y are [0,1] axis coords
            ha='right', va='bottom',        # align text to bottom right
            fontsize=10, color='dimgray', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        plt.tight_layout()
        plt.show()

    # Plots actual over median values to reveal a over/under value bias.
    def plot_bias():
        plt.scatter(predictions[0.5], y_test, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Predicted Median")
        plt.ylabel("Actual WTI Price")
        plt.title("Calibration Plot")
        plt.grid(True)
        plt.show()
    # plot_bias()

    return models, predictions, y_test

def main():
    df = data.get_data_from_csv()
    df.dropna(inplace=True)
    # df.drop(columns=["Brent ($/bbl)"], inplace=True)
    # print(df.isna().any(axis=1).sum())

    # model, preds, y_true = train_random_forest_forecast_model(df)
    models, preds, y_test = train_quantile_forecast_model(df, horizon=21)

    return 1

if __name__ == "__main__":
    main()
