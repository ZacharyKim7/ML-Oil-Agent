from data_helpers import data_tabulate as data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
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

def train_quantile_forecast_model2(df, target_column='WTI ($/bbl)', horizon=21, test_ratio=0.2, quantiles=[0.05, 0.5, 0.95], scale_features=False):
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

    # Train models for each quantile
    for q in quantiles:
        model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[q] = model
        predictions[q] = y_pred

    # === Bias Correction: Median ===
    median_pred = predictions[0.5]
    median_bias = (y_test - median_pred).mean()
    corrected_median = median_pred + median_bias
    predictions['corrected_median'] = corrected_median

    # === Bias Correction: Quantile Bounds ===
    # Lower (5%): only consider residuals where prediction < actual
    lower_residuals = y_test - predictions[0.05]
    lower_bias = lower_residuals[lower_residuals > 0].mean()
    corrected_lower = predictions[0.05] + lower_bias

    # Upper (95%): only consider residuals where prediction > actual
    upper_residuals = y_test - predictions[0.95]
    upper_bias = upper_residuals[upper_residuals < 0].mean()
    corrected_upper = predictions[0.95] + upper_bias

    predictions['corrected_lower'] = corrected_lower
    predictions['corrected_upper'] = corrected_upper

    # === Evaluation: Median Forecast ===
    print("📊 Median (50%) Forecast Evaluation:")
    print(f"  Raw        → MSE: {mean_squared_error(y_test, median_pred):.2f}, "
          f"MAE: {mean_absolute_error(y_test, median_pred):.2f}, "
          f"R²: {r2_score(y_test, median_pred):.4f}")
    print(f"  Corrected  → MSE: {mean_squared_error(y_test, corrected_median):.2f}, "
          f"MAE: {mean_absolute_error(y_test, corrected_median):.2f}, "
          f"R²: {r2_score(y_test, corrected_median):.4f}")

    # === Recompute Coverage with Corrected Interval ===
    within_bounds_corrected = (y_test >= corrected_lower) & (y_test <= corrected_upper)
    coverage_corrected = within_bounds_corrected.mean() * 100
    print(f"✅ Corrected Coverage: {coverage_corrected:.2f}% within corrected 90% interval")

    # === Plot Forecast with Bias-Corrected Median & Interval ===
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label='Actual', color='black')
    plt.plot(y_test.index, corrected_median, label='Corrected Median Forecast', color='blue')
    plt.fill_between(y_test.index, corrected_lower, corrected_upper, color='blue', alpha=0.2,
                     label='Bias-Corrected 90% Prediction Interval')
    plt.legend()
    plt.title("WTI Forecast with Bias-Corrected Quantile Intervals")
    plt.xlabel("Date")
    plt.ylabel("WTI Price ($/bbl)")
    plt.text(
        x=0.99, y=0.02,
        s=f"Coverage: {coverage_corrected:.2f}% within corrected 90% interval",
        transform=plt.gca().transAxes,
        ha='right', va='bottom',
        fontsize=10, color='dimgray',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )
    plt.tight_layout()
    plt.show()

    # === Optional: Calibration Plot ===
    def plot_bias():
        plt.scatter(corrected_median, y_test, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Predicted Median")
        plt.ylabel("Actual WTI Price")
        plt.title("Calibration Plot")
        plt.grid(True)
        plt.show()
    # plot_bias()

    return models, predictions, y_test

def train_xgboost_quantile_forecast_model(df, target_column='WTI ($/bbl)', horizon=21,
                                          test_ratio=0.2, quantiles=[0.05, 0.5, 0.95],
                                          lags=[1, 5, 21], roll_windows=[5, 21],
                                          scale_features=False):

    def create_lag_features(df, target_column, lags=[1, 5, 21], roll_windows=[5, 21]):
        df = df.copy()
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        for window in roll_windows:
            df[f'{target_column}_rollmean_{window}'] = df[target_column].shift(1).rolling(window).mean()
            df[f'{target_column}_rollstd_{window}'] = df[target_column].shift(1).rolling(window).std()
        return df
    
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Create target: rolling average over next `horizon` business days
    df['WTI_target'] = df[target_column].shift(-1).rolling(horizon).mean()

    # Create lag and rolling window features for the target column
    df = create_lag_features(df, target_column, lags, roll_windows)

    # Optionally, create lag/roll features for other variables (not shown for brevity)

    df.dropna(inplace=True)
    df = df.sort_index()

    # Drop target leakage and unrelated fields
    features = df.drop(columns=['WTI_target', target_column])
    target = df['WTI_target']

    # Train/test split
    split_index = int(len(df) * (1 - test_ratio))
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    if scale_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    models = {}
    predictions = {}

    for q in quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[q] = model
        predictions[q] = y_pred

        if q == 0.5:
            # Evaluate median forecast
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("📊 Median (50%) Forecast Evaluation:")
            print(f"  MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # Prediction Interval Coverage
    if all(q in predictions for q in [0.05, 0.95]):
        within_bounds = (y_test >= predictions[0.05]) & (y_test <= predictions[0.95])
        coverage = within_bounds.mean() * 100
        print(f"Coverage: {coverage:.2f}% of actual values fall within the 90% prediction interval.")

        # Plot results
        plt.figure(figsize=(12, 5))
        plt.plot(y_test.index, y_test.values, label='Actual', color='black')
        plt.plot(y_test.index, predictions[0.5], label='Median Forecast', color='blue')
        plt.fill_between(y_test.index, predictions[0.05], predictions[0.95], color='blue', alpha=0.2,
                         label='90% Prediction Interval')
        plt.legend()
        plt.title("WTI Forecast with Quantile Intervals (XGBoost)")
        plt.xlabel("Date")
        plt.ylabel("WTI Price ($/bbl)")
        plt.text(
            x=0.99, y=0.02,
            s=f"Coverage: {coverage:.2f}%",
            transform=plt.gca().transAxes,
            ha='right', va='bottom',
            fontsize=10, color='dimgray', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
        plt.tight_layout()
        plt.show()

    # === Optional: Calibration Plot ===
    def plot_bias():
        # plt.scatter(predictions[0.5], y_test, alpha=0.4)
        # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        # plt.xlabel("Predicted Median")
        # plt.ylabel("Actual WTI Price")
        # plt.title("Calibration Plot")
        # plt.grid(True)
        plt.plot(predictions[0.05], label="0.05 Quantile", color="red")
        plt.plot(predictions[0.95], label="0.95 Quantile", color="green")
        plt.legend(); plt.show()
        plt.show()
    plot_bias()

def main():
    df = data.get_data_from_csv()
    df.dropna(inplace=True)
    # df.drop(columns=["Brent ($/bbl)"], inplace=True)
    # print(df.isna().any(axis=1).sum())
    # print(df['USD-YEN'].dtype)
    
    # for col in df.columns:
    #     print(f"{col}: {df[col].dtype}")

    # models, preds, y_test = train_random_forest_forecast_model(df)
    # models, preds, y_test = train_quantile_forecast_model(df, horizon=21)
    # models, preds, y_test = train_quantile_forecast_model2(df, horizon=21)
    train_xgboost_quantile_forecast_model(df)

    return 1

# if __name__ == "__main__":
main()
