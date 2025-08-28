import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
import warnings
from data_helpers import data_tabulate as data
import os
import joblib
import json
from config import *

warnings.filterwarnings('ignore')

def create_features(df):
    """
    Create comprehensive features for oil price prediction including:
    - Technical indicators (moving averages, volatility, momentum)
    - Fundamental factors (supply/demand ratios, inventory levels)
    - Economic indicators (currency, inflation, growth)
    - Seasonal patterns
    
    CRITICAL: All features must use only PAST information (lookback only)
    """
    df = df.copy()
    
    # Ensure chronological order
    df = df.sort_index()
    
    # Technical indicators for WTI (lookback only)
    df['wti_ma_5'] = df['WTI ($/bbl)'].rolling(5, min_periods=5).mean()
    df['wti_ma_21'] = df['WTI ($/bbl)'].rolling(21, min_periods=21).mean()
    df['wti_ma_63'] = df['WTI ($/bbl)'].rolling(63, min_periods=63).mean()
    df['wti_volatility_21'] = df['WTI ($/bbl)'].rolling(21, min_periods=21).std()
    df['wti_rsi'] = calculate_rsi(df['WTI ($/bbl)'], 14)
    
    # Momentum (always looking backward)
    df['wti_momentum_5'] = df['WTI ($/bbl)'] - df['WTI ($/bbl)'].shift(5)
    df['wti_momentum_21'] = df['WTI ($/bbl)'] - df['WTI ($/bbl)'].shift(21)
    
    # Brent-WTI spread (current and historical)
    df['brent_wti_spread'] = df['Brent ($/bbl)'] - df['WTI ($/bbl)']
    df['brent_wti_spread_ma'] = df['brent_wti_spread'].rolling(21, min_periods=21).mean()
    
    # Supply/Demand fundamentals (current values only, no forward-looking)
    df['total_supply'] = df['OPEC P (tbbl/d)'] + df['NOPEC P (tbbl/d)']
    df['total_demand'] = df['OECD C (tbbl/d)'] + df['China C (tbbl/d)'] + df['India C (tbbl/d)']
    df['supply_demand_ratio'] = df['total_supply'] / df['total_demand']
    df['supply_demand_balance'] = df['total_supply'] - df['total_demand']
    
    # Inventory indicators (lookback only)
    df['total_stocks'] = df['OECD S (mbbl)'] + df['USA S (mbbl)']
    df['stocks_ma_21'] = df['total_stocks'].rolling(21, min_periods=21).mean()
    df['stocks_change'] = df['total_stocks'] - df['total_stocks'].shift(21)  # Change from 21 days ago
    
    # US-specific indicators
    df['us_production_proxy'] = df['USA rigs (m)']
    df['us_net_imports_ma'] = df['US net imports (tbbl/d)'].rolling(21, min_periods=21).mean()
    
    # Economic indicators (all backward-looking)
    df['cpi_change'] = df['CPI'].pct_change(21)  # 21-day change
    df['gdp_momentum'] = df['GDP (yoy%)'].diff()  # Change in growth rate
    df['usd_strength'] = (1/df['USD-GBP'] + df['USD-YEN']/100) / 2
    df['usd_change'] = df['usd_strength'].pct_change(21)
    
    # Seasonal patterns (safe - based on calendar date)
    df['month'] = pd.to_datetime(df.index).month
    df['quarter'] = pd.to_datetime(df.index).quarter
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # Lagged features (explicitly backward-looking)
    for lag in [1, 5, 21]:
        df[f'wti_lag_{lag}'] = df['WTI ($/bbl)'].shift(lag)
        df[f'supply_demand_lag_{lag}'] = df['supply_demand_ratio'].shift(lag)
        df[f'stocks_lag_{lag}'] = df['total_stocks'].shift(lag)
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_target_variable(df, forecast_horizon=21):
    """
    Create target variable: monthly average WTI price 21 days ahead
    CRITICAL: Ensure no data leakage by using only past information
    """
    df = df.copy()
    
    # Debug: Check for duplicate dates
    print(f"🔍 Checking data integrity...")
    duplicate_dates = df.index.duplicated().sum()
    if duplicate_dates > 0:
        print(f"⚠️  Found {duplicate_dates} duplicate dates - removing duplicates...")
        df = df[~df.index.duplicated(keep='first')]
        print(f"✅ After deduplication: {len(df)} rows")
    
    # Sort by date to ensure proper chronological order
    df = df.sort_index()
    
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total business days: {len(df)}")
    
    # Calculate monthly averages for all months in the dataset
    df_temp = df.copy()
    df_temp['year_month'] = df_temp.index.to_period('M')
    monthly_averages = df_temp.groupby('year_month')['WTI ($/bbl)'].mean()
    
    print(f"Monthly averages calculated for {len(monthly_averages)} months")
    
    # For each row, find what the monthly average will be 21 business days in the future
    target_values = []
    
    for i, current_date in enumerate(df.index):
        # Find the date 21 business days in the future
        try:
            # Method 1: Add 21 business days using pandas business day offset
            future_date = current_date + pd.tseries.offsets.BDay(forecast_horizon)
            
            # Get the year-month period for that future date
            future_period = future_date.to_period('M')
            
            # Look up the monthly average for that period
            if future_period in monthly_averages.index:
                target_value = monthly_averages[future_period]
            else:
                target_value = np.nan
                
        except Exception as e:
            print(f"Error processing date {current_date}: {e}")
            target_value = np.nan
            
        target_values.append(target_value)
    
    df['target_monthly_avg'] = target_values
    
    # Remove rows where we don't have future data
    df_clean = df.dropna(subset=['target_monthly_avg'])
    
    print(f"✅ Target variable created: {len(df_clean)} valid predictions")
    print(f"Removed {len(df) - len(df_clean)} rows without future data")
    
    return df_clean

def train_oil_price_model(df, forecast_horizon=21):
    """
    Train Random Forest model to predict monthly average WTI oil price
    with strict anti-leakage measures
    """
    print("🛢️  Training WTI Oil Price Prediction Model")
    print("=" * 50)
    
    # Feature engineering
    df_features = df.copy()
    df_with_target = prepare_target_variable(df_features, forecast_horizon)
    
    # Select feature columns (exclude target and non-predictive columns)
    exclude_cols = [
        'WTI ($/bbl)', 'target_monthly_avg', 'monthly_avg_current', 'year_month',
        'future_date', 'future_year_month', 'Brent ($/bbl)'  # Also exclude Brent as it's too correlated
    ]
    feature_cols = [col for col in df_with_target.columns if col not in exclude_cols]
    
    # Remove rows with NaN values
    df_clean = df_with_target.dropna()
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    
    if len(df_clean) < 100:
        print("⚠️  Warning: Limited data available after feature engineering")
        print("Consider reducing the number of rolling window features")
        return None, None, None, None, None
    
    X = df_clean[feature_cols]
    y = df_clean['target_monthly_avg']
    
    # Use time series split for validation (critical for preventing leakage)
    tscv = TimeSeriesSplit(n_splits=3, test_size=252)  # ~1 year test sets
    
    # More conservative Random Forest to prevent overfitting
    model = RandomForestRegressor(
        n_estimators=50,        # Reduced from 200
        max_depth=6,           # Reduced from 10
        min_samples_split=10,  # Increased from 5
        min_samples_leaf=5,    # Increased from 2
        max_features=0.3,      # Add feature subsampling
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
    print(f"Cross-validation MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Train on full dataset
    model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, X, y, feature_cols, df_clean

def generate_predictions(model, df, feature_cols, forecast_horizon=21):
    """
    Generate predictions and calculate performance metrics
    """
    df_features = df.copy()
    df_with_target = prepare_target_variable(df_features, forecast_horizon)
    df_clean = df_with_target.dropna()
    
    X = df_clean[feature_cols]
    y_true = df_clean['target_monthly_avg']
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'date': df_clean.index,
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred)
    })
    
    return results_df, {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

def plot_results(results_df, metrics):
    """
    Create comprehensive visualization of model results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('WTI Oil Price Prediction Model Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted Time Series
    ax1 = axes[0, 0]
    ax1.plot(results_df['date'], results_df['actual'], label='Actual', color='blue', alpha=0.7)
    ax1.plot(results_df['date'], results_df['predicted'], label='Predicted', color='red', alpha=0.7)
    ax1.set_title('Actual vs Predicted Monthly Average WTI Prices')
    ax1.set_ylabel('Price ($/bbl)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: Predicted vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(results_df['actual'], results_df['predicted'], alpha=0.6, color='green')
    min_price = min(results_df['actual'].min(), results_df['predicted'].min())
    max_price = max(results_df['actual'].max(), results_df['predicted'].max())
    ax2.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8)
    ax2.set_xlabel('Actual Price ($/bbl)')
    ax2.set_ylabel('Predicted Price ($/bbl)')
    ax2.set_title(f'Predicted vs Actual (R² = {metrics["r2"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction Errors Over Time
    ax3 = axes[1, 0]
    ax3.plot(results_df['date'], results_df['error'], color='purple', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Prediction Errors Over Time')
    ax3.set_ylabel('Error ($/bbl)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    ax4 = axes[1, 1]
    ax4.hist(results_df['error'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_title('Distribution of Prediction Errors')
    ax4.set_xlabel('Error ($/bbl)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oil_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Oil prediction plots saved as 'oil_prediction_results.png'")
    
    # Additional metrics summary
    print(f"\n📊 Additional Model Insights:")
    print(f"Average absolute error: ${results_df['abs_error'].mean():.2f}")
    print(f"Median absolute error: ${results_df['abs_error'].median():.2f}")
    print(f"95th percentile error: ${results_df['abs_error'].quantile(0.95):.2f}")
    print(f"Error standard deviation: ${results_df['error'].std():.2f}")

def predict_future_price(model, df, feature_cols, forecast_horizon=21):
    """
    Make a prediction for the current monthly average price 21 days out
    """
    df_features = df.copy()
    
    # Get the most recent complete feature set
    latest_features = df_features[feature_cols].dropna().iloc[-1:]
    
    if len(latest_features) == 0:
        print("⚠️  Cannot make prediction - insufficient recent data")
        return None
    
    prediction = model.predict(latest_features)[0]
    
    # Calculate prediction date
    last_date = df.index[-1]
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # Add 21 business days
    prediction_date = last_date + pd.offsets.BDay(forecast_horizon)
    prediction_month = prediction_date.strftime('%B %Y')
    
    print(f"\n🔮 Future Prediction:")
    print(f"Current WTI price: ${df['WTI ($/bbl)'].iloc[-1]:.2f}")
    print(f"Predicted monthly average for {prediction_month}: ${prediction:.2f}")
    print(f"Prediction date range: ~{prediction_date.strftime('%Y-%m-%d')}")
    
    return prediction, prediction_date

def oil_price_prediction_pipeline(df, verbose=True):
    """
    Complete pipeline for oil price prediction
    """
    if verbose:
        print("🚀 Starting WTI Oil Price Prediction Analysis")
        print("=" * 60)
    
    # CRITICAL FIX: Handle reverse chronological data
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    
    # Check if data is in reverse chronological order and fix it
    if df.index[0] > df.index[-1]:
        df = df.iloc[::-1]
    
    print(f"Data range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(df)}")

    df = create_features(df)
    
    # Train model
    model, X, y, feature_cols, df_clean = train_oil_price_model(df)
    
    # Generate predictions and evaluate
    results_df, metrics = generate_predictions(model, df, feature_cols)
    
    # Plot results
    plot_results(results_df, metrics)

    # # Production retrain on 100% of data and forecast next month's average value
    # prod_model, forecast_point, forecast_date, forecast_month = fit_production_and_forecast(
    #     df_raw=df, feature_cols=feature_cols, X_full=X, y_full=y, forecast_horizon=forecast_horizon
    # )
    
    # Make future prediction
    future_pred, pred_date = predict_future_price(model, df, feature_cols)
    
    return model, results_df, metrics, future_pred

def run_oil_prediction(df, verbose=True):
    """
    Run the complete oil price prediction analysis
    
    Parameters:
    df: DataFrame with oil data
    verbose: Whether to print detailed output
    
    Returns:
    tuple: (trained_model, results_dataframe, performance_metrics, future_prediction)
    """
    # Run the pipeline
    model, results, metrics, prediction = oil_price_prediction_pipeline(df, verbose)
    
    return model, results, metrics, prediction

def create_stock_features(df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=TICKERS):
    df = df_stocks.copy()

    # --- Oil & macro features (shared) ---
    df['wti_current'] = df['WTI ($/bbl)']
    df['wti_ma_5']  = df['WTI ($/bbl)'].rolling(5,  min_periods=5).mean()
    df['wti_ma_21'] = df['WTI ($/bbl)'].rolling(21, min_periods=21).mean()
    df['wti_volatility'] = df['WTI ($/bbl)'].rolling(21, min_periods=21).std()
    df['wti_momentum']   = df['WTI ($/bbl)'] - df['WTI ($/bbl)'].shift(21)

    # Oil model prediction as a feature
    try:
        df_oil_clean = df_oil_features.dropna()
        preds = []
        for idx in df.index:
            if idx in df_oil_clean.index:
                feats = df_oil_clean.loc[idx:idx, oil_feature_cols]
                preds.append(oil_model.predict(feats)[0] if not feats.isnull().any().any() else np.nan)
            else:
                preds.append(np.nan)
        df['oil_price_prediction'] = preds
        df['oil_prediction_premium'] = df['oil_price_prediction'] - df['wti_current']
    except Exception as e:
        print(f"Warning: oil preds unavailable: {e}")
        df['oil_price_prediction'] = np.nan
        df['oil_prediction_premium'] = np.nan

    # Econ features
    if 'usd_strength' not in df.columns:
        df['usd_strength'] = (1/df['USD-GBP'] + df['USD-YEN']/100) / 2
    df['usd_change'] = df['usd_strength'].pct_change(21)

    # Seasonal
    dt = pd.to_datetime(df.index)
    df['month'] = dt.month
    df['quarter'] = dt.quarter

    # --- Per-ticker technicals/relations ---
    for t in tickers:
        # assume df[t] exists and is the daily close for that ticker
        df[f'{t}_ma_5']  = df[t].rolling(5,  min_periods=5).mean()
        df[f'{t}_ma_21'] = df[t].rolling(21, min_periods=21).mean()
        df[f'{t}_vol']   = df[t].rolling(21, min_periods=21).std()
        df[f'{t}_mom_21']= df[t] - df[t].shift(21)
        df[f'{t}_rsi']   = calculate_rsi(df[t], 14)

        # oil-stock relationship for this ticker
        df[f'{t}_corr_wti_63'] = df['WTI ($/bbl)'].rolling(63).corr(df[t])
        df[f'{t}_ratio_wti']   = df['WTI ($/bbl)'] / df[t]
        df[f'{t}_ratio_ma_21'] = df[f'{t}_ratio_wti'].rolling(21, min_periods=21).mean()

        # lags of the ticker and WTI
        for lag in [1, 5, 21]:
            df[f'{t}_lag_{lag}'] = df[t].shift(lag)
        # (one shared set of oil lags is fine; optional per-ticker duplicates removed)

    return df

def prepare_stock_target_variable_multi(df, tickers, forecast_horizon=21):
    df = df.sort_index().copy()
    for t in tickers:
        df[f'target_{t}'] = df[t].shift(-forecast_horizon)
    # drop only rows where any target is missing
    target_cols = [f'target_{t}' for t in tickers]
    df_labeled = df.dropna(subset=target_cols)
    print(f"Stock targets created for {tickers}: {len(df_labeled)} valid rows")
    return df, df_labeled, target_cols


def plot_stock_results(results_df, metrics):
    """
    Multi-output plotting:
      - Saves one 2x3 dashboard per ticker: stock_results_<TICKER>.png
      - Saves a comparison overlay: stock_results_comparison.png
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    tickers = sorted(results_df['ticker'].unique())

    # ---------- Per-ticker dashboards ----------
    for t in tickers:
        df_t = results_df[results_df['ticker'] == t].copy()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{t} Stock Price Prediction Model Results', fontsize=16, fontweight='bold')

        # 1) Actual vs Predicted (time series)
        ax1 = axes[0, 0]
        ax1.plot(df_t['date'], df_t['actual'], label='Actual', alpha=0.7)
        ax1.plot(df_t['date'], df_t['predicted'], label='Predicted', alpha=0.7)
        ax1.set_title('Actual vs Predicted')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Scatter Pred vs Actual
        ax2 = axes[0, 1]
        ax2.scatter(df_t['actual'], df_t['predicted'], alpha=0.6)
        min_p = min(df_t['actual'].min(), df_t['predicted'].min())
        max_p = max(df_t['actual'].max(), df_t['predicted'].max())
        ax2.plot([min_p, max_p], [min_p, max_p], linestyle='--', alpha=0.8)
        r2 = metrics.get(t, {}).get('r2', np.nan)
        ax2.set_xlabel('Actual ($)')
        ax2.set_ylabel('Predicted ($)')
        ax2.set_title(f'Predicted vs Actual (R² = {r2:.3f})' if np.isfinite(r2) else 'Predicted vs Actual')
        ax2.grid(True, alpha=0.3)

        # 3) Errors over time
        ax3 = axes[0, 2]
        ax3.plot(df_t['date'], df_t['error'], alpha=0.7)
        ax3.axhline(y=0, linestyle='--', alpha=0.5)
        ax3.set_title('Prediction Errors Over Time')
        ax3.set_ylabel('Error ($)')
        ax3.grid(True, alpha=0.3)

        # 4) Stock vs Oil Price
        ax4 = axes[1, 0]
        if 'actual_oil_price' in df_t.columns:
            ax4.scatter(df_t['actual_oil_price'], df_t['actual'], alpha=0.6, label='Actual')
            ax4.set_xlabel('Oil Price ($/bbl)')
            ax4.set_ylabel(f'{t} Price ($)')
            ax4.set_title('Stock vs Oil Price')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5) Oil predictions vs Stock (if available)
        ax5 = axes[1, 1]
        if 'oil_prediction' in df_t.columns:
            valid_oil_pred = df_t.dropna(subset=['oil_prediction'])
            if len(valid_oil_pred) > 0:
                ax5.scatter(valid_oil_pred['oil_prediction'], valid_oil_pred['actual'], alpha=0.6)
                ax5.set_xlabel('Oil Price Prediction ($/bbl)')
                ax5.set_ylabel(f'{t} Price ($)')
                ax5.set_title('Stock vs Oil Prediction')
            ax5.grid(True, alpha=0.3)

        # 6) Error distribution
        ax6 = axes[1, 2]
        ax6.hist(df_t['error'], bins=30, alpha=0.7, edgecolor='black')
        ax6.axvline(x=0, linestyle='--', alpha=0.8)
        ax6.set_title('Distribution of Prediction Errors')
        ax6.set_xlabel('Error ($)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = f'stock_results_{t}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved per-ticker plot: {out_path}")

    # ---------- Combined comparison overlay ----------
    # Actual vs Predicted overlay for all tickers
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Actual vs Predicted (All Tickers)')
    for t in tickers:
        df_t = results_df[results_df['ticker'] == t]
        ax.plot(df_t['date'], df_t['actual'], alpha=0.6, label=f'{t} Actual')
        ax.plot(df_t['date'], df_t['predicted'], alpha=0.8, label=f'{t} Predicted', linestyle='--')
    ax.set_ylabel('Price ($)')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = 'stock_results_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined comparison plot: {out_path}")

def build_stock_dataset(df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=["SLB"], horizon=21, verbose=False):
    df_feat_all = create_stock_features(df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=tickers)
    df_tgt_all, df_labeled, target_cols = prepare_stock_target_variable_multi(df_feat_all, tickers, horizon)

    # feature selection (exclude raw prices & targets; keep engineered)
    exclude = set(['WTI ($/bbl)','Brent ($/bbl)'] + list(tickers) + target_cols)
    feature_cols = [c for c in df_labeled.columns if c not in exclude and df_labeled[c].dtype != 'object']

    X = df_labeled[feature_cols]
    y = df_labeled[target_cols]          # shape (n_samples, 2)
    return df_feat_all, df_labeled, X, y, feature_cols, target_cols

def train_stock_model_on_dataset(X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=126)
    # manual CV because multi-output + neg_mae is awkward with cross_val_score
    maes = []
    for tr, te in tscv.split(X):
        m = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_split=8, min_samples_leaf=4,
            max_features=0.4, random_state=42, n_jobs=-1
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])  # (n,2)
        mae_each = np.mean(np.abs(pred - y.iloc[te].values), axis=0)  # per-output
        maes.append(mae_each)
    maes = np.array(maes)
    print(f"CV MAE per ticker: {y.columns.tolist()} -> {maes.mean(axis=0)} ± {maes.std(axis=0)}")

    model = RandomForestRegressor(
        n_estimators=100, max_depth=8,
        min_samples_split=8, min_samples_leaf=4,
        max_features=0.4, random_state=42, n_jobs=-1
    )
    model.fit(X, y)  # one fit, two outputs
    return model

def evaluate_stock_model_multi(model, df_clean, feature_cols, target_cols, tickers=TICKERS):
    X = df_clean[feature_cols]
    y_true = df_clean[target_cols].values  # (n,2)
    y_pred = model.predict(X)              # (n,2)

    metrics = {}
    results_frames = []
    for i, t in enumerate(tickers):
        yt, yp = y_true[:, i], y_pred[:, i]
        mae  = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / yt)) * 100
        metrics[t] = dict(mae=mae, rmse=rmse, r2=r2, mape=mape)

        results_frames.append(pd.DataFrame({
            'date': df_clean.index,
            'ticker': t,
            'actual': yt,
            'predicted': yp,
            'error': yt - yp,
            'abs_error': np.abs(yt - yp),
            'actual_oil_price': df_clean['wti_current'],
            'oil_prediction': df_clean.get('oil_price_prediction', np.nan)
        }))

    results_df = pd.concat(results_frames, ignore_index=True)
    print("\n📈 Per-ticker metrics:")
    for t in tickers:
        m = metrics[t]
        print(f"{t}: MAE ${m['mae']:.2f} | RMSE ${m['rmse']:.2f} | R² {m['r2']:.3f} | MAPE {m['mape']:.2f}%")
    return results_df, metrics

def predict_future_stock_price_multi(model, df_features_all, feature_cols, tickers=TICKERS, horizon=21):
    latest_row = df_features_all[feature_cols].dropna().iloc[-1:]
    if latest_row.empty:
        print("⚠️  Cannot make stock prediction - insufficient recent data")
        return None, None

    preds = model.predict(latest_row).ravel()  # array of length len(tickers)
    last_date = pd.to_datetime(df_features_all.index[-1])
    pred_date = last_date + pd.tseries.offsets.BDay(horizon)

    print(f"\n🔮 Future Stock Predictions for {pred_date:%Y-%m-%d}:")
    out = {}
    for t, p in zip(tickers, preds):
        out[t] = float(p)
        print(f"  {t}: ${p:.2f}")
    return out, pred_date

def run_stock_prediction(df_stocks, oil_model, oil_feature_cols, df_oil_features):
    """
    Main function for stock prediction - now with single feature computation
    Predicts monthly average price for combined SLB + HAL stocks
    """
    print("\n🚀 Starting Oil Service Stock Price Prediction Analysis")
    print("=" * 70)
    print("Target: Monthly average price for SAL and HAL")
    print("=" * 70)

    TICKERS = ["SLB", "HAL"]

    if 'Date' in df_stocks.columns:
        df_stocks = df_stocks.set_index('Date')
    df_stocks.index = pd.to_datetime(df_stocks.index)
    if df_stocks.index[0] > df_stocks.index[-1]:
        df_stocks = df_stocks.iloc[::-1]

    print(f"Stock data range: {df_stocks.index.min():%Y-%m-%d} to {df_stocks.index.max():%Y-%m-%d}")
    print(f"Total observations: {len(df_stocks)}")

    # 🎯 SINGLE FEATURE COMPUTATION - Build dataset once, reuse everywhere
    print("Building stock features and dataset...")
    df_feat_all, df_clean, X, y, feature_cols, target_cols = build_stock_dataset(df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=TICKERS)    
    print(f"✅ Features built. Dataset shape: {df_clean.shape}, Features: {len(feature_cols)}")
    
    # Debug: Show the date range for predictions
    print(f"📅 Dataset date range: {df_clean.index.min().strftime('%Y-%m-%d')} to {df_clean.index.max().strftime('%Y-%m-%d')}")
    print(f"📅 Target variable range: {df_clean['target_SLB'].dropna().index.min().strftime('%Y-%m-%d')} to {df_clean['target_SLB'].dropna().index.max().strftime('%Y-%m-%d')}")

    # Train model
    print("Training stock prediction model...")
    model = train_stock_model_on_dataset(X, y)

    # Evaluate model
    print("Evaluating model performance...")
    results_df, metrics = evaluate_stock_model_multi(model, df_clean, feature_cols, target_cols, tickers=TICKERS)

    # Plot results
    plot_stock_results(results_df, metrics)

    # Make future prediction using the SAME feature dataset
    print("Generating future predictions...")
    preds, pred_date = predict_future_stock_price_multi(model, df_feat_all, feature_cols, tickers=TICKERS)

    return model, results_df, metrics, preds

from sklearn.linear_model import Ridge

def train_base_model_on_anchors(X, y, all_tickers, anchor_tickers, n_splits=3):
    """
    Train the multi-output base model on anchor tickers only.
    Returns:
        base_model: fitted RandomForestRegressor (multi-output for anchors)
        anchor_cols: list of y column names used
        anchor_idx_map: {ticker -> column index in base_model output}
    """
    anchor_cols = [f'target_{t}' for t in anchor_tickers]
    y_anchor = y[anchor_cols]

    # CV on anchors (optional but nice to keep)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=126)
    maes = []
    for tr, te in tscv.split(X):
        m = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_split=8, min_samples_leaf=4,
            max_features=0.4, random_state=42, n_jobs=-1
        )
        m.fit(X.iloc[tr], y_anchor.iloc[tr])
        pred = m.predict(X.iloc[te])
        mae_each = np.mean(np.abs(pred - y_anchor.iloc[te].values), axis=0)
        maes.append(mae_each)
    maes = np.array(maes)
    print(f"[Base/Anchors] CV MAE -> {anchor_cols}: {maes.mean(axis=0)} ± {maes.std(axis=0)}")

    base_model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=6, min_samples_leaf=3,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    base_model.fit(X, y_anchor)

    anchor_idx_map = {t: i for i, t in enumerate(anchor_tickers)}
    return base_model, anchor_cols, anchor_idx_map


def compute_basket_pred(base_model, X, anchor_tickers, anchor_idx_map):
    """
    Run base model and compute the anchor basket prediction as the mean across anchors.
    Returns:
        basket_pred: pd.Series aligned to X.index
        preds_df:    pd.DataFrame with per-anchor predictions (columns = anchor tickers)
    """
    import pandas as pd
    base_preds = base_model.predict(X)  # shape (n, len(anchors))
    preds_df = pd.DataFrame(
        base_preds,
        index=X.index,
        columns=anchor_tickers
    )
    basket_pred = preds_df.mean(axis=1).rename("basket_pred")
    return basket_pred, preds_df


def train_transfer_adapters(df_clean, feature_cols, target_cols,
                            basket_pred, shared_extra_cols=None,
                            all_tickers=None, anchor_tickers=None):
    """
    For each NON-anchor ticker, train a small adapter (Ridge by default):
        adapter_features = [basket_pred] + selected shared features
        target = target_<ticker>
    Returns:
        adapters: dict[ticker] = fitted adapter model
        adapter_features_cols: list of columns used for adapters
    """
    import pandas as pd
    adapters = {}

    if all_tickers is None:
        # infer from target_cols
        all_tickers = [c.replace('target_', '') for c in target_cols]
    non_anchor = [t for t in all_tickers if t not in set(anchor_tickers or [])]

    # Build adapter design matrix
    # Start with basket_pred (mandatory signal)
    design = pd.DataFrame(index=df_clean.index)
    design['basket_pred'] = basket_pred

    # A few robust shared features (low risk of leakage/regime-shift-sensitive)
    default_shared = ['wti_current', 'usd_strength', 'month', 'quarter', 'oil_price_prediction']
    shared_extra_cols = shared_extra_cols or []
    adapter_cols = ['basket_pred'] + [c for c in default_shared + shared_extra_cols
                                      if c in df_clean.columns and c in feature_cols]

    for c in adapter_cols[1:]:  # already added basket_pred
        design[c] = df_clean[c]

    # Train one adapter per non-anchor ticker
    for t in non_anchor:
        target_col = f'target_{t}'
        if target_col not in df_clean.columns:
            print(f"[Adapter] Skipping {t}: no target column found.")
            continue

        # Use only rows where both features and target exist
        df_t = design.join(df_clean[[target_col]], how='inner').dropna()
        if len(df_t) < 200:
            print(f"[Adapter] Skipping {t}: insufficient overlap rows ({len(df_t)}).")
            continue

        X_t = df_t[adapter_cols]
        y_t = df_t[target_col]

        # A simple, stable adapter
        adapter = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
        adapter.fit(X_t, y_t)
        adapters[t] = adapter

        # Quick diagnostics
        yhat = adapter.predict(X_t)
        mae = mean_absolute_error(y_t, yhat)
        r2  = r2_score(y_t, yhat)
        print(f"[Adapter] {t}: trained on {len(df_t)} rows | MAE={mae:.3f} | R²={r2:.3f}")

    return adapters, adapter_cols

def evaluate_with_adapters(base_model, df_clean, feature_cols, target_cols,
                           tickers, anchor_tickers, anchor_idx_map, adapter_models, adapter_cols):
    """
    Produces a results_df/metrics using:
      - anchor tickers -> base_model directly
      - non-anchors    -> adapter(basket_pred, shared features)
    """
    import pandas as pd
    # Base preds for anchors + basket
    X = df_clean[feature_cols]
    basket_pred, anchor_pred_df = compute_basket_pred(base_model, X, anchor_tickers, anchor_idx_map)

    metrics = {}
    frames = []

    for t in tickers:
        target_col = f"target_{t}"
        if target_col not in df_clean.columns:
            continue

        # align target rows
        valid = df_clean[target_col].notna()
        y_true = df_clean.loc[valid, target_col].values
        dates  = df_clean.loc[valid].index

        if t in anchor_tickers:
            # direct from base model column
            y_hat_full = anchor_pred_df[t]
            y_pred = y_hat_full.loc[dates].values
        else:
            # adapter path
            if t not in adapter_models:
                print(f"[Eval] No adapter for {t}; skipping.")
                continue
            # build adapter features (aligned)
            A = pd.DataFrame(index=df_clean.index)
            A['basket_pred'] = basket_pred
            for c in adapter_cols:
                if c == 'basket_pred':
                    continue
                if c in df_clean.columns:
                    A[c] = df_clean[c]
            A = A.loc[dates, adapter_cols].dropna()
            # make sure target aligns with features (drop rows that got NaN in A)
            y_true = df_clean.loc[A.index, target_col].values
            y_pred = adapter_models[t].predict(A.values)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        metrics[t] = dict(mae=mae, rmse=rmse, r2=r2, mape=mape)

        frames.append(pd.DataFrame({
            'date': dates,
            'ticker': t,
            'actual': y_true,
            'predicted': y_pred,
            'error': y_true - y_pred,
            'abs_error': np.abs(y_true - y_pred),
            'actual_oil_price': df_clean.loc[dates, 'wti_current'],
            'oil_prediction': df_clean.loc[dates, 'oil_price_prediction'] if 'oil_price_prediction' in df_clean.columns else np.nan
        }))

    results_df = pd.concat(frames, ignore_index=True)
    print("\n📈 Per-ticker metrics (Base+Adapters):")
    for t in metrics:
        m = metrics[t]
        print(f"{t}: MAE ${m['mae']:.2f} | RMSE ${m['rmse']:.2f} | R² {m['r2']:.3f} | MAPE {m['mape']:.2f}%")

    return results_df, metrics

def predict_future_with_adapters(base_model, df_features_all, feature_cols,
                                 tickers, anchor_tickers, anchor_idx_map,
                                 adapter_models, adapter_cols, horizon=21):
    """
    Predict one horizon ahead using latest available features.
    """
    import pandas as pd

    latest_row = df_features_all[feature_cols].dropna().iloc[-1:]
    if latest_row.empty:
        print("⚠️  Cannot predict - insufficient recent data")
        return None, None

    # 1) base preds (anchors) + basket
    base_pred = base_model.predict(latest_row)  # shape (1, len(anchors))
    anchor_pred = {t: float(base_pred[0, i]) for t, i in anchor_idx_map.items()}
    basket_val = float(sum(anchor_pred.values()) / len(anchor_pred))
    # 2) adapters for non-anchors
    adapter_input = latest_row.copy()
    # assemble adapter feature vector
    adapter_input = adapter_input.assign(basket_pred=basket_val)
    for c in adapter_cols:
        if c == 'basket_pred':
            continue
        if c not in adapter_input.columns and c in df_features_all.columns:
            adapter_input[c] = df_features_all[c].iloc[-1]

    last_date = pd.to_datetime(df_features_all.index[-1])
    pred_date = last_date + pd.tseries.offsets.BDay(horizon)

    out = {}
    # anchors from base
    for t in anchor_tickers:
        out[t] = anchor_pred[t]

    # non-anchors via adapters
    for t, adapter in adapter_models.items():
        Xa = adapter_input[adapter_cols].values
        out[t] = float(adapter.predict(Xa)[0])

    print(f"\n🔮 Future Stock Predictions (Base+Adapters) for {pred_date:%Y-%m-%d}:")
    for t in tickers:
        if t in out:
            print(f"  {t}: ${out[t]:.2f}")
    return out, pred_date

def run_stock_prediction_with_transfer(df_stocks, oil_model, oil_feature_cols, df_oil_features,
                                       anchor_tickers=("SLB","HAL"), all_tickers=("SLB","HAL"),
                                       horizon=21):
    """
    Same as run_stock_prediction, but:
      - trains a base model on anchors only
      - trains adapters for non-anchors
    """
    print("\n🚀 Starting Oil Service Stock Price Prediction (Transfer Learning)")
    print("=" * 70)
    print(f"Anchors: {list(anchor_tickers)} | All tickers: {list(all_tickers)}")

    # --- prep ---
    if 'Date' in df_stocks.columns:
        df_stocks = df_stocks.set_index('Date')
    df_stocks.index = pd.to_datetime(df_stocks.index)
    if df_stocks.index[0] > df_stocks.index[-1]:
        df_stocks = df_stocks.iloc[::-1]

    # Build features/targets for ALL tickers (so targets exist for adapters)
    df_feat_all, df_clean, X, y, feature_cols, target_cols = build_stock_dataset(
        df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=list(all_tickers), horizon=horizon
    )

    # --- Stage 1: Base on anchors ---
    base_model, anchor_cols, anchor_idx_map = train_base_model_on_anchors(
        X, y, all_tickers=list(all_tickers), anchor_tickers=list(anchor_tickers)
    )
    basket_pred, _anchor_pred_df = compute_basket_pred(base_model, X, list(anchor_tickers), anchor_idx_map)

    # --- Stage 2: Adapters for non-anchors ---
    adapter_models, adapter_cols = train_transfer_adapters(
        df_clean, feature_cols, target_cols, basket_pred,
        shared_extra_cols=[],  # you can pass more shared cols if you want
        all_tickers=list(all_tickers),
        anchor_tickers=list(anchor_tickers)
    )

    # --- Evaluate combined (base + adapters) ---
    results_df, metrics = evaluate_with_adapters(
        base_model, df_clean, feature_cols, target_cols,
        tickers=list(all_tickers),
        anchor_tickers=list(anchor_tickers),
        anchor_idx_map=anchor_idx_map,
        adapter_models=adapter_models, adapter_cols=adapter_cols
    )

    # Plot with your existing function
    plot_stock_results(results_df, metrics)

    # --- Predict future ---
    preds, pred_date = predict_future_with_adapters(
        base_model, df_feat_all, feature_cols,
        tickers=list(all_tickers),
        anchor_tickers=list(anchor_tickers),
        anchor_idx_map=anchor_idx_map,
        adapter_models=adapter_models, adapter_cols=adapter_cols,
        horizon=horizon
    )

    # Package models so you can reuse later
    transfer_bundle = {
        "base_model": base_model,
        "anchor_tickers": list(anchor_tickers),
        "anchor_idx_map": anchor_idx_map,
        "adapter_models": adapter_models,
        "adapter_cols": adapter_cols,
        "feature_cols": feature_cols
    }

    return transfer_bundle, results_df, metrics, preds

def save_transfer_bundle(transfer_bundle, path="models/transfer_bundle.pkl", extra_meta=None):
    """
    Persist the whole bundle (base model, adapters, columns, maps).
    Also writes a small sidecar JSON with light metadata for sanity checks.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(transfer_bundle, path)

    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "anchor_tickers": transfer_bundle.get("anchor_tickers", []),
        "adapter_tickers": sorted(list(transfer_bundle.get("adapter_models", {}).keys())),
        "n_adapter_features": len(transfer_bundle.get("adapter_cols", [])),
        "n_feature_cols": len(transfer_bundle.get("feature_cols", [])),
        "sklearn_hint": "Bundle contains sklearn models; restore with joblib.load().",
    }
    if extra_meta:
        meta.update(extra_meta)

    json_path = os.path.splitext(path)[0] + ".meta.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved transfer bundle → {path}")
    print(f"📝 Wrote metadata → {json_path}")


def load_transfer_bundle(path="models/transfer_bundle.pkl"):
    """
    Load the previously saved transfer bundle.
    """
    bundle = joblib.load(path)
    print(f"📦 Loaded transfer bundle from {path}")
    # quick sanity notes
    print("  Anchors:", bundle.get("anchor_tickers"))
    print("  Adapter tickers:", sorted(list(bundle.get("adapter_models", {}).keys())))
    print("  #feature_cols:", len(bundle.get("feature_cols", [])))
    print("  #adapter_cols:", len(bundle.get("adapter_cols", [])))
    return bundle

def predict_future_from_loaded_bundle(transfer_bundle, df_features_all, horizon=21):
    """
    Make a one-step-ahead prediction using a previously loaded transfer bundle.
    Assumes df_features_all already has the same feature engineering as during training.
    """
    base_model   = transfer_bundle["base_model"]
    feature_cols = transfer_bundle["feature_cols"]
    adapter_cols = transfer_bundle["adapter_cols"]
    anchor_ticks = transfer_bundle["anchor_tickers"]
    anchor_idx   = transfer_bundle["anchor_idx_map"]
    adapters     = transfer_bundle["adapter_models"]

    # 1) latest feature row (aligned to training feature_cols)
    latest_row = df_features_all[feature_cols].dropna().iloc[-1:]
    if latest_row.empty:
        print("⚠️  Cannot predict — no complete latest feature row.")
        return None, None

    # 2) anchors via base model
    base_pred = base_model.predict(latest_row)  # shape (1, len(anchors))
    anchor_pred = {t: float(base_pred[0, anchor_idx[t]]) for t in anchor_ticks}
    basket_val = float(sum(anchor_pred.values()) / max(1, len(anchor_pred)))

    # 3) assemble adapter input
    adapter_input = latest_row.copy()
    adapter_input = adapter_input.assign(basket_pred=basket_val)

    # ensure remaining adapter_cols exist (e.g., ['basket_pred', 'wti_current', ...])
    for c in adapter_cols:
        if c not in adapter_input.columns:
            # try to pull from df_features_all if it’s a shared feature
            if c in df_features_all.columns:
                adapter_input[c] = df_features_all[c].iloc[-1]
            elif c == "basket_pred":
                adapter_input[c] = basket_val
            else:
                # missing column is a hard stop; it means FE changed vs training
                raise ValueError(f"Missing adapter feature at inference: {c}")

    # 4) predict horizon timestamp
    last_date = pd.to_datetime(df_features_all.index[-1])
    pred_date = last_date + pd.tseries.offsets.BDay(horizon)

    # 5) build output for both anchors and any adapters
    out = {}
    # anchors
    for t in anchor_ticks:
        out[t] = anchor_pred[t]
    # non-anchors with adapters
    if adapters:
        Xa = adapter_input[adapter_cols].values
        for t, adapter in adapters.items():
            out[t] = float(adapter.predict(Xa)[0])

    print(f"\n🔮 Future Stock Predictions (Loaded Bundle) for {pred_date:%Y-%m-%d}:")
    for t, p in out.items():
        print(f"  {t}: ${p:.2f}")

    return out, pred_date

def main():
    ANCHORS = TICKERS
    ALL_TICKERS = TICKERS + EXTRA_TICKERS
    df_raw = None
    if NEW_DATA:
        df_raw = data.get_combined_oil_df()
    else:
        df_raw = data.get_data_from_csv()

    if 'Date' in df_raw.columns:
        df_raw = df_raw.set_index('Date')
    df_raw.index = pd.to_datetime(df_raw.index)

    df_stocks = data.add_stocks_to_df(df_raw, " ".join(ALL_TICKERS), NEW_DATA)
    df_raw = df_stocks.drop(columns=ALL_TICKERS)
    
    # First run oil price prediction
    print("\n📊 PHASE 1: Oil Price Prediction")
    print("-" * 50)
    oil_model, oil_results, oil_metrics, oil_prediction = run_oil_prediction(df_raw, verbose=True)

    # Get stock data (this will process df_raw internally)
    print("\n📈 PHASE 2: Stock Data Integration")  
    print("-" * 50)
    df_stocks = data.add_stocks_to_df(df_raw, " ".join(ALL_TICKERS), NEW_DATA)
    print(f"✅ Stock data integrated. Shape: {df_stocks.shape}")
    
    df_temp = df_raw.copy()
    
    df_oil_features = create_features(df_temp)
    df_oil_with_target = prepare_target_variable(df_oil_features, 21)
    exclude_cols = [
        'WTI ($/bbl)', 'target_monthly_avg', 'monthly_avg_current', 'year_month',
        'future_date', 'future_year_month', 'Brent ($/bbl)'
    ]
    oil_feature_cols = [col for col in df_oil_with_target.columns if col not in exclude_cols]
    print(f"✅ Oil feature columns extracted: {len(oil_feature_cols)} features")
    
    # Run stock prediction using oil model and data
    print(f"\n📈 PHASE 3: Stock Price Prediction")
    print("-" * 50)
    
    # stock_model, stock_results, stock_metrics, stock_prediction = run_stock_prediction(
    #     df_stocks, oil_model, oil_feature_cols, df_oil_features
    # )

    if TRAIN_NEW_MODEL:
        transfer_bundle, stock_results, stock_metrics, stock_prediction = run_stock_prediction_with_transfer(
            df_stocks, oil_model, oil_feature_cols, df_oil_features,
            anchor_tickers=ANCHORS, all_tickers=ALL_TICKERS, horizon=21
        )

        if SAVE_MODEL:
            save_transfer_bundle(
                transfer_bundle,
                path="models/transfer_bundle.pkl",
                extra_meta={"horizon_bd": 21, "code_tag": "v1.0"}
            )
    else:
        loaded = load_transfer_bundle("models/transfer_bundle.pkl")

        # Important: rebuild features the same way you did during training
        # (same create_stock_features pipeline, same column names!)
        df_feat_all = create_stock_features(
            df_stocks, oil_model, oil_feature_cols, df_oil_features, tickers=ALL_TICKERS
        )

        # predict without retraining
        preds, pred_date = predict_future_from_loaded_bundle(loaded, df_feat_all, horizon=21)
    
if __name__ == "__main__":
    main()