import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from data_helpers import data_tabulate as data

warnings.filterwarnings('ignore')

TICKERS = ['SLB', 'HAL']

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
    print("Generating future prediction...")
    preds, pred_date = predict_future_stock_price_multi(model, df_feat_all, feature_cols, tickers=TICKERS)

    return model, results_df, metrics, preds
    
if __name__ == "__main__":
    df_raw = data.get_data_from_csv()

    # First run oil price prediction
    print("\n📊 PHASE 1: Oil Price Prediction")
    print("-" * 50)
    oil_model, oil_results, oil_metrics, oil_prediction = run_oil_prediction(df_raw, verbose=True)

    # Get stock data (this will process df_raw internally)
    print("\n📈 PHASE 2: Stock Data Integration")  
    print("-" * 50)
    df_stocks = data.add_stocks_to_df(df_raw)
    print(f"✅ Stock data integrated. Shape: {df_stocks.shape}")
    
    # Extract feature columns from oil model training (avoid reprocessing)
    df_temp = df_raw.copy()
    if 'Date' in df_temp.columns:
        df_temp = df_temp.set_index('Date')
    df_temp.index = pd.to_datetime(df_temp.index)
    
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
    
    stock_model, stock_results, stock_metrics, stock_prediction = run_stock_prediction(
        df_stocks, oil_model, oil_feature_cols, df_oil_features
    )
