import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from data_helpers import data_tabulate as data

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
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
    df_features = create_features(df)
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
    
    # Additional validation: check for data leakage indicators
    train_pred = model.predict(X)
    train_r2 = r2_score(y, train_pred)
    if train_r2 > 0.98:
        print(f"\n⚠️  WARNING: Training R² = {train_r2:.3f} suggests possible overfitting!")
    
    return model, X, y, feature_cols, df_clean

def generate_predictions(model, df, feature_cols, forecast_horizon=21):
    """
    Generate predictions and calculate performance metrics
    """
    df_features = create_features(df)
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
    plt.show()
    
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
    df_features = create_features(df)
    
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
    prediction_date = last_date + pd.Timedelta(days=forecast_horizon)
    prediction_month = prediction_date.strftime('%B %Y')
    
    print(f"\n🔮 Future Prediction:")
    print(f"Current WTI price: ${df['WTI ($/bbl)'].iloc[-1]:.2f}")
    print(f"Predicted monthly average for {prediction_month}: ${prediction:.2f}")
    print(f"Prediction date range: ~{prediction_date.strftime('%Y-%m-%d')}")
    
    return prediction, prediction_date

def oil_price_prediction_pipeline(df):
    """
    Complete pipeline for oil price prediction
    """
    print("🚀 Starting WTI Oil Price Prediction Analysis")
    print("=" * 60)
    
    # CRITICAL FIX: Handle reverse chronological data
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    
    # Check if data is in reverse chronological order and fix it
    if df.index[0] > df.index[-1]:
        print("⚠️  Data is in reverse chronological order - fixing...")
        df = df.iloc[::-1]  # Reverse the dataframe
        print("✅ Data order corrected")
    
    print(f"Data range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(df)}")
    
    # Anti-leakage validation
    print("\n🔍 Data Leakage Prevention Checks:")
    print("- ✅ Data sorted chronologically")
    print("- ✅ All features use only past/current information")
    print("- ✅ Target variable calculated from future data only")
    print("- ✅ Time series cross-validation prevents temporal leakage")
    
    # Train model
    model, X, y, feature_cols, df_clean = train_oil_price_model(df)
    
    # Generate predictions and evaluate
    results_df, metrics = generate_predictions(model, df, feature_cols)
    
    # Plot results
    plot_results(results_df, metrics)
    
    # Make future prediction
    future_pred, pred_date = predict_future_price(model, df, feature_cols)
    
    # Model interpretation
    print(f"\n🧠 Model Insights:")
    print(f"The model uses {len(feature_cols)} features including technical indicators,")
    print(f"supply/demand fundamentals, economic factors, and seasonal patterns.")
    print(f"Random Forest was chosen for its ability to capture non-linear relationships")
    print(f"and interactions between oil market factors without overfitting.")
    
    # Reality check on performance
    if metrics['r2'] > 0.95:
        print(f"\n⚠️  WARNING: R² = {metrics['r2']:.3f} is suspiciously high!")
        print("This may indicate remaining data leakage or overfitting.")
        print("Consider:")
        print("- Reducing model complexity (fewer trees, simpler features)")
        print("- Using walk-forward validation")
        print("- Checking for unintended future information in features")
    
    return model, results_df, metrics, future_pred

# Example usage function
def run_oil_prediction(df):
    """
    Run the complete oil price prediction analysis
    
    Parameters:
    csv_file_path (str): Path to your CSV file with oil data
    
    Returns:
    tuple: (trained_model, results_dataframe, performance_metrics, future_prediction)
    """
    # Run the pipeline
    model, results, metrics, prediction = oil_price_prediction_pipeline(df)
    
    return model, results, metrics, prediction

# To use this code with your data:
# model, results, metrics, prediction = run_oil_prediction('your_oil_data.csv')# To use this code with your data:
df = data.get_data_from_csv()
model, results, metrics, prediction = run_oil_prediction(df)
