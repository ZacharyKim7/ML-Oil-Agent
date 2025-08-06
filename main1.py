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

def create_features(df):
    """
    Create comprehensive features for oil price prediction including:
    - Technical indicators (moving averages, volatility, momentum)
    - Fundamental factors (supply/demand ratios, inventory levels)
    - Economic indicators (currency, inflation, growth)
    - Seasonal patterns
    """
    df = df.copy()
    
    # Technical indicators for WTI
    df['wti_ma_5'] = df['WTI ($/bbl)'].rolling(5).mean()
    df['wti_ma_21'] = df['WTI ($/bbl)'].rolling(21).mean()
    df['wti_ma_63'] = df['WTI ($/bbl)'].rolling(63).mean()
    df['wti_volatility_21'] = df['WTI ($/bbl)'].rolling(21).std()
    df['wti_rsi'] = calculate_rsi(df['WTI ($/bbl)'], 14)
    df['wti_momentum_5'] = df['WTI ($/bbl)'] - df['WTI ($/bbl)'].shift(5)
    df['wti_momentum_21'] = df['WTI ($/bbl)'] - df['WTI ($/bbl)'].shift(21)
    
    # Brent-WTI spread (important for arbitrage)
    df['brent_wti_spread'] = df['Brent ($/bbl)'] - df['WTI ($/bbl)']
    df['brent_wti_spread_ma'] = df['brent_wti_spread'].rolling(21).mean()
    
    # Supply/Demand fundamentals
    df['total_supply'] = df['OPEC P (tbbl/d)'] + df['NOPEC P (tbbl/d)']
    df['total_demand'] = df['OECD C (tbbl/d)'] + df['China C (tbbl/d)'] + df['India C (tbbl/d)']
    df['supply_demand_ratio'] = df['total_supply'] / df['total_demand']
    df['supply_demand_balance'] = df['total_supply'] - df['total_demand']
    
    # Inventory indicators
    df['total_stocks'] = df['OECD S (mbbl)'] + df['USA S (mbbl)']
    df['stocks_ma_21'] = df['total_stocks'].rolling(21).mean()
    df['stocks_change'] = df['total_stocks'] - df['total_stocks'].shift(21)
    
    # US-specific indicators
    df['us_production_proxy'] = df['USA rigs (m)']  # Rig count as production proxy
    df['us_net_imports_ma'] = df['US net imports (tbbl/d)'].rolling(21).mean()
    
    # Economic indicators
    df['cpi_change'] = df['CPI'].pct_change(21)  # Inflation proxy
    df['gdp_momentum'] = df['GDP (yoy%)'].diff()
    df['usd_strength'] = (1/df['USD-GBP'] + df['USD-YEN']/100) / 2  # USD strength index
    df['usd_change'] = df['usd_strength'].pct_change(21)
    
    # Seasonal patterns
    df['month'] = pd.to_datetime(df.index).month
    df['quarter'] = pd.to_datetime(df.index).quarter
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)  # Heating season
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)   # Driving season
    
    # Lagged features (important for time series)
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
    """
    df = df.copy()
    
    # Calculate monthly averages
    df['year_month'] = pd.to_datetime(df.index).to_period('M')
    monthly_avg = df.groupby('year_month')['WTI ($/bbl)'].mean()
    
    # Map back to daily data
    df['monthly_avg_current'] = df['year_month'].map(monthly_avg)
    
    # Shift forward by forecast horizon to create target
    df['target_monthly_avg'] = df['monthly_avg_current'].shift(-forecast_horizon)
    
    return df

def train_oil_price_model(df, forecast_horizon=21):
    """
    Train Random Forest model to predict monthly average WTI oil price
    """
    print("🛢️  Training WTI Oil Price Prediction Model")
    print("=" * 50)
    
    # Feature engineering
    df_features = create_features(df)
    df_with_target = prepare_target_variable(df_features, forecast_horizon)
    
    # Select feature columns (exclude target and non-predictive columns)
    feature_cols = [col for col in df_with_target.columns if col not in [
        'WTI ($/bbl)', 'target_monthly_avg', 'monthly_avg_current', 'year_month'
    ]]
    
    # Remove rows with NaN values
    df_clean = df_with_target.dropna()
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    
    if len(df_clean) < 100:
        print("⚠️  Warning: Limited data available after feature engineering")
        print("Consider reducing the number of rolling window features")
    
    X = df_clean[feature_cols]
    y = df_clean['target_monthly_avg']
    
    # Use time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
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
    
    # Ensure Date column is the index
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    
    print(f"Data range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(df)}")
    
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
    
    return model, results_df, metrics, future_pred

# Example usage function
def run_oil_prediction(df):
    """
    Run the complete oil price prediction analysis
    
    Parameters:
    df (pandas)
    
    Returns:
    tuple: (trained_model, results_dataframe, performance_metrics, future_prediction)
    """
    # Load data
    df = df.copy()
    
    # Run the pipeline
    model, results, metrics, prediction = oil_price_prediction_pipeline(df)
    
    return model, results, metrics, prediction

# To use this code with your data:
df = data.get_data_from_csv()
model, results, metrics, prediction = run_oil_prediction(df)