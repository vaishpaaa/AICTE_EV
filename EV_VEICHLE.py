# --------------------------------------------------------------------------
# 🏆 Advanced EV Adoption Forecasting Model
# --------------------------------------------------------------------------
# This script loads EV population data, performs feature engineering,
# tunes an XGBoost model, and saves the final model for future use.
# --------------------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main function to run the entire data processing and modeling pipeline."""
    
    print("--- Starting EV Forecasting Script ---")

    # ---------------------------------
    # 1. Data Loading and Cleaning
    # ---------------------------------
    print("\n[Step 1/6] Loading and cleaning data...")
    try:
        df = pd.read_csv('Electric_Vehicle_Population_By_County.csv')
    except FileNotFoundError:
        print("Error: 'Electric_Vehicle_Population_By_County.csv' not found.")
        print("Please make sure the dataset is in the same directory as this script.")
        return

    # Convert 'Date' column to datetime and clean numeric columns
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Non-Electric Vehicle Total', 'Total Vehicles']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
    df.dropna(subset=['County'], inplace=True)
    print("Data loaded and cleaned successfully.")

    # ---------------------------------
    # 2. Feature Engineering
    # ---------------------------------
    print("\n[Step 2/6] Performing feature engineering...")
    # Sort data chronologically for time-series operations
    df = df.sort_values(['County', 'Date']).reset_index(drop=True)

    # Create time-based, lag, and rolling window features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['months_since_start'] = (df['year'] - df['year'].min()) * 12 + df['month']
    
    grouped = df.groupby('County')['Electric Vehicle (EV) Total']
    df['ev_total_lag_1'] = grouped.shift(1)
    df['ev_total_roll_mean_3'] = grouped.shift(1).rolling(window=3, min_periods=1).mean()

    # Encode categorical 'County' feature
    county_encoder = LabelEncoder()
    df['county_encoded'] = county_encoder.fit_transform(df['County'])
    df.dropna(inplace=True)
    print("Feature engineering complete.")

    # ---------------------------------
    # 3. Model Preparation
    # ---------------------------------
    print("\n[Step 3/6] Preparing data for the model...")
    features = [
        'year', 'month', 'months_since_start', 'county_encoded', 
        'ev_total_lag_1', 'ev_total_roll_mean_3'
    ]
    target = 'Electric Vehicle (EV) Total'

    X = df[features]
    y = df[target]

    # Time-based split: Train on data before 2023, test on 2023+
    split_date = pd.to_datetime('2023-01-01')
    train_mask = (df['Date'] < split_date)
    test_mask = (df['Date'] >= split_date)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Training set size: {X_train.shape[0]} samples. Test set size: {X_test.shape[0]} samples.")

    # ---------------------------------
    # 4. Hyperparameter Tuning
    # ---------------------------------
    print("\n[Step 4/6] Tuning XGBoost model (this may take a few minutes)...")
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_reg, param_distributions=param_grid, n_iter=50, 
        cv=tscv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print("Hyperparameter tuning complete. Best model found.")
    print(f"Best parameters: {random_search.best_params_}")

    # ---------------------------------
    # 5. Model Evaluation & Visualization
    # ---------------------------------
    print("\n[Step 5/6] Evaluating model and creating visualizations...")
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- Model Performance ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Create and save feature importance plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(best_model, ax=ax, height=0.8)
    ax.set_title("Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nSaved 'feature_importance.png'")

    # Create and save forecast plots for sample counties
    results_df = df[test_mask].copy()
    results_df['Predicted EV Total'] = y_pred
    sample_counties = ['King', 'Snohomish', 'Pierce']

    for county in sample_counties:
        plt.figure(figsize=(15, 6))
        plot_data = results_df[results_df['County'] == county]
        plt.plot(plot_data['Date'], plot_data['Electric Vehicle (EV) Total'], label='Actual Values', marker='.', linestyle='-')
        plt.plot(plot_data['Date'], plot_data['Predicted EV Total'], label='XGBoost Forecast', marker='.', linestyle='--')
        plt.title(f'EV Adoption Forecast vs. Actual for {county} County', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Total Electric Vehicles')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"forecast_plot_{county}.png")
        print(f"Saved 'forecast_plot_{county}.png'")
    
    plt.close('all') # Close all plot figures

    # ---------------------------------
    # 6. Save Final Model
    # ---------------------------------
    print("\n[Step 6/6] Saving the final trained model...")
    joblib.dump(best_model, 'xgb_model.pkl')
    joblib.dump(county_encoder, 'county_encoder.pkl')
    print("✅ Model 'xgb_model.pkl' and 'county_encoder.pkl' saved successfully.")
    
    print("\n--- Script finished ---")

if __name__ == '__main__':
    main()
