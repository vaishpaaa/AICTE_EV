
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

class Config:
    
    DATA_PATH = 'Electric_Vehicle_Population_By_County.csv'
    OUTPUT_DIR = 'project_outputs' 
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'final_random_forest_model.pkl')
    PREPROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'preprocessed_ev_data.csv')
    
    TARGET_VARIABLE = 'Electric Vehicle (EV) Total'
    LAG_FEATURES_TO_CREATE = [1, 2, 3]
    ROLLING_WINDOW_SIZE = 3

    RANDOM_FOREST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1  
    }
    
    
    TEST_SET_MONTHS = 12  
    
    
    NUM_TOP_COUNTIES_TO_PLOT = 3 
def set_professional_plot_style():
    """Sets a clean and professional style for all matplotlib plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (14, 7),
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

def load_and_clean_data(file_path):
    """
    Loads the raw EV population data, standardizes column names,
    and performs initial data type conversions.
    """
    print(f"--> Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
        return None

    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['county', 'date']).reset_index(drop=True)
    
    print("--> Data loading and initial cleaning complete.")
    return df

def engineer_features(df, target_col, lags, window_size):
    """
    Creates time-series and categorical features for the model.
    This is a critical step for forecasting performance.
    """
    print("--> Starting feature engineering...")
    
    df_featured = df.copy()

    df_featured['year'] = df_featured['date'].dt.year
    df_featured['month'] = df_featured['date'].dt.month
    df_featured['days_since_start'] = (df_featured['date'] - df_featured['date'].min()).dt.days

    df_featured['county_encoded'] = LabelEncoder().fit_transform(df_featured['county'])

    for lag in lags:
        df_featured[f'ev_total_lag_{lag}'] = df_featured.groupby('county')[target_col].shift(lag)

    df_featured[f'ev_total_roll_mean_{window_size}'] = df_featured.groupby('county')[target_col].transform(
        lambda x: x.shift(1).rolling(window=window_size).mean()
    )

    df_featured = df_featured.dropna().reset_index(drop=True)
    
    print("--> Feature engineering complete.")
    return df_featured

def split_data_for_forecasting(df, test_months):
    """
    Splits the data into training and testing sets based on a time cutoff.
    This simulates a real-world forecasting scenario.
    """
    print(f"--> Splitting data: using the last {test_months} months for the test set.")
    
    split_date = df['date'].max() - pd.DateOffset(months=test_months)
    
    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date]

    features = [
        'year', 'month', 'days_since_start', 'county_encoded',
        'ev_total_lag_1', 'ev_total_lag_2', 'ev_total_lag_3',
        'ev_total_roll_mean_3'
    ]
    target = 'electric_vehicle_(ev)_total'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"    Training set contains {len(X_train)} records.")
    print(f"    Test set contains {len(X_test)} records.")
    
    return X_train, y_train, X_test, y_test, test_df 

def train_and_save_model(X_train, y_train, params, model_path):
    """Trains the Random Forest model and saves it to a file."""
    print("--> Training the Random Forest model...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("--> Model training complete.")
    
    
    joblib.dump(model, model_path)
    print(f"--> Model successfully saved to '{model_path}'")
    return model

def evaluate_model_and_visualize(model, X_test, y_test, original_test_df, output_dir):
    """
    Evaluates the model's performance and creates key visualizations
    to understand its predictions and feature importances.
    """
    print("--> Evaluating model and generating visualizations...")
    
    
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} vehicles")
    print("--------------------------------\n")

    feature_importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure()
    sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis_r')
    plt.title('Key Drivers of EV Adoption (Feature Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    importance_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(importance_path)
    print(f"--> Saved feature importance plot to '{importance_path}'")

    results_df = original_test_df.copy()
    results_df['predicted_ev_total'] = y_pred

    latest_date = results_df['date'].max()
    top_counties = results_df[results_df['date'] == latest_date]\
        .nlargest(Config.NUM_TOP_COUNTIES_TO_PLOT, 'electric_vehicle_(ev)_total')['county'].tolist()

    print(f"--> Generating forecast plots for top {len(top_counties)} counties: {', '.join(top_counties)}")
    for county in top_counties:
        plt.figure()
        plot_data = results_df[results_df['county'] == county]
        
        plt.plot(plot_data['date'], plot_data['electric_vehicle_(ev)_total'], label='Actual EV Count', marker='o', linestyle='-', color='dodgerblue')
        plt.plot(plot_data['date'], plot_data['predicted_ev_total'], label='Random Forest Forecast', marker='x', linestyle='--', color='red')
        
        plt.title(f'EV Adoption Forecast vs. Actual - {county.title()} County')
        plt.ylabel('Total Electric Vehicles')
        plt.xlabel('Date')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        forecast_path = os.path.join(output_dir, f"forecast_plot_{county.lower()}.png")
        plt.savefig(forecast_path)
        print(f"    - Saved forecast plot for {county.title()} County.")
        
    plt.close('all') 

def main():
    """Main function to orchestrate the entire forecasting pipeline."""
    print("--- Running Final EV Forecasting Project Script ---")

    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    set_professional_plot_style()


    df = load_and_clean_data(config.DATA_PATH)
    if df is None:
        return 


    target_col_name = config.TARGET_VARIABLE.lower().replace(' ', '_').replace('(', '').replace(')', '')
    df_featured = engineer_features(df, target_col_name, config.LAG_FEATURES_TO_CREATE, config.ROLLING_WINDOW_SIZE)
    df_featured.to_csv(config.PREPROCESSED_DATA_PATH, index=False)
    print(f"--> Saved preprocessed data to '{config.PREPROCESSED_DATA_PATH}'")

    X_train, y_train, X_test, y_test, test_df_original = split_data_for_forecasting(df_featured, config.TEST_SET_MONTHS)
    
    model = train_and_save_model(X_train, y_train, config.RANDOM_FOREST_PARAMS, config.MODEL_PATH)

    evaluate_model_and_visualize(model, X_test, y_test, test_df_original, config.OUTPUT_DIR)
    
    print("\n--- ✅ Script finished successfully! All outputs are in the 'project_outputs' folder. ---")


if __name__ == '__main__':
    main()
