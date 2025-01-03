import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def train_model():
    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/preprocessed_train.csv')
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature engineering
    print("Performing feature engineering...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Prepare features and target
    features = [
        'store', 'item', 'year', 'month', 'day', 
        'dayofweek', 'is_weekend', 'is_month_start', 
        'is_month_end'
    ]
    target = 'sales'
    
    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        df[features], 
        df[target],
        test_size=0.2,
        random_state=42
    )
    
    # Create LightGBM datasets
    print("Creating LightGBM datasets...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Parameters from the notebook with adjustments
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mape',
        'num_leaves': 64,
        'max_depth': 5,
        'learning_rate': 0.1,  # Reduced from 0.2
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,  # Reduced regularization
        'lambda_l2': 0.1,
        'min_child_weight': 1.0,  # Adjusted
        'min_split_gain': 0.01,  # Adjusted
        'min_data_in_leaf': 20,  # Added
        'verbose': -1  # Reduced verbosity
    }
    
    print("Training model...")
    # Train model
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,  # Reduced number of rounds
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print("Saving model and features...")
    # Save the model
    joblib.dump(model, 'sales_forecast_model.pkl')
    
    # Save feature list for API reference
    joblib.dump(features, 'feature_list.pkl')
    
    print("Training completed!")
    
if __name__ == '__main__':
    train_model() 