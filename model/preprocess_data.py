import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_dataset(input_file, output_file):
    """Preprocess the sales dataset"""
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract time features
    print("Extracting time features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Handle outliers in sales
    print("Handling outliers...")
    Q1 = df['sales'].quantile(0.25)
    Q3 = df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    df['sales_cleaned'] = df['sales'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    # Scale numerical features
    print("Scaling features...")
    scaler = StandardScaler()
    numerical_features = ['store', 'item']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save scaler for future use
    print("Saving preprocessing artifacts...")
    os.makedirs('../model/preprocessors', exist_ok=True)
    joblib.dump(scaler, '../model/preprocessors/scaler.pkl')
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    df.to_csv(output_file, index=False)
    
    print("Preprocessing completed!")
    return df

if __name__ == "__main__":
    input_file = '../data/train.csv'
    output_file = '../data/preprocessed_train.csv'
    preprocess_dataset(input_file, output_file) 