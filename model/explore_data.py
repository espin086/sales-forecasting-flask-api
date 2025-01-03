import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def explore_dataset():
    """Explore and analyze the sales dataset"""
    print("Loading dataset...")
    df = pd.read_csv('../data/train.csv')
    
    # Basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nDataset Shape:", df.shape)
    
    print("\nSample Data:")
    print(df.head())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Unique values in each column
    print("\nUnique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Time-based analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Create visualizations directory
    os.makedirs('../visualizations', exist_ok=True)
    
    # Sales distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sales'], bins=50)
    plt.title('Distribution of Sales')
    plt.savefig('../visualizations/sales_distribution.png')
    plt.close()
    
    # Sales by day of week
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dayofweek', y='sales', data=df)
    plt.title('Sales by Day of Week')
    plt.savefig('../visualizations/sales_by_dayofweek.png')
    plt.close()
    
    # Average sales by month
    monthly_sales = df.groupby('month')['sales'].mean()
    plt.figure(figsize=(10, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Average Sales by Month')
    plt.savefig('../visualizations/avg_sales_by_month.png')
    plt.close()
    
    # Store analysis
    store_stats = df.groupby('store')['sales'].agg(['mean', 'std', 'count'])
    print("\nStore Statistics:")
    print(store_stats.head())
    
    # Item analysis
    item_stats = df.groupby('item')['sales'].agg(['mean', 'std', 'count'])
    print("\nItem Statistics:")
    print(item_stats.head())
    
    # Correlation analysis
    correlation = df[['sales', 'store', 'item', 'year', 'month', 'day', 'dayofweek']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.savefig('../visualizations/correlation_matrix.png')
    plt.close()
    
    # Save summary statistics
    with open('../visualizations/summary_statistics.txt', 'w') as f:
        f.write("Dataset Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Basic Statistics:\n")
        f.write(str(df.describe()) + "\n\n")
        
        f.write("Store Statistics:\n")
        f.write(str(store_stats.describe()) + "\n\n")
        
        f.write("Item Statistics:\n")
        f.write(str(item_stats.describe()) + "\n\n")
        
        f.write("Correlation Matrix:\n")
        f.write(str(correlation) + "\n")

if __name__ == "__main__":
    explore_dataset() 