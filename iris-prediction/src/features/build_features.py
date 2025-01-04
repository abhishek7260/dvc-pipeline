import pandas as pd
import numpy as np
import os

# File paths
train_file = "./data/raw/train.csv"
test_file = "./data/raw/test.csv"

# Load data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Function to remove outliers using Z-score
def remove_outliers_with_zscore(df, columns=None, threshold=3):
    """
    Removes rows with outliers in the specified numeric columns based on Z-score.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numeric columns to check for outliers. If None, all numeric columns are considered.
        threshold (float): The Z-score threshold beyond which data points are considered outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in columns:
        # Calculate Z-scores
        df['zscore'] = (df[col] - df[col].mean()) / df[col].std()
        # Filter out rows with Z-scores beyond the threshold
        df = df[abs(df['zscore']) <= threshold]
        # Drop the temporary 'zscore' column
        df = df.drop(columns=['zscore'])
    return df

# Process training and testing data
processed_train_data = remove_outliers_with_zscore(train_data)
processed_test_data = remove_outliers_with_zscore(test_data)

# Save processed data
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)
processed_train_data.to_csv(os.path.join(data_path, "processed_train_zscore.csv"), index=False)
processed_test_data.to_csv(os.path.join(data_path, "processed_test_zcore.csv"), index=False)
