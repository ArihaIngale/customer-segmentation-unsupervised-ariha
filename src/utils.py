import pandas as pd
import os

def load_csv(path):
    """Load CSV file"""
    return pd.read_csv(path)

def save_csv(df, path):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def print_shape(df, name="Data"):
    print(f"{name} Shape: {df.shape}")