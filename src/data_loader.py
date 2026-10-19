"""
Self-contained Data Loader for Siemens Process Energy Optimization Engine.
Dataset sourced from Kaggle.com (Siemens Industrial Predictive Maintenance & Energy Telemetry).
"""
import os
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telemetry_sample.parquet")
CSV_BACKUP_PATH = os.path.join(PROJECT_ROOT, "data", "telemetry_sample.csv")

def load_data(nrows=None, sample_step=1):
    """
    Loads telemetry data with datetime parsing and optional row slicing.
    
    Args:
        nrows (int, optional): Max number of rows to load.
        sample_step (int, optional): Step size for downsampling (e.g. 2 takes every 2nd row).
        
    Returns:
        pd.DataFrame: Loaded telemetry dataset.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
    elif os.path.exists(CSV_BACKUP_PATH):
        df = pd.read_csv(CSV_BACKUP_PATH, nrows=nrows)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please ensure telemetry_sample.parquet exists in the data/ directory."
        )
    
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    if nrows is not None:
        df = df.iloc[:nrows]
    if sample_step > 1:
        df = df.iloc[::sample_step]
        
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Data successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")
