import pandas as pd
import numpy as np

def prepare_bakery_data():
    """Load and merge the bakery features and targets data."""
    print("Loading bakery data...")
    
    # Load the parquet files
    features_df = pd.read_parquet('data/Bakery_features.parquet')
    targets_df = pd.read_parquet('data/Bakery_targets.parquet')
    
    # Merge on common columns
    merged_df = pd.merge(
        features_df, 
        targets_df[['bdID', 'target']], 
        on='bdID', 
        how='inner'
    )
    
    print(f"Original features shape: {features_df.shape}")
    print(f"Original targets shape: {targets_df.shape}")
    print(f"Merged data shape: {merged_df.shape}")
    
    # Clean and prepare the data
    # Remove rows where target is missing or not_for_sale = 1
    merged_df = merged_df[
        (merged_df['target'].notna()) & 
        (merged_df['not_for_sale'] == 0)
    ].copy()
    
    # Sort by date
    merged_df = merged_df.sort_values(['date', 'storeID', 'skuID']).reset_index(drop=True)
    
    # Select relevant columns for modeling
    feature_cols = [
        'public_holiday_0', 'school_holiday_0', 'rain_mm_0', 
        'temp_avg_0', 'temp_max_0', 'temp_min_0', 'promotion_0', 'lag_target_1'
    ]
    
    # Create the final dataframe
    model_df = merged_df[['date', 'companyID', 'storeID', 'skuID', 'target'] + feature_cols].copy()
    
    # Fill missing values
    for col in feature_cols:
        if model_df[col].isna().any():
            model_df[col] = model_df[col].fillna(model_df[col].median())
    
    # Set date as index
    model_df = model_df.set_index('date')
    
    print(f"Final prepared data shape: {model_df.shape}")
    print(f"Date range: {model_df.index.min()} to {model_df.index.max()}")
    print(f"Number of unique stores: {model_df['storeID'].nunique()}")
    print(f"Number of unique SKUs: {model_df['skuID'].nunique()}")
    
    return model_df

if __name__ == "__main__":
    df = prepare_bakery_data()
    
    # Save the prepared data
    df.to_csv('bakery_data_merged.csv')
    print("✅ Prepared data saved to 'bakery_data_merged.csv'")
    
    # Show sample of the data
    print("\nSample of prepared data:")
    print(df.head(10))
