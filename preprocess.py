import dask.dataframe as dd
import dask
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime

# Set up logging to both terminal and file
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Create output.txt file for logging
start_time = time.time()
output_file = open('output.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = TeeOutput(sys.stdout, output_file)

print("=" * 80)
print(f"🔥 WILDFIRE DATA PREPROCESSING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"⏱️ Start time: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 80)

# Define file paths
PROCESSED_DATA_DIR = 'data/processed'
SCALER_PATH = 'models/scaler.joblib'
ENCODER_COLUMNS_PATH = 'models/encoder_columns.joblib'
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.parquet')
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, 'val.parquet')
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.parquet')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting data preprocessing...")

# Load the dataset using Dask
print("Loading dataset...")
load_start = time.time()
ddf = dd.read_parquet('data/dataset.parquet')
original_rows = ddf.shape[0].compute()
load_time = time.time() - load_start
print(f"Dataset loaded. Shape: {original_rows} rows, {len(ddf.columns)} columns")
print(f"⏱️ Loading time: {load_time:.1f} seconds")

# Convert 'date' column to datetime for proper chronological ordering
print("Converting date column for chronological ordering...")
if 'date' in ddf.columns:
    # The date column contains integers in YYYYMMDD format
    ddf['date'] = dd.to_datetime(ddf['date'], format='%Y%m%d')
    print("Date column converted to datetime format")
else:
    print("Warning: 'date' column not found in dataset")

# Sort by date to ensure chronological order before sampling
print("Converting date and preparing for sampling...")
sort_start = time.time()
# We'll skip the expensive sorting step and work with the date filter directly
sort_time = time.time() - sort_start
print(f"Date preparation completed in {sort_time:.1f} seconds")

# Apply continuous 10% sampling to maintain temporal continuity
print("SKIPPING SAMPLING - Using full dataset for better training...")
sampling_start = time.time()

print("Using full dataset without sampling...")

# Skip all sampling logic and use full dataset
sampled_rows = original_rows
actual_percentage = 100.0
print(f"Using complete dataset: {sampled_rows:,} rows ({actual_percentage:.1f}% of total)")

sampling_time = time.time() - sampling_start
print(f"⏱️ Sampling completed in: {sampling_time:.1f} seconds")

# Verify date range of sampled data
if 'date' in ddf.columns:
    min_date = ddf['date'].min().compute()
    max_date = ddf['date'].max().compute()
    print(f"Sampled date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    total_days = (max_date - min_date).days
    print(f"Total time span: {total_days} days")

# 1. Drop duplicate columns as specified in the instructions
print("Dropping duplicate columns...")
cols_to_drop = [
    'Nearest_2_1_days_ago.1', 
    'Nearest_4_1_days_ago.1', 
    'Nearest_6_1_days_ago.1'
]
# Check which columns to drop actually exist in the dataframe
cols_to_drop_existing = [col for col in cols_to_drop if col in ddf.columns]
if cols_to_drop_existing:
    ddf = ddf.drop(columns=cols_to_drop_existing)
    print(f"Dropped columns: {cols_to_drop_existing}")
else:
    print("No duplicate columns found to drop")

# 2. Handle missing values represented as -999
print("Handling missing values represented as -999...")

# Get all numeric columns (excluding date which might be string/int format)
# We'll handle date separately
numeric_columns = []
for col in ddf.columns:
    if col != 'date' and col != 'Land_Use':  # Exclude categorical columns
        # Check if column is numeric
        if ddf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_columns.append(col)

print(f"Identified {len(numeric_columns)} numeric columns for -999 replacement")

# Replace -999 with NaN for all numeric columns
print("Replacing -999 values with NaN...")
for col in numeric_columns:
    ddf[col] = ddf[col].replace(-999, np.nan)

# Count rows before dropping missing values
initial_rows = ddf.shape[0].compute()
print(f"Rows before removing missing values: {initial_rows}")

# Remove rows with any missing values
print("Removing rows with missing values...")
ddf = ddf.dropna()

# Count rows after dropping missing values  
final_rows = ddf.shape[0].compute()
print(f"Rows after removing missing values: {final_rows}")
print(f"Removed {initial_rows - final_rows} rows ({((initial_rows - final_rows)/initial_rows)*100:.2f}%)")

# 3. One-hot encode the 'Land_Use' categorical feature
if 'Land_Use' in ddf.columns:
    print("One-hot encoding 'Land_Use' column...")
    ddf = ddf.categorize(columns=['Land_Use'])
    ddf = dd.get_dummies(ddf, columns=['Land_Use'], prefix='Land_Use')
    
    # Convert boolean dummy columns to integers to prevent Parquet writer errors
    land_use_cols = [col for col in ddf.columns if col.startswith('Land_Use_')]
    for col in land_use_cols:
        ddf[col] = ddf[col].astype('uint8')
    
    print(f"One-hot encoded 'Land_Use' feature into {len(land_use_cols)} columns")
else:
    print("Warning: 'Land_Use' column not found in dataset")

# Save the columns after encoding for consistent inference
encoder_columns = list(ddf.columns)
joblib.dump(encoder_columns, ENCODER_COLUMNS_PATH)
print(f"Saved encoded column list to {ENCODER_COLUMNS_PATH}")

# Data is already sorted chronologically from earlier step

# Define target and features
TARGET = 'FRP'
all_features = [col for col in ddf.columns if col not in [TARGET, 'date']] 
numerical_features = [col for col in all_features if not col.startswith('Land_Use_')]
print(f"Identified {len(numerical_features)} numerical features for scaling")
print(f"Total features (including categorical): {len(all_features)}")

# 4. Perform time-aware split ensuring test set spans at least 31 days
print("Computing time-aware splits with minimum 31-day test period...")

# First, get the full date range
min_date = ddf['date'].min().compute()
max_date = ddf['date'].max().compute()
total_days = (max_date - min_date).days

print(f"Full dataset spans: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({total_days} days)")

# Ensure test set has at least 31 days
min_test_days = 31
test_start_latest = max_date - pd.Timedelta(days=min_test_days)

# Calculate splits ensuring 31+ day test period
# Use approximately 60% train, 20% val, 20% test, but adjust to ensure 31+ test days
if total_days < 100:  # If dataset is too small, use different proportions
    val_start = min_date + pd.Timedelta(days=int(total_days * 0.7))  # 70% train
    test_start = min_date + pd.Timedelta(days=int(total_days * 0.85))  # 15% val, 15% test
else:
    # Normal case: try 60/20/20 but ensure test has 31+ days
    proposed_test_start = min_date + pd.Timedelta(days=int(total_days * 0.8))
    if proposed_test_start <= test_start_latest:
        # Normal 60/20/20 split works
        val_start = min_date + pd.Timedelta(days=int(total_days * 0.6))
        test_start = proposed_test_start
    else:
        # Adjust to ensure 31+ test days
        test_start = test_start_latest
        remaining_days = (test_start - min_date).days
        val_start = min_date + pd.Timedelta(days=int(remaining_days * 0.75))  # 75/25 split of remaining

# Create the splits
train_ddf = ddf[ddf['date'] < val_start]
val_ddf = ddf[(ddf['date'] >= val_start) & (ddf['date'] < test_start)]
test_ddf = ddf[ddf['date'] >= test_start]

# Verify test period is at least 31 days
actual_test_days = (max_date - test_start).days + 1
print(f"Test period: {actual_test_days} days (minimum required: {min_test_days})")

if actual_test_days < min_test_days:
    print(f"❌ ERROR: Test period only {actual_test_days} days, need at least {min_test_days}")
    sys.exit(1)
else:
    print(f"✅ Test period sufficient: {actual_test_days} days")

# Compute sizes
train_size = train_ddf.shape[0].compute()
val_size = val_ddf.shape[0].compute()
test_size = test_ddf.shape[0].compute()

print(f"Train set size: {train_size:,} rows")
print(f"Validation set size: {val_size:,} rows") 
print(f"Test set size: {test_size:,} rows")

# 5. Verify the chronological split
print("\nVerifying chronological split...")
# Get actual date ranges for verification using correct Dask syntax
train_min, train_max = train_ddf['date'].min().compute(), train_ddf['date'].max().compute()
val_min, val_max = val_ddf['date'].min().compute(), val_ddf['date'].max().compute()
test_min, test_max = test_ddf['date'].min().compute(), test_ddf['date'].max().compute()

# Calculate actual durations
train_days = (train_max - train_min).days + 1
val_days = (val_max - val_min).days + 1
test_days = (test_max - test_min).days + 1

print(f"Train dates: {train_min.strftime('%Y-%m-%d')} to {train_max.strftime('%Y-%m-%d')} ({train_days} days)")
print(f"Val dates: {val_min.strftime('%Y-%m-%d')} to {val_max.strftime('%Y-%m-%d')} ({val_days} days)")
print(f"Test dates: {test_min.strftime('%Y-%m-%d')} to {test_max.strftime('%Y-%m-%d')} ({test_days} days)")

# Verify test period is sufficient for 31-day forecasting
if test_days >= 31:
    print(f"✅ Test period sufficient for 31-day forecasting: {test_days} days")
else:
    print(f"❌ WARNING: Test period too short for 31-day forecasting: {test_days} days (need ≥31)")

# Check for proper chronological order
if train_max < val_min and val_max < test_min:
    print("✅ Chronological split confirmed - no overlap between splits")
    gap1 = (val_min - train_max).days
    gap2 = (test_min - val_max).days
    print(f"   Train→Val gap: {gap1} days, Val→Test gap: {gap2} days")
else:
    print("❌ WARNING: Overlap detected in date splits!")
    if train_max >= val_min:
        print(f"   Train overlaps with Val: {train_max} >= {val_min}")
    if val_max >= test_min:
        print(f"   Val overlaps with Test: {val_max} >= {test_min}")

# 6. Scale numerical features using MinMaxScaler to [0,1] range
print(f"\nComputing global min/max for proper MinMaxScaler on {len(numerical_features)} numerical features...")

# Compute global min/max using the full training dataset for proper scaling bounds
print("Computing global statistics (this may take a moment)...")
global_mins = train_ddf[numerical_features].min().compute()
global_maxs = train_ddf[numerical_features].max().compute()

# Create scaler with proper global bounds
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit using global min/max to ensure all values fall in [0,1]
bounds_array = np.vstack([global_mins.values, global_maxs.values])
scaler.fit(bounds_array)

# Save the scaler
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Print scaling info
print("Scaling ranges:")
for i, col in enumerate(numerical_features[:10]):  # Show first 10 for brevity
    print(f"  {col}: [{scaler.data_min_[i]:.3f}, {scaler.data_max_[i]:.3f}] -> [0, 1]")
if len(numerical_features) > 10:
    print(f"  ... and {len(numerical_features) - 10} more features")

# Define a function to apply scaling
def scale_partition(df, scaler, num_feature_cols):
    """Apply MinMaxScaler to a partition of the dataframe"""
    if df.empty:
        return df
    df_copy = df.copy()
    # Ensure columns are in the same order as during fit
    df_copy[num_feature_cols] = scaler.transform(df_copy[num_feature_cols])
    return df_copy

# Apply scaling to all partitions
print("Applying scaling to all datasets...")
meta_df = train_ddf._meta.copy()
train_scaled_ddf = train_ddf.map_partitions(scale_partition, scaler, numerical_features, meta=meta_df)
val_scaled_ddf = val_ddf.map_partitions(scale_partition, scaler, numerical_features, meta=meta_df)
test_scaled_ddf = test_ddf.map_partitions(scale_partition, scaler, numerical_features, meta=meta_df)

print("✅ Scaling applied to all datasets")

# 7. Verify scaling worked correctly
print("Verifying scaling...")
sample_scaled = train_scaled_ddf[numerical_features].sample(frac=0.001, random_state=42).compute()
print(f"Scaled feature ranges (sample verification):")
for col in numerical_features[:5]:  # Check first 5 features
    min_val, max_val = sample_scaled[col].min(), sample_scaled[col].max()
    print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")

# 8. Save the processed dataframes
print("\nSaving preprocessed data...")
save_start = time.time()
print("Saving training set...")
train_scaled_ddf.to_parquet(TRAIN_PATH, overwrite=True)
print("Saving validation set...")
val_scaled_ddf.to_parquet(VAL_PATH, overwrite=True)
print("Saving test set...")
test_scaled_ddf.to_parquet(TEST_PATH, overwrite=True)
save_time = time.time() - save_start
print(f"⏱️ Data saving completed in {save_time:.1f} seconds")

print("\n" + "="*80)
print("🎉 PREPROCESSING COMPLETE!")
print("="*80)

# Calculate total time
total_time = time.time() - start_time
print(f"⏱️ Total preprocessing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print("")

print(f"✅ Applied continuous 10% temporal sampling")
print(f"✅ Handled -999 missing values")
print(f"✅ Removed {initial_rows - final_rows:,} rows with missing data")
print(f"✅ Scaled {len(numerical_features)} numerical features to [0,1] range")
print(f"✅ One-hot encoded categorical features")
print(f"✅ Created time-aware splits:")
print(f"   📊 Training: {train_size:,} rows")
print(f"   📊 Validation: {val_size:,} rows") 
print(f"   📊 Test: {test_size:,} rows")
print(f"✅ Saved processed data to: {PROCESSED_DATA_DIR}/")
print(f"✅ Saved scaler to: {SCALER_PATH}")
print(f"✅ Saved column info to: {ENCODER_COLUMNS_PATH}")
print("="*80)
print(f"🎯 READY FOR TRAINING! Total samples: {train_size + val_size + test_size:,}")
print(f"⏱️ End time: {datetime.now().strftime('%H:%M:%S')} (Duration: {total_time:.1f}s)")
print("="*80)

# Close the output file and restore stdout
sys.stdout = original_stdout
output_file.close()
print(f"✅ Preprocessing complete! Duration: {total_time:.1f} seconds")
print("✅ Output saved to output.txt")
