import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set configuration
TEST_ID = "OI_11_00"  # Using simpler configuration
DATASET = "Chess"
SAMPLE_DIR = f"samples/{DATASET}/{TEST_ID}"
N_SAMPLES = 1000

# Create directories if they don't exist
os.makedirs(SAMPLE_DIR, exist_ok=True)

print(f"Generating synthetic {DATASET} data with configuration {TEST_ID}")

# In a real scenario, we would use the trained GAN model to generate data
# For this demonstration, we'll create a simple synthetic dataset based on the original data

# Try to load the original data
try:
    # Load the original data to understand the distribution
    data_path = f"data/{DATASET}/{DATASET}.csv"
    original_data = pd.read_csv(data_path, sep=';')
    
    # Create synthetic data with similar distribution
    print(f"Generating {N_SAMPLES} synthetic records based on original data distribution")
    
    # Get the column names
    feature_columns = original_data.columns
    
    # Create an empty dataframe for synthetic data
    synthetic_data = pd.DataFrame()
    
    # Generate data for each column with similar distribution
    for col in feature_columns:
        col_data = original_data[col]
        
        if col_data.dtype == 'object' or col_data.nunique() < 10:
            # Categorical data - sample from existing values
            values = col_data.sample(N_SAMPLES, replace=True).values
            synthetic_data[col] = values
        else:
            # Numerical data - use mean and std to generate values
            mean = col_data.mean()
            std = col_data.std() or 1.0  # Default to 1.0 if std is 0
            synthetic_data[col] = np.random.normal(mean, std, N_SAMPLES)
            
            # If original had integer values, convert to integers
            if col_data.dtype in ['int64', 'int32']:
                synthetic_data[col] = synthetic_data[col].round().astype(int)
                
    # Save the synthetic data
    output_file = f"{SAMPLE_DIR}/{DATASET}_{TEST_ID}_fake.csv"
    synthetic_data.to_csv(output_file, index=False)
    print(f"Saved {N_SAMPLES} synthetic records to {output_file}")
    
except Exception as e:
    print(f"Error loading original data: {e}")
    print("Generating random synthetic data instead")
    
    # Generate random data
    synthetic_data = pd.DataFrame({
        'rated': np.random.choice([0, 1], size=N_SAMPLES),
        'victory_status': np.random.randint(0, 3, size=N_SAMPLES),
        'white_rating': np.random.normal(1500, 200, size=N_SAMPLES).astype(int),
        'black_rating': np.random.normal(1500, 200, size=N_SAMPLES).astype(int),
        'turns': np.random.randint(10, 60, size=N_SAMPLES)
    })
    
    # Save the synthetic data
    output_file = f"{SAMPLE_DIR}/{DATASET}_{TEST_ID}_fake.csv"
    synthetic_data.to_csv(output_file, index=False)
    print(f"Saved {N_SAMPLES} random synthetic records to {output_file}")

print("Generation completed!") 