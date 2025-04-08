import pandas as pd
import numpy as np

# Load the covtype dataset - it appears to be semicolon-delimited
print("Loading Covtype dataset...")
covtype_df = pd.read_csv("data/covtype/covtype.csv", sep=';')

# If we still have a single column, the file might be using a different delimiter
if len(covtype_df.columns) == 1:
    print("Dataset appears to have a different delimiter, trying other options...")
    
    # Try to read the first few lines to determine the delimiter
    with open("data/covtype/covtype.csv", 'r') as f:
        first_line = f.readline().strip()
    
    if ',' in first_line:
        covtype_df = pd.read_csv("data/covtype/covtype.csv")
    elif '\t' in first_line:
        covtype_df = pd.read_csv("data/covtype/covtype.csv", sep='\t')

# Print first few rows to inspect data
print("\nDataset shape:", covtype_df.shape)
print("\nColumn names:", list(covtype_df.columns))
print("\nData types:")
print(covtype_df.dtypes)

# If we have more than 10 columns, proceed with processing
if len(covtype_df.columns) > 10:
    # The Cover_Type is typically the last column
    target_col = covtype_df.columns[-1]
    print(f"\nUsing {target_col} as the target column")
    
    # Try to convert to numeric if it's not already
    if not pd.api.types.is_numeric_dtype(covtype_df[target_col]):
        try:
            covtype_df[target_col] = pd.to_numeric(covtype_df[target_col])
        except:
            print(f"Could not convert {target_col} to numeric, treating as categorical")
    
    # Check if it's numeric now
    if pd.api.types.is_numeric_dtype(covtype_df[target_col]):
        # Make it binary (1 if > median, 0 otherwise)
        median_val = covtype_df[target_col].median()
        print(f"Target median value: {median_val}")
        covtype_df['label'] = (covtype_df[target_col] > median_val).astype(int)
    else:
        # For categorical target, just use label encoding
        unique_vals = covtype_df[target_col].unique()
        print(f"Unique values in target: {unique_vals[:10]}")
        
        # Create a binary label (0 or 1) based on category
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded = le.fit_transform(covtype_df[target_col])
        # Make binary (0 if in first half of categories, 1 otherwise)
        median_encoded = np.median(encoded)
        covtype_df['label'] = (encoded > median_encoded).astype(int)
    
    # Remove the original target column
    covtype_df = covtype_df.drop(target_col, axis=1)
    
    # Convert all remaining columns to numeric if possible
    for col in covtype_df.columns:
        if col != 'label' and not pd.api.types.is_numeric_dtype(covtype_df[col]):
            try:
                covtype_df[col] = pd.to_numeric(covtype_df[col], errors='coerce')
            except:
                # Drop columns that can't be converted to numeric
                print(f"Dropping non-numeric column: {col}")
                covtype_df = covtype_df.drop(col, axis=1)
    
    # Fill NaN values with column means
    covtype_df = covtype_df.fillna(covtype_df.mean())
    
    # Select only numeric columns except label
    numeric_cols = [col for col in covtype_df.columns if col != 'label' and pd.api.types.is_numeric_dtype(covtype_df[col])]
    
    # Select a subset of features if there are too many
    max_features = 20
    if len(numeric_cols) > max_features:
        selected_cols = numeric_cols[:max_features]
    else:
        selected_cols = numeric_cols
    
    print(f"\nSelected {len(selected_cols)} numerical features")
    
    # Create a subset with only the selected features
    covtype_subset = covtype_df[selected_cols]
    
    # Save the dataset and labels
    print("Saving processed Covtype dataset...")
    # Ensure directory exists
    import os
    if not os.path.exists("data/Covtype"):
        os.makedirs("data/Covtype")
    
    # Save the data (selected features)
    covtype_subset.to_csv("data/Covtype/Covtype.csv", index=False, sep=';')
    
    # Save the label separately
    covtype_df['label'].to_csv("data/Covtype/Covtype_labels.csv", index=False)
    
    print("Covtype preprocessing completed.")
else:
    print("\nError: Dataset doesn't have enough columns. Please check the file format.") 