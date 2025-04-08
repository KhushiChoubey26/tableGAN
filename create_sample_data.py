import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

print("Creating sample datasets for testing...")

# Create sample Chess dataset (in case the real one has issues)
print("Creating sample Chess dataset...")
n_samples = 1000
n_features = 5

# Create features
np.random.seed(42)
chess_data = pd.DataFrame({
    'rated': np.random.choice([0, 1], size=n_samples),
    'victory_status': np.random.randint(0, 3, size=n_samples),
    'white_rating': np.random.normal(1500, 200, size=n_samples).astype(int),
    'black_rating': np.random.normal(1500, 200, size=n_samples).astype(int),
    'turns': np.random.randint(10, 60, size=n_samples)
})

# Create labels (1 if white_rating > black_rating, else 0)
chess_labels = (chess_data['white_rating'] > chess_data['black_rating']).astype(int)

# Ensure the Chess directory exists
import os
if not os.path.exists("data/Chess"):
    os.makedirs("data/Chess")

# Save the sample Chess data
chess_data.to_csv("data/Chess/Chess.csv", index=False, sep=';')
chess_labels.to_csv("data/Chess/Chess_labels.csv", index=False)

# Create sample Covtype dataset
print("Creating sample Covtype dataset...")
n_samples = 1000
n_features = 10

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=n_samples, 
    n_features=n_features, 
    n_informative=5, 
    n_redundant=3,
    random_state=42
)

# Convert to DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
covtype_data = pd.DataFrame(X, columns=feature_names)

# Convert labels to binary
covtype_labels = pd.Series(y)

# Ensure the Covtype directory exists
if not os.path.exists("data/Covtype"):
    os.makedirs("data/Covtype")

# Save the sample Covtype data
covtype_data.to_csv("data/Covtype/Covtype.csv", index=False, sep=';')
covtype_labels.to_csv("data/Covtype/Covtype_labels.csv", index=False)

print("Sample datasets created successfully!") 