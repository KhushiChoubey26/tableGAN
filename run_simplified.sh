#!/bin/bash

# Make sure we have sample data to work with
echo "Creating sample data..."
python create_sample_data.py

# Run training simulations for Chess dataset
echo "Running Chess dataset training simulation..."
python simple_train_chess.py

# Run generation for Chess dataset
echo "Generating synthetic Chess data..."
python simple_generate_chess.py

# Run training simulations for Covtype dataset
echo "Running Covtype dataset training simulation..."
python simple_train_covtype.py

# Run generation for Covtype dataset
echo "Generating synthetic Covtype data..."
python simple_generate_covtype.py

# Run the evaluation script
echo "Running evaluation..."
python evaluate_results.py

echo "All tasks completed!" 