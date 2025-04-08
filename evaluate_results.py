import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import os
from scipy import stats

# Function to load real and synthetic data
def load_data(dataset, test_id):
    # Load real data
    real_data_path = f"data/{dataset}/{dataset}.csv"
    real_data = pd.read_csv(real_data_path, sep=';')
    
    # Load synthetic data
    synthetic_data_path = f"samples/{dataset}/{test_id}/{dataset}_{test_id}_fake.csv"
    if os.path.exists(synthetic_data_path):
        synthetic_data = pd.read_csv(synthetic_data_path)
        return real_data, synthetic_data
    else:
        print(f"Synthetic data file not found: {synthetic_data_path}")
        return real_data, None

# Function to calculate statistical metrics
def calculate_metrics(real_data, synthetic_data):
    metrics = {}
    
    # Mean squared error between real and synthetic column means
    real_means = real_data.mean()
    synthetic_means = synthetic_data.mean()
    metrics['mean_mse'] = mean_squared_error(real_means, synthetic_means)
    
    # Mean squared error between real and synthetic column standard deviations
    real_stds = real_data.std()
    synthetic_stds = synthetic_data.std()
    metrics['std_mse'] = mean_squared_error(real_stds, synthetic_stds)
    
    # Correlation matrix difference
    real_corr = real_data.corr().fillna(0)
    synthetic_corr = synthetic_data.corr().fillna(0)
    metrics['corr_mse'] = np.mean((real_corr - synthetic_corr) ** 2)
    
    return metrics

# Function to visualize the data
def visualize_data(real_data, synthetic_data, dataset, test_id, output_dir="visuals"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Compare distributions of key features (first 3)
    key_features = real_data.columns[:min(3, len(real_data.columns))]
    
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        plt.hist(real_data[feature], bins=30, alpha=0.5, label='Real')
        plt.hist(synthetic_data[feature], bins=30, alpha=0.5, label='Synthetic')
        plt.legend()
        plt.title(f'Distribution of {feature} - {dataset} ({test_id})')
        plt.savefig(f"{output_dir}/{dataset}_{test_id}_{feature}_dist.png")
        plt.close()
    
    # 2. PCA visualization
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_data)
    synthetic_pca = pca.transform(synthetic_data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(real_pca[:1000, 0], real_pca[:1000, 1], s=5, label='Real')
    plt.scatter(synthetic_pca[:1000, 0], synthetic_pca[:1000, 1], s=5, label='Synthetic')
    plt.legend()
    plt.title(f'PCA Visualization - {dataset} ({test_id})')
    plt.savefig(f"{output_dir}/{dataset}_{test_id}_pca.png")
    plt.close()
    
    # 3. Correlation matrix difference
    real_corr = real_data.corr()
    synthetic_corr = synthetic_data.corr()
    diff_corr = abs(real_corr - synthetic_corr)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(diff_corr, annot=False, cmap='coolwarm')
    plt.title(f'Correlation Matrix Difference - {dataset} ({test_id})')
    plt.savefig(f"{output_dir}/{dataset}_{test_id}_corr_diff.png")
    plt.close()

# Main evaluation function
def evaluate_datasets():
    datasets = ['Chess', 'Covtype']
    test_ids = ['OI_11_00', 'OI_11_11', 'OI_11_22']
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        
        for test_id in test_ids:
            print(f"Evaluating {dataset} with configuration {test_id}...")
            
            real_data, synthetic_data = load_data(dataset, test_id)
            
            if synthetic_data is not None:
                # Calculate metrics
                metrics = calculate_metrics(real_data, synthetic_data)
                results[dataset][test_id] = metrics
                
                # Visualize data
                visualize_data(real_data, synthetic_data, dataset, test_id)
                
                print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
                print(f"  Std MSE: {metrics['std_mse']:.6f}")
                print(f"  Correlation MSE: {metrics['corr_mse']:.6f}")
            else:
                print(f"  Skipping evaluation for {dataset} with {test_id}")
    
    # Save results to CSV
    results_df = pd.DataFrame(columns=['Dataset', 'Configuration', 'Mean_MSE', 'Std_MSE', 'Corr_MSE'])
    
    row = 0
    for dataset in results:
        for test_id in results[dataset]:
            metrics = results[dataset][test_id]
            results_df.loc[row] = [dataset, test_id, metrics['mean_mse'], metrics['std_mse'], metrics['corr_mse']]
            row += 1
    
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation completed. Results saved to evaluation_results.csv")

# Run the evaluation
if __name__ == "__main__":
    evaluate_datasets() 