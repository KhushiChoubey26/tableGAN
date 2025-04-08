#!/usr/bin/env bash

# Step 1: Preprocess datasets
echo "Preprocessing datasets..."
python preprocess_chess.py
python preprocess_covtype.py

# Step 2: Train models with different configurations
echo "Training Chess dataset models..."

# Configuration 1: OI_11_00 (beta:1.0, delta_v: 0.0, delta_m: 0.0)
python main.py --train --dataset=Chess --epoch=5 --test_id=OI_11_00

# Configuration 2: OI_11_11 (beta:1.0, delta_v: 0.1, delta_m: 0.1)
python main.py --train --dataset=Chess --epoch=5 --test_id=OI_11_11

# Configuration 3: OI_11_22 (beta:1.0, delta_v: 0.2, delta_m: 0.2)
python main.py --train --dataset=Chess --epoch=5 --test_id=OI_11_22

echo "Training Covtype dataset models..."

# Configuration 1: OI_11_00 (beta:1.0, delta_v: 0.0, delta_m: 0.0)
python main.py --train --dataset=Covtype --epoch=5 --test_id=OI_11_00

# Configuration 2: OI_11_11 (beta:1.0, delta_v: 0.1, delta_m: 0.1)
python main.py --train --dataset=Covtype --epoch=5 --test_id=OI_11_11

# Configuration 3: OI_11_22 (beta:1.0, delta_v: 0.2, delta_m: 0.2)
python main.py --train --dataset=Covtype --epoch=5 --test_id=OI_11_22

# Step 3: Generate synthetic data
echo "Generating synthetic data for Chess dataset..."

# Configuration 1: OI_11_00
python main.py --dataset=Chess --test_id=OI_11_00

# Configuration 2: OI_11_11
python main.py --dataset=Chess --test_id=OI_11_11

# Configuration 3: OI_11_22
python main.py --dataset=Chess --test_id=OI_11_22

echo "Generating synthetic data for Covtype dataset..."

# Configuration 1: OI_11_00
python main.py --dataset=Covtype --test_id=OI_11_00

# Configuration 2: OI_11_11
python main.py --dataset=Covtype --test_id=OI_11_11

# Configuration 3: OI_11_22
python main.py --dataset=Covtype --test_id=OI_11_22

# Step 4: Evaluate results
echo "Evaluating results..."
python evaluate_results.py

echo "All tasks completed!" 