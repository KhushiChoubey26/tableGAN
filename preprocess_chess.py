import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the chess dataset
print("Loading Chess dataset...")
chess_df = pd.read_csv("data/chess/games.csv")

# Select relevant columns for our analysis
relevant_cols = ['rated', 'victory_status', 'winner', 'white_rating', 'black_rating', 'turns']
chess_df = chess_df[relevant_cols]

# Handle categorical variables
le_rated = LabelEncoder()
le_victory = LabelEncoder()
le_winner = LabelEncoder()

chess_df['rated'] = le_rated.fit_transform(chess_df['rated'])
chess_df['victory_status'] = le_victory.fit_transform(chess_df['victory_status'])
chess_df['winner'] = le_winner.fit_transform(chess_df['winner'])

# Create a binary label (to keep consistent with the tableGAN approach)
# 1 if white player won, 0 otherwise
chess_df['label'] = (chess_df['winner'] == le_winner.transform(['white'])[0]).astype(int)

# Remove the winner column as it would leak the label
chess_df = chess_df.drop('winner', axis=1)

# Save the dataset and labels
print("Saving processed Chess dataset...")
# Ensure directory exists
import os
if not os.path.exists("data/Chess"):
    os.makedirs("data/Chess")

# Save the data (everything except the label)
chess_df.drop('label', axis=1).to_csv("data/Chess/Chess.csv", index=False, sep=';')

# Save the label separately
chess_df['label'].to_csv("data/Chess/Chess_labels.csv", index=False)

print("Chess preprocessing completed.") 