# scripts/create_mock_nhl_models.py
"""
Create mock NHL models for initial deployment testing
This allows you to deploy and test the API while working on real NHL data integration
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from mapie.classification import SplitConformalClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
import os

print("Creating mock NHL models for testing...")

# Create mock NHL training data
np.random.seed(42)
n_games = 1000

# NHL teams
nhl_teams = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 
             'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 
             'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 
             'TOR', 'VAN', 'VGK', 'WSH', 'WPG']

# Generate mock features
mock_data = []
for i in range(n_games):
    # Random team matchup
    home_team = np.random.choice(nhl_teams)
    away_team = np.random.choice([t for t in nhl_teams if t != home_team])
    
    # Mock NHL features (realistic ranges)
    goals_for_diff = np.random.normal(0, 0.5)  # Goals per game differential
    goals_against_diff = np.random.normal(0, 0.5)  # Goals allowed differential
    shots_diff = np.random.normal(0, 3)  # Shots per game differential
    powerplay_diff = np.random.normal(0, 0.05)  # PP% differential
    giveaway_diff = np.random.normal(0, 0.05)  # Giveaway differential
    
    # Home ice advantage (~55% win rate)
    home_advantage = 0.1
    
    # Calculate win probability based on features
    skill_diff = (goals_for_diff + goals_against_diff + 
                 shots_diff * 0.02 + powerplay_diff + giveaway_diff)
    
    win_prob = 1 / (1 + np.exp(-(skill_diff + home_advantage)))
    home_win = 1 if np.random.random() < win_prob else 0
    
    mock_data.append({
        'home_team': home_team,
        'away_team': away_team,
        'goals_for_diff': goals_for_diff,
        'goals_against_diff': goals_against_diff,
        'shots_diff': shots_diff,
        'powerplay_diff': powerplay_diff,
        'giveaway_diff': giveaway_diff,
        'home_win': home_win
    })

df = pd.DataFrame(mock_data)
print(f"Generated {len(df)} mock NHL games")

# Define features
features_no_odds = ['goals_for_diff', 'goals_against_diff', 'shots_diff', 'powerplay_diff', 'giveaway_diff']
features_with_odds = features_no_odds.copy()  # Same for now

# Prepare data
X = df[features_no_odds]
y = df['home_win']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    n_estimators=100,  # Reduced for mock data
    reg_lambda=2.0,
    reg_alpha=0.0,
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

# Calibrate the model
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Use calibrated model directly (simpler approach for mock models)
conformal_model = calibrated_model

# Test accuracy
test_accuracy = conformal_model.score(X_test, y_test)
print(f"Mock model test accuracy: {test_accuracy:.3f}")

# Save models
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Save both versions (with and without odds are the same for now)
joblib.dump(conformal_model, os.path.join(MODELS_DIR, 'nhl_win_predictor_no_odds.pkl'), protocol=2)
joblib.dump(conformal_model, os.path.join(MODELS_DIR, 'nhl_win_predictor_with_odds.pkl'), protocol=2)
joblib.dump(features_no_odds, os.path.join(MODELS_DIR, 'features_no_odds.pkl'), protocol=2)
joblib.dump(features_with_odds, os.path.join(MODELS_DIR, 'features_with_odds.pkl'), protocol=2)

print("✅ Mock NHL models created successfully!")
print("Models saved:")
print("  - nhl_win_predictor_no_odds.pkl")
print("  - nhl_win_predictor_with_odds.pkl") 
print("  - features_no_odds.pkl")
print("  - features_with_odds.pkl")
print()
print("🚀 Your API is now ready for deployment testing!")
print("   You can deploy to Google Cloud and test with mock predictions.")
print("   Later, replace with real NHL data models.")