# scripts/train_model.py

import nfl_data_py as nfl
import pandas as pd
import xgboost as xgb
from mapie.classification import SplitConformalClassifier
import joblib
import os
from datetime import datetime

print("Starting ADVANCED model training process...")

# --- 1. Data Ingestion (Play-by-Play, Schedules, and Odds) ---
print("Fetching PBP, schedule, and odds data...")
current_year = datetime.now().year
seasons = list(range(2018, current_year + 1)) # Odds data is more reliable from 2018 onwards

# Fetch all necessary dataframes
pbp_df = nfl.import_pbp_data(seasons, downcast=True, cache=False)
schedule_df = nfl.import_schedules(seasons)
odds_df = nfl.import_odds(seasons)

# --- 2. Data Merging and Preparation ---
print("Merging data sources...")

# Select necessary schedule columns and merge with odds
game_data = schedule_df[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
game_data['home_win'] = (game_data['home_score'] > game_data['away_score']).astype(int)

# Select and rename odds columns, then merge
odds_data = odds_df[['game_id', 'spread_line']].copy()
game_data = pd.merge(game_data, odds_data, on='game_id', how='left')

# --- 3. Advanced Feature Engineering (Leakage-Free) ---
print("Engineering advanced rolling features...")

pbp_reg = pbp_df.loc[(pbp_df['season_type'] == 'REG') & (pbp_df['epa'].notna())].copy()

# Define what constitutes an explosive play or a turnover
pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype(int)
pbp_reg['is_turnover'] = (pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1).astype(int)

# Aggregate stats per team, per game
team_game_stats = pbp_reg.groupby(['game_id', 'posteam']).agg(
    off_epa_per_play=('epa', 'mean'),
    explosive_play_rate=('is_explosive', 'mean'),
    turnover_rate=('is_turnover', 'mean')
).reset_index()

# Merge game info (season, week) to the stats
game_info = schedule_df[['game_id', 'season', 'week']]
team_game_stats = pd.merge(team_game_stats, game_info, on='game_id', how='left')

# Calculate rolling averages for each stat to use as pre-game features
team_game_stats.sort_values(['posteam', 'season', 'week'], inplace=True)
roll_window = 4 # Use a 4-game rolling window
for stat in ['off_epa_per_play', 'explosive_play_rate', 'turnover_rate']:
    team_game_stats[f'pre_game_{stat}'] = team_game_stats.groupby(['posteam', 'season'])[stat] \
        .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=1).mean())

# Merge the rolling stats back into the main game dataframe
# Merge for home team
final_df = pd.merge(game_data, team_game_stats.add_prefix('home_'),
                    left_on=['game_id', 'home_team'],
                    right_on=['home_game_id', 'home_posteam'],
                    how='left')
# Merge for away team
final_df = pd.merge(final_df, team_game_stats.add_prefix('away_'),
                    left_on=['game_id', 'away_team'],
                    right_on=['away_game_id', 'away_posteam'],
                    how='left')

# Create final differential features
final_df['epa_diff'] = final_df['home_pre_game_off_epa_per_play'] - final_df['away_pre_game_off_epa_per_play']
final_df['explosive_diff'] = final_df['home_pre_game_explosive_play_rate'] - final_df['away_pre_game_explosive_play_rate']
final_df['turnover_diff'] = final_df['home_pre_game_turnover_rate'] - final_df['away_pre_game_turnover_rate']

# --- 4. Model Training ---
print("Training the model...")

# Define the full feature set
features = ['spread_line', 'epa_diff', 'explosive_diff', 'turnover_diff']
target = 'home_win'

# Drop any games with missing data (e.g., early season games, missing odds)
final_df.dropna(subset=features, inplace=True)

# Chronological split: train on earlier data, calibrate on more recent data
X = final_df[features]
y = final_df[target]
split_idx = int(0.8 * len(final_df)) # Use 80% for training, 20% for calibration
X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training on {len(X_train)} games, calibrating on {len(X_calib)} games.")

# Initialize and train XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Conformalize the model to get prediction sets
conformal_model = SplitConformalClassifier(estimator=model, confidence_level=0.95, conformity_score="lac", prefit=True)
conformal_model.conformalize(X_calib, y_calib)

# --- 5. Save Artifacts ---
print("Saving model and feature list...")
os.makedirs('../models', exist_ok=True)
joblib.dump(conformal_model, '../models/nfl_win_predictor.pkl')
joblib.dump(features, '../models/features.pkl')

print("Advanced model training complete. 🚀")