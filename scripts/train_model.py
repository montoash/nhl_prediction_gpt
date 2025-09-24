import nfl_data_py as nfl
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from mapie.classification import SplitConformalClassifier
import joblib
import os
from datetime import datetime

print("Starting model training process...")

# --- 1. Data Ingestion ---
# Fetch play-by-play data for recent seasons
print("Fetching data season-by-season from nfl_data_py...")
# Build dynamic season list: 2015 -> current NFL season year
current_year = datetime.now().year
seasons = list(range(2015, current_year + 1))

# We'll process each season separately to reduce memory footprint and avoid leakage
season_frames = []
for yr in seasons:
    print(f"Processing season {yr}...")
    # Try cache first for speed; fallback to downloading if cache missing
    try:
        pbp = nfl.import_pbp_data([yr], downcast=True, cache=True, alt_path=None)
    except Exception as e:
        print(f"  Cache not available for {yr} or error occurred ({e}); downloading...")
        try:
            pbp = nfl.import_pbp_data([yr], downcast=True, cache=False, alt_path=None)
        except Exception as e2:
            print(f"  Skipping {yr} due to error: {e2}")
            continue
    # Keep only necessary columns early to save memory
    keep_cols = [
        'game_id', 'season', 'week', 'posteam', 'epa', 'success',
        'home_team', 'away_team', 'home_score', 'away_score', 'season_type'
    ]
    pbp = pbp[[c for c in keep_cols if c in pbp.columns]]

    # --- Feature Engineering per season (leakage-free) ---
    pbp_reg = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()

    # Team-level stats per game
    team_stats = pbp_reg.groupby(['game_id', 'season', 'week', 'posteam']).agg(
        off_epa_per_play=('epa', 'mean'),
        off_success_rate=('success', 'mean')
    ).reset_index()

    # Pre-game rolling features within season
    team_stats = team_stats.sort_values(['season', 'posteam', 'week']).copy()
    roll_window = 3
    team_stats['pre_epa'] = team_stats.groupby(['season', 'posteam'])['off_epa_per_play'] \
        .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=1).mean())
    team_stats['pre_success'] = team_stats.groupby(['season', 'posteam'])['off_success_rate'] \
        .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=1).mean())

    # Game results
    game_results = pbp_reg[['game_id', 'home_team', 'away_team', 'home_score', 'away_score', 'season', 'week']].drop_duplicates()
    game_results['home_win'] = (game_results['home_score'] > game_results['away_score']).astype(int)

    # Merge pre-game stats for home and away
    home_stats = team_stats.rename(columns={
        'posteam': 'home_team',
        'pre_epa': 'home_pre_epa',
        'pre_success': 'home_pre_success'
    })
    away_stats = team_stats.rename(columns={
        'posteam': 'away_team',
        'pre_epa': 'away_pre_epa',
        'pre_success': 'away_pre_success'
    })

    final_df_season = pd.merge(
        game_results,
        home_stats[['game_id', 'home_pre_epa', 'home_pre_success']],
        on='game_id', how='left'
    )
    final_df_season = pd.merge(
        final_df_season,
        away_stats[['game_id', 'away_pre_epa', 'away_pre_success']],
        on='game_id', how='left'
    )

    # Features
    final_df_season['epa_diff'] = final_df_season['home_pre_epa'] - final_df_season['away_pre_epa']
    final_df_season['success_diff'] = final_df_season['home_pre_success'] - final_df_season['away_pre_success']

    # Drop rows with missing feature data
    final_df_season.dropna(subset=['epa_diff', 'success_diff'], inplace=True)

    season_frames.append(final_df_season[['game_id', 'season', 'week', 'epa_diff', 'success_diff', 'home_win']])

# Combine all seasons and perform chronological split
final_df = pd.concat(season_frames, ignore_index=True)
final_df = final_df.sort_values(['season', 'week']).reset_index(drop=True)

# --- 2. Feature Engineering (Simplified) ---
print("Engineering features...")
features = ['epa_diff', 'success_diff']
target = 'home_win'
X = final_df[features]
y = final_df[target]
split_idx = int(0.7 * len(final_df))
X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

# --- 3. Supervised Model Training ---
print("Training XGBoost model...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# --- 4. Calibration and Uncertainty (Conformal Prediction) ---
print("Applying conformal prediction wrapper...")
# Use SplitConformalClassifier for conformal prediction (Mapie v0.8+ API)
conformal_model = SplitConformalClassifier(
    estimator=model,
    confidence_level=0.95,
    conformity_score="lac",
    prefit=True
)
conformal_model.conformalize(X_calib, y_calib)


# --- 5. Save Model Artifacts ---
print("Saving model artifacts...")
os.makedirs('../models', exist_ok=True)
joblib.dump(conformal_model, '../models/nfl_win_predictor.pkl')
joblib.dump(features, '../models/features.pkl')

print("Model training complete and artifacts saved.")