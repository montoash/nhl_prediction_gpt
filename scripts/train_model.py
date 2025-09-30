# scripts/train_model.py

import requests
import pandas as pd
import xgboost as xgb
from mapie.classification import SplitConformalClassifier
from mapie.regression import MapieRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime
import gc # Garbage Collector interface

print("Starting MEMORY-EFFICIENT ADVANCED NHL model training process...")

# --- 1. Data Ingestion Setup ---
current_year = datetime.now().year
# Allow overriding season range via environment variables for faster/local runs
start_season = int(os.getenv('START_SEASON', '2018'))
end_season = int(os.getenv('END_SEASON', str(current_year)))
seasons = list(range(start_season, end_season + 1))

# --- 2. Process Data Season-by-Season ---
all_seasons_df = []
for season in seasons:
    print(f"Processing season {season}...")
    
    # Ingest data for the single season (skip if unavailable)
    try:
        pbp_df = nfl.import_pbp_data([season], downcast=True, cache=False)
        schedule_df = nfl.import_schedules([season])
    except Exception as e:
        print(f"Season {season} data unavailable: {e}. Skipping.")
        continue
    
    # Merge schedules and odds
    # Use schedules as source of odds and metadata (spread_line, total_line, moneylines are included)
    game_data = schedule_df[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline']].copy()
    game_data['home_win'] = (game_data['home_score'] > game_data['away_score']).astype(int)

    # Compute implied probabilities from moneylines with basic vig removal
    def moneyline_to_implied_prob(ml):
        if pd.isna(ml):
            return None
        try:
            ml = float(ml)
        except Exception:
            return None
        if ml < 0:
            return (-ml) / ((-ml) + 100)
        else:
            return 100 / (ml + 100)

    game_data['home_implied_raw'] = game_data['home_moneyline'].apply(moneyline_to_implied_prob)
    game_data['away_implied_raw'] = game_data['away_moneyline'].apply(moneyline_to_implied_prob)
    # Normalize to remove vig when both sides available
    def normalize_pair(row):
        h, a = row['home_implied_raw'], row['away_implied_raw']
        if h is None or a is None or pd.isna(h) or pd.isna(a) or (h + a) == 0:
            return None
        return h / (h + a)
    game_data['implied_home_prob'] = game_data.apply(normalize_pair, axis=1)

    # Engineer features for the season
    pbp_reg = pbp_df.loc[(pbp_df['season_type'] == 'REG') & (pbp_df['epa'].notna())].copy()
    pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype(int)
    pbp_reg['is_turnover'] = ((pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1)).astype(int)

    # Offensive team-level features per game
    team_game_stats = pbp_reg.groupby(['game_id', 'posteam']).agg(
        off_epa_per_play=('epa', 'mean'),
        explosive_play_rate=('is_explosive', 'mean'),
        turnover_rate=('is_turnover', 'mean')
    ).reset_index()

    # Defensive features per game (EPA allowed per play by defense)
    defense_game_stats = pbp_reg.groupby(['game_id', 'defteam']).agg(
        def_epa_allowed_per_play=('epa', 'mean')
    ).reset_index().rename(columns={'defteam': 'posteam'})

    game_info = schedule_df[['game_id', 'week']]
    team_game_stats = pd.merge(team_game_stats, game_info, on='game_id', how='left')
    defense_game_stats = pd.merge(defense_game_stats, game_info, on='game_id', how='left')
    team_game_stats.sort_values(['posteam', 'week'], inplace=True)
    defense_game_stats.sort_values(['posteam', 'week'], inplace=True)

    roll_window = 4
    for stat in ['off_epa_per_play', 'explosive_play_rate', 'turnover_rate']:
        team_game_stats[f'pre_game_{stat}'] = team_game_stats.groupby('posteam')[stat] \
            .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=1).mean())
    defense_game_stats['pre_game_def_epa_allowed_per_play'] = defense_game_stats.groupby('posteam')['def_epa_allowed_per_play'] \
        .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=1).mean())

    final_df_season = pd.merge(game_data, team_game_stats.add_prefix('home_'),
                               left_on=['game_id', 'home_team'],
                               right_on=['home_game_id', 'home_posteam'],
                               how='left')
    final_df_season = pd.merge(final_df_season, team_game_stats.add_prefix('away_'),
                               left_on=['game_id', 'away_team'],
                               right_on=['away_game_id', 'away_posteam'],
                               how='left')

    # Merge defensive rolling stats (avoid overlapping suffix columns)
    home_def = defense_game_stats[['game_id', 'posteam', 'pre_game_def_epa_allowed_per_play']].rename(
        columns={'posteam': 'home_team', 'pre_game_def_epa_allowed_per_play': 'home_pre_game_def_epa_allowed_per_play'}
    )
    away_def = defense_game_stats[['game_id', 'posteam', 'pre_game_def_epa_allowed_per_play']].rename(
        columns={'posteam': 'away_team', 'pre_game_def_epa_allowed_per_play': 'away_pre_game_def_epa_allowed_per_play'}
    )
    final_df_season = pd.merge(final_df_season, home_def, on=['game_id', 'home_team'], how='left')
    final_df_season = pd.merge(final_df_season, away_def, on=['game_id', 'away_team'], how='left')

    final_df_season['off_epa_diff'] = final_df_season['home_pre_game_off_epa_per_play'] - final_df_season['away_pre_game_off_epa_per_play']
    # Defensive advantage for home: lower EPA allowed is better => away - home so that positive favors home
    final_df_season['def_epa_allowed_diff'] = final_df_season['away_pre_game_def_epa_allowed_per_play'] - final_df_season['home_pre_game_def_epa_allowed_per_play']
    final_df_season['explosive_diff'] = final_df_season['home_pre_game_explosive_play_rate'] - final_df_season['away_pre_game_explosive_play_rate']
    final_df_season['turnover_diff'] = final_df_season['home_pre_game_turnover_rate'] - final_df_season['away_pre_game_turnover_rate']
    
    all_seasons_df.append(final_df_season)
    
    # Clean up memory before next loop
    del pbp_df, schedule_df, final_df_season, team_game_stats, defense_game_stats
    gc.collect()

# --- 3. Combine All Seasons and Train ---
print("Combining all seasons and training models...")
final_df = pd.concat(all_seasons_df, ignore_index=True)
final_df.sort_values(['season', 'week'], inplace=True)

# Resolve models directory relative to repo root to avoid CWD issues
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
target = 'home_win'

import xgboost as xgb

print("Training Primary Model (with betting odds)...")
features_with_odds = ['spread_line', 'total_line', 'implied_home_prob', 'off_epa_diff', 'def_epa_allowed_diff', 'explosive_diff', 'turnover_diff']
# Require target and spread_line; allow other vegas fields to be NaN if unavailable
df_with_odds = final_df.dropna(subset=['spread_line', target]).copy()
X = df_with_odds[features_with_odds]
y = df_with_odds[target]

# Time-aware split: last 20% as calibration/validation
split_idx = int(0.8 * len(df_with_odds))
X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

model_odds = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    n_estimators=2000,
    reg_lambda=2.0,
    reg_alpha=0.0,
    tree_method='hist',
)
model_odds.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)

# Isotonic calibration on held-out calibration set
calibrated_odds = CalibratedClassifierCV(model_odds, method='isotonic', cv='prefit')
calibrated_odds.fit(X_calib, y_calib)

# Conformal wrapper for uncertainty quantification
conformal_model_odds = SplitConformalClassifier(estimator=calibrated_odds, confidence_level=0.95, conformity_score="lac", prefit=True)
conformal_model_odds.conformalize(X_calib, y_calib)

joblib.dump(conformal_model_odds, os.path.join(MODELS_DIR, 'nhl_win_predictor_with_odds.pkl'), protocol=2)
joblib.dump(features_with_odds, os.path.join(MODELS_DIR, 'features_with_odds.pkl'), protocol=2)
print("Primary model saved.")

# === Train and Save Fallback Model (no odds) ===
print("Training Fallback Model (without betting odds)...")
features_no_odds = ['off_epa_diff', 'def_epa_allowed_diff', 'explosive_diff', 'turnover_diff']
df_no_odds = final_df.dropna(subset=features_no_odds + [target]).copy()
X = df_no_odds[features_no_odds]
y = df_no_odds[target]
split_idx = int(0.8 * len(df_no_odds))
X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

model_no_odds = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    n_estimators=2000,
    reg_lambda=2.0,
    reg_alpha=0.0,
    tree_method='hist',
)
model_no_odds.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)
calibrated_no_odds = CalibratedClassifierCV(model_no_odds, method='isotonic', cv='prefit')
calibrated_no_odds.fit(X_calib, y_calib)
conformal_model_no_odds = SplitConformalClassifier(estimator=calibrated_no_odds, confidence_level=0.95, conformity_score="lac", prefit=True)
conformal_model_no_odds.conformalize(X_calib, y_calib)

joblib.dump(conformal_model_no_odds, os.path.join(MODELS_DIR, 'nhl_win_predictor_no_odds.pkl'), protocol=2)
joblib.dump(features_no_odds, os.path.join(MODELS_DIR, 'features_no_odds.pkl'), protocol=2)
print("Fallback model saved.")

print("Advanced model training complete. 🚀")

# === Optional: Train spread and total point regressors ===
if os.getenv('TRAIN_REGRESSORS', '1') == '1':
    try:
        print("Training spread and total regressors (optional)...")
        # True targets from schedule
        final_df['true_spread'] = (final_df['home_score'] - final_df['away_score']).astype(float)
        final_df['true_total'] = (final_df['home_score'] + final_df['away_score']).astype(float)

        reg_features = ['off_epa_diff', 'def_epa_allowed_diff', 'explosive_diff', 'turnover_diff', 'implied_home_prob']
        # Use vegas info when present; allow NaN rows to be dropped per target
        df_spread = final_df.dropna(subset=reg_features + ['true_spread']).copy()
        df_total = final_df.dropna(subset=reg_features + ['true_total']).copy()

        def train_reg(df, target):
            split_idx = int(0.8 * len(df))
            Xtr, Xva = df[reg_features].iloc[:split_idx], df[reg_features].iloc[split_idx:]
            ytr, yva = df[target].iloc[:split_idx], df[target].iloc[split_idx:]
            base = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                n_estimators=1500,
                reg_lambda=1.0,
                tree_method='hist',
            )
            base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = base.predict(Xva)
            mae = mean_absolute_error(yva, pred)
            print(f"{target} MAE: {mae:.2f}")
            # Conformal interval for spread/total (absolute error)
            mr = MapieRegressor(estimator=base, method="naive")
            mr.fit(Xtr, ytr)
            return mr

        spread_reg = train_reg(df_spread, 'true_spread')
        total_reg = train_reg(df_total, 'true_total')
        joblib.dump(spread_reg, os.path.join(MODELS_DIR, 'spread_regressor.pkl'), protocol=2)
        joblib.dump(total_reg, os.path.join(MODELS_DIR, 'total_regressor.pkl'), protocol=2)
        print("Regressors saved.")
    except Exception as e:
        print(f"Regressor training skipped due to error: {e}")