# scripts/train_nhl_model.py

import requests
import pandas as pd
import xgboost as xgb
from mapie.classification import MapieClassifier
try:
    from mapie.regression import MapieRegressor
except ImportError:
    MapieRegressor = None
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime
import gc # Garbage Collector interface
import json
import time

print("Starting MEMORY-EFFICIENT ADVANCED NHL model training process...")

# --- 1. NHL Data Ingestion Setup ---
current_year = datetime.now().year
# NHL seasons are formatted as "20232024" for 2023-24 season
start_season = int(os.getenv('START_SEASON', '2018'))
end_season = int(os.getenv('END_SEASON', str(current_year)))

def get_nhl_season_format(year):
    """Convert year to NHL season format (e.g., 2023 -> '20232024')"""
    return f"{year}{year+1}"

def fetch_nhl_games(season_str):
    """Fetch NHL games for a given season"""
    try:
        # Use NHL API to get schedule
        url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season_str}&gameType=R"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for date_data in data['dates']:
            for game in date_data['games']:
                game_data = {
                    'game_id': game['gamePk'],
                    'season': season_str,
                    'date': game['gameDate'],
                    'home_team': game['teams']['home']['team']['abbreviation'],
                    'away_team': game['teams']['away']['team']['abbreviation'],
                    'home_score': game['teams']['home'].get('score', 0),
                    'away_score': game['teams']['away'].get('score', 0),
                    'game_state': game['status']['detailedState']
                }
                if game_data['game_state'] == 'Final':
                    games.append(game_data)
        
        return pd.DataFrame(games)
    except Exception as e:
        print(f"Error fetching NHL games for season {season_str}: {e}")
        return pd.DataFrame()

def fetch_game_stats(game_id):
    """Fetch detailed stats for a specific NHL game"""
    try:
        url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/boxscore"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        home_stats = data['teams']['home']['teamStats']['teamSkaterStats']
        away_stats = data['teams']['away']['teamStats']['teamSkaterStats']
        
        return {
            'home_goals': home_stats.get('goals', 0),
            'home_shots': home_stats.get('shots', 0),
            'home_powerplay_goals': home_stats.get('powerPlayGoals', 0),
            'home_powerplay_opportunities': home_stats.get('powerPlayOpportunities', 0),
            'home_faceoff_wins': home_stats.get('faceOffWinPercentage', 50.0),
            'home_hits': home_stats.get('hits', 0),
            'home_giveaways': home_stats.get('giveaways', 0),
            'home_takeaways': home_stats.get('takeaways', 0),
            'away_goals': away_stats.get('goals', 0),
            'away_shots': away_stats.get('shots', 0),
            'away_powerplay_goals': away_stats.get('powerPlayGoals', 0),
            'away_powerplay_opportunities': away_stats.get('powerPlayOpportunities', 0),
            'away_faceoff_wins': away_stats.get('faceOffWinPercentage', 50.0),
            'away_hits': away_stats.get('hits', 0),
            'away_giveaways': away_stats.get('giveaways', 0),
            'away_takeaways': away_stats.get('takeaways', 0)
        }
    except Exception as e:
        print(f"Error fetching stats for game {game_id}: {e}")
        return {}

# --- 2. Process Data Season-by-Season ---
all_seasons_df = []
for year in range(start_season, end_season + 1):
    season_str = get_nhl_season_format(year)
    print(f"Processing NHL season {season_str}...")
    
    # Fetch schedule data
    schedule_df = fetch_nhl_games(season_str)
    if schedule_df.empty:
        print(f"No data available for season {season_str}. Skipping.")
        continue
    
    # Add home_win column
    schedule_df['home_win'] = (schedule_df['home_score'] > schedule_df['away_score']).astype(int)
    
    # Fetch detailed game stats for feature engineering
    game_stats_list = []
    for idx, row in schedule_df.iterrows():
        stats = fetch_game_stats(row['game_id'])
        if stats:
            stats['game_id'] = row['game_id']
            game_stats_list.append(stats)
        
        # Rate limiting to avoid overwhelming the API
        if idx % 10 == 0:
            time.sleep(1)
    
    if not game_stats_list:
        print(f"No game stats available for season {season_str}. Skipping.")
        continue
    
    game_stats_df = pd.DataFrame(game_stats_list)
    
    # Merge schedule with game stats
    final_df_season = pd.merge(schedule_df, game_stats_df, on='game_id', how='left')
    
    # Calculate team-level rolling averages for features
    # We'll use simpler metrics that can be calculated from the available data
    teams = pd.concat([schedule_df['home_team'], schedule_df['away_team']]).unique()
    
    # Create game-by-game team performance tracking
    team_games = []
    for _, row in final_df_season.iterrows():
        # Home team game
        team_games.append({
            'game_id': row['game_id'],
            'date': row['date'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'is_home': 1,
            'goals_for': row['home_goals'],
            'goals_against': row['away_goals'],
            'shots_for': row.get('home_shots', 0),
            'shots_against': row.get('away_shots', 0),
            'pp_goals': row.get('home_powerplay_goals', 0),
            'pp_opportunities': row.get('home_powerplay_opportunities', 0),
            'giveaways': row.get('home_giveaways', 0),
            'takeaways': row.get('home_takeaways', 0),
            'win': 1 if row['home_score'] > row['away_score'] else 0
        })
        
        # Away team game
        team_games.append({
            'game_id': row['game_id'],
            'date': row['date'],
            'team': row['away_team'],
            'opponent': row['home_team'],
            'is_home': 0,
            'goals_for': row['away_goals'],
            'goals_against': row['home_goals'],
            'shots_for': row.get('away_shots', 0),
            'shots_against': row.get('home_shots', 0),
            'pp_goals': row.get('away_powerplay_goals', 0),
            'pp_opportunities': row.get('away_powerplay_opportunities', 0),
            'giveaways': row.get('away_giveaways', 0),
            'takeaways': row.get('home_takeaways', 0),  # Opponent's takeaways
            'win': 1 if row['away_score'] > row['home_score'] else 0
        })
    
    team_games_df = pd.DataFrame(team_games)
    team_games_df['date'] = pd.to_datetime(team_games_df['date'])
    team_games_df = team_games_df.sort_values(['team', 'date'])
    
    # Calculate rolling averages (last 10 games)
    roll_window = 10
    rolling_stats = team_games_df.groupby('team').apply(
        lambda x: x.assign(
            avg_goals_for=x['goals_for'].shift(1).rolling(roll_window, min_periods=3).mean(),
            avg_goals_against=x['goals_against'].shift(1).rolling(roll_window, min_periods=3).mean(),
            avg_shots_for=x['shots_for'].shift(1).rolling(roll_window, min_periods=3).mean(),
            avg_shots_against=x['shots_against'].shift(1).rolling(roll_window, min_periods=3).mean(),
            avg_pp_percentage=((x['pp_goals'] / (x['pp_opportunities'] + 0.01)).shift(1).rolling(roll_window, min_periods=3).mean()),
            avg_giveaway_diff=(x['takeaways'] - x['giveaways']).shift(1).rolling(roll_window, min_periods=3).mean()
        )
    ).reset_index(drop=True)
    
    # Merge rolling stats back to games
    home_stats = rolling_stats[rolling_stats['is_home'] == 1][
        ['game_id', 'avg_goals_for', 'avg_goals_against', 'avg_shots_for', 'avg_shots_against', 
         'avg_pp_percentage', 'avg_giveaway_diff']
    ].add_prefix('home_')
    
    away_stats = rolling_stats[rolling_stats['is_home'] == 0][
        ['game_id', 'avg_goals_for', 'avg_goals_against', 'avg_shots_for', 'avg_shots_against',
         'avg_pp_percentage', 'avg_giveaway_diff']
    ].add_prefix('away_')
    
    # Merge with main dataset
    final_df_season = pd.merge(final_df_season, home_stats, on='game_id', how='left')
    final_df_season = pd.merge(final_df_season, away_stats, on='game_id', how='left')
    
    # Calculate differentials for model features
    final_df_season['goals_for_diff'] = final_df_season['home_avg_goals_for'] - final_df_season['away_avg_goals_for']
    final_df_season['goals_against_diff'] = final_df_season['away_avg_goals_against'] - final_df_season['home_avg_goals_against']
    final_df_season['shots_diff'] = final_df_season['home_avg_shots_for'] - final_df_season['away_avg_shots_for']
    final_df_season['powerplay_diff'] = final_df_season['home_avg_pp_percentage'] - final_df_season['away_avg_pp_percentage']
    final_df_season['giveaway_diff'] = final_df_season['home_avg_giveaway_diff'] - final_df_season['away_avg_giveaway_diff']
    
    all_seasons_df.append(final_df_season)
    
    # Clean up memory
    del final_df_season, game_stats_df, rolling_stats
    gc.collect()
    
    print(f"Completed season {season_str}")

# --- 3. Combine All Seasons and Train ---
print("Combining all seasons and training models...")
final_df = pd.concat(all_seasons_df, ignore_index=True)
final_df = final_df.sort_values(['season', 'date'])

# Resolve models directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

target = 'home_win'

# Train model without odds first (fallback model)
print("Training NHL Model (statistics-based)...")
features_no_odds = ['goals_for_diff', 'goals_against_diff', 'shots_diff', 'powerplay_diff', 'giveaway_diff']
df_no_odds = final_df.dropna(subset=features_no_odds + [target]).copy()

if len(df_no_odds) == 0:
    print("No data available for training. Please check data collection.")
    exit(1)

X = df_no_odds[features_no_odds]
y = df_no_odds[target]

# Time-aware split: last 20% as calibration/validation
split_idx = int(0.8 * len(df_no_odds))
X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

# Train XGBoost model
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

print(f"Training on {len(X_train)} games, validating on {len(X_calib)} games")
model_no_odds.fit(X_train, y_train, eval_set=[(X_calib, y_calib)], verbose=False)

# Calibrate the model
calibrated_no_odds = CalibratedClassifierCV(model_no_odds, method='isotonic', cv='prefit')
calibrated_no_odds.fit(X_calib, y_calib)

# Add conformal prediction for uncertainty quantification
conformal_model_no_odds = MapieClassifier(
    estimator=calibrated_no_odds, 
    method="lac",
    cv="prefit"
)
conformal_model_no_odds.fit(X_calib, y_calib)

# Save models
joblib.dump(conformal_model_no_odds, os.path.join(MODELS_DIR, 'nhl_win_predictor_no_odds.pkl'), protocol=2)
joblib.dump(features_no_odds, os.path.join(MODELS_DIR, 'features_no_odds.pkl'), protocol=2)
print("NHL statistics-based model saved.")

# For now, create a placeholder odds-based model (same as no-odds model)
# In a real implementation, you would integrate with betting odds APIs
print("Creating placeholder model with odds (same features for now)...")
joblib.dump(conformal_model_no_odds, os.path.join(MODELS_DIR, 'nhl_win_predictor_with_odds.pkl'), protocol=2)
joblib.dump(features_no_odds, os.path.join(MODELS_DIR, 'features_with_odds.pkl'), protocol=2)

print("NHL model training complete! 🏒")
print(f"Trained on {len(final_df)} games across {len(all_seasons_df)} seasons")