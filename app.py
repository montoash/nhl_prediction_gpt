# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import nfl_data_py as nfl
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load the saved model and features list
model = joblib.load('./models/nfl_win_predictor.pkl')
model_features = joblib.load('./models/features.pkl')

# --- Helper function to get ADVANCED features for a matchup ---
def get_latest_features(home_team_abbr, away_team_abbr):
    print(f"Fetching advanced live features for {home_team_abbr} vs {away_team_abbr}...")
    current_year = datetime.now().year
    
    # 1. Get Live Betting Odds
    latest_odds = nfl.import_odds([current_year])
    game_odds = latest_odds[
        (latest_odds['home_team'] == home_team_abbr) & 
        (latest_odds['away_team'] == away_team_abbr)
    ].tail(1)
    
    if game_odds.empty:
        raise ValueError("Could not find live betting odds for this matchup.")
    spread_line = game_odds['spread_line'].values[0]

    # 2. Get Rolling Team Stats
    pbp = nfl.import_pbp_data([current_year], downcast=True, cache=True)
    pbp_reg = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
    
    pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype(int)
    pbp_reg['is_turnover'] = (pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1).astype(int)
    
    team_game_stats = pbp_reg.groupby(['posteam', 'week']).agg(
        off_epa_per_play=('epa', 'mean'),
        explosive_play_rate=('is_explosive', 'mean'),
        turnover_rate=('is_turnover', 'mean')
    ).reset_index()

    team_stats_rolling = {}
    for team in [home_team_abbr, away_team_abbr]:
        team_df = team_game_stats[team_game_stats['posteam'] == team].copy()
        team_df.sort_values('week', inplace=True)
        roll_window = 4
        
        # Calculate rolling average for each stat and get the most recent value
        epa = team_df['off_epa_per_play'].rolling(roll_window, min_periods=1).mean().iloc[-1]
        explosive = team_df['explosive_play_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1]
        turnover = team_df['turnover_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1]
        team_stats_rolling[team] = {'epa': epa, 'explosive': explosive, 'turnover': turnover}
    
    # 3. Calculate Differential Features
    epa_diff = team_stats_rolling[home_team_abbr]['epa'] - team_stats_rolling[away_team_abbr]['epa']
    explosive_diff = team_stats_rolling[home_team_abbr]['explosive'] - team_stats_rolling[away_team_abbr]['explosive']
    turnover_diff = team_stats_rolling[home_team_abbr]['turnover'] - team_stats_rolling[away_team_abbr]['turnover']
    
    # 4. Assemble Final Feature DataFrame
    feature_df = pd.DataFrame([[spread_line, epa_diff, explosive_diff, turnover_diff]], columns=model_features)
    return feature_df

# --- API Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    home_team = request.args.get('home')
    away_team = request.args.get('away')

    if not home_team or not away_team:
        return jsonify({'error': 'Please provide both home and away team abbreviations.'}), 400

    try:
        features_df = get_latest_features(home_team, away_team)

        # Get probability from the underlying XGBoost model
        xgb_estimator = model.estimator
        probabilities = xgb_estimator.predict_proba(features_df)
        home_win_prob = probabilities[0][1]

        # Get prediction set from the conformal model
        y_pred, y_pis = model.predict(features_df)
        prediction_set_indices = np.where(y_pis[0])[0]
        outcomes = ["Away Win", "Home Win"]
        plausible_outcomes = [outcomes[i] for i in prediction_set_indices]

        result = {
            'home_team': home_team,
            'away_team': away_team,
            'calibrated_home_win_prob': f"{home_win_prob:.2%}",
            'conformal_prediction_set': plausible_outcomes,
            'model_features_used': features_df.to_dict('records')[0]
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)