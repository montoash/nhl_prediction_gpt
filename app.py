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

# --- Helper function to get ROLLING AVERAGE features for two teams ---
def get_latest_features(home_team_abbr, away_team_abbr):
    print(f"Fetching live rolling stats for {home_team_abbr} vs {away_team_abbr}...")
    
    current_year = datetime.now().year
    
    # Fetch current season data
    pbp = nfl.import_pbp_data([current_year], downcast=True, cache=True)
    pbp_reg = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
    
    # Calculate per-game stats
    team_stats = pbp_reg.groupby(['posteam', 'week']).agg(
        off_epa_per_play=('epa', 'mean'),
        off_success_rate=('success', 'mean')
    ).reset_index()
    
    # Calculate 3-game rolling average for the entire season so far
    team_stats = team_stats.sort_values(['posteam', 'week']).copy()
    roll_window = 3
    team_stats['pre_epa'] = team_stats.groupby('posteam')['off_epa_per_play'] \
        .transform(lambda s: s.rolling(roll_window, min_periods=1).mean())
    team_stats['pre_success'] = team_stats.groupby('posteam')['off_success_rate'] \
        .transform(lambda s: s.rolling(roll_window, min_periods=1).mean())

    # Get the MOST RECENT rolling average for each team
    home_stats = team_stats[team_stats['posteam'] == home_team_abbr].tail(1)
    away_stats = team_stats[team_stats['posteam'] == away_team_abbr].tail(1)
    
    if home_stats.empty or away_stats.empty:
        raise ValueError("Could not find stats for one or both teams. They may not have played enough regular season games yet.")
        
    home_pre_epa = home_stats['pre_epa'].values[0]
    away_pre_epa = away_stats['pre_epa'].values[0]
    home_pre_success = home_stats['pre_success'].values[0]
    away_pre_success = away_stats['pre_success'].values[0]
    
    # Calculate the feature differences
    epa_diff = home_pre_epa - away_pre_epa
    success_diff = home_pre_success - away_pre_success
    
    # Create a DataFrame in the correct format expected by the model
    feature_df = pd.DataFrame([[epa_diff, success_diff]], columns=model_features)
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

        # Get the underlying XGBoost model to predict probabilities
        xgb_estimator = model.estimator
        probabilities = xgb_estimator.predict_proba(features_df)
        home_win_prob = probabilities[0][1]

        # Use the conformal model to get the prediction set for uncertainty
        y_pred, y_pis = model.predict(features_df)
        
        # Interpret the prediction set
        prediction_set_indices = np.where(y_pis[0])[0]
        outcomes = ["Away Win", "Home Win"]
        plausible_outcomes = [outcomes[i] for i in prediction_set_indices]

        result = {
            'home_team': home_team,
            'away_team': away_team,
            'calibrated_home_win_prob': f"{home_win_prob:.2%}",
            'conformal_prediction_set': plausible_outcomes
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)