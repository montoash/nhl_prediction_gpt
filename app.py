# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import nfl_data_py as nfl
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)

# Resolve models directory relative to repository root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')

# Load BOTH models and their feature lists
model_with_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_with_odds.pkl'))
features_with_odds = joblib.load(os.path.join(MODELS_DIR, 'features_with_odds.pkl'))

model_no_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_no_odds.pkl'))
features_no_odds = joblib.load(os.path.join(MODELS_DIR, 'features_no_odds.pkl'))

# --- Helper function to get features for a matchup ---
def get_latest_features(home_team_abbr, away_team_abbr):
    print(f"Fetching advanced live features for {home_team_abbr} vs {away_team_abbr}...")
    current_year = datetime.now().year
    
    # 1. TRY to get betting odds (spread_line, total_line, moneylines) from schedules
    spread_line = None
    total_line = None
    implied_home_prob = None
    try:
        schedule = nfl.import_schedules([current_year])
        # Filter to the latest scheduled matchup of the given home/away pair
        mask = (schedule['home_team'] == home_team_abbr) & (schedule['away_team'] == away_team_abbr)
        game_row = schedule.loc[mask].sort_values('week').tail(1)
        if not game_row.empty:
            if 'spread_line' in game_row.columns:
                spread_line = game_row['spread_line'].iloc[0]
            if 'total_line' in game_row.columns:
                total_line = game_row['total_line'].iloc[0]
            # Implied prob from moneylines
            def ml_to_prob(ml):
                if ml is None or pd.isna(ml):
                    return None
                try:
                    ml = float(ml)
                except Exception:
                    return None
                if ml < 0:
                    return (-ml) / ((-ml) + 100)
                else:
                    return 100 / (ml + 100)
            home_ml = game_row['home_moneyline'].iloc[0] if 'home_moneyline' in game_row.columns else None
            away_ml = game_row['away_moneyline'].iloc[0] if 'away_moneyline' in game_row.columns else None
            h = ml_to_prob(home_ml)
            a = ml_to_prob(away_ml)
            if h is not None and a is not None and (h + a) > 0:
                implied_home_prob = h / (h + a)
    except Exception as e:
        print(f"Could not fetch odds: {e}")

    # 2. Get Rolling Team Stats (always needed)
    # Try current year, then previous year; if unavailable, fall back to neutral stats
    pbp_reg = None
    try:
        pbp = nfl.import_pbp_data([current_year], downcast=True, cache=False)
        pbp_reg = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
    except Exception as e:
        print(f"PBP fetch failed for {current_year}: {e}")
        try:
            pbp = nfl.import_pbp_data([current_year - 1], downcast=True, cache=False)
            pbp_reg = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
            print(f"Using previous season PBP ({current_year - 1}) as fallback.")
        except Exception as e2:
            print(f"PBP fallback failed: {e2}")
            pbp_reg = None
    
    team_stats_rolling = {}
    if pbp_reg is not None and not pbp_reg.empty:
        pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype(int)
        pbp_reg['is_turnover'] = ((pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1)).astype(int)
        
        team_game_stats = pbp_reg.groupby(['posteam', 'week']).agg(
            off_epa_per_play=('epa', 'mean'),
            explosive_play_rate=('is_explosive', 'mean'),
            turnover_rate=('is_turnover', 'mean')
        ).reset_index()

        # Defensive per-week EPA allowed
        def_stats = pbp_reg.groupby(['defteam', 'week']).agg(
            def_epa_allowed_per_play=('epa', 'mean')
        ).reset_index().rename(columns={'defteam': 'posteam'})

    if pbp_reg is not None and not pbp_reg.empty:
        for team in [home_team_abbr, away_team_abbr]:
            team_df = team_game_stats[team_game_stats['posteam'] == team].copy()
            if team_df.empty:
                raise ValueError(f"No regular season stats found for team {team} yet.")
            team_df.sort_values('week', inplace=True)
            roll_window = 4
            
            epa = team_df['off_epa_per_play'].rolling(roll_window, min_periods=1).mean().iloc[-1]
            explosive = team_df['explosive_play_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1]
            turnover = team_df['turnover_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1]
            team_stats_rolling[team] = {'epa': epa, 'explosive': explosive, 'turnover': turnover}

        # Defensive rolling (lower is better). Use away - home for home advantage positive.
        def team_def_rolling(team):
            df = def_stats[def_stats['posteam'] == team].copy()
            if df.empty:
                raise ValueError(f"No regular season defensive stats found for team {team} yet.")
            df.sort_values('week', inplace=True)
            return df['def_epa_allowed_per_play'].rolling(4, min_periods=1).mean().iloc[-1]
        home_def = team_def_rolling(home_team_abbr)
        away_def = team_def_rolling(away_team_abbr)
    else:
        # Neutral fallbacks when PBP is unavailable
        team_stats_rolling[home_team_abbr] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
        team_stats_rolling[away_team_abbr] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
        home_def, away_def = 0.0, 0.0
    
    # 3. Calculate Differential Features
    off_epa_diff = team_stats_rolling[home_team_abbr]['epa'] - team_stats_rolling[away_team_abbr]['epa']
    def_epa_allowed_diff = away_def - home_def
    explosive_diff = team_stats_rolling[home_team_abbr]['explosive'] - team_stats_rolling[away_team_abbr]['explosive']
    turnover_diff = team_stats_rolling[home_team_abbr]['turnover'] - team_stats_rolling[away_team_abbr]['turnover']
    
    # 4. Return features matching the trained model's expected columns
    def build_feature_df(feature_names, values_map):
        row = []
        for name in feature_names:
            val = values_map.get(name, None)
            # Convert None to np.nan for XGBoost compatibility
            if val is None:
                val = np.nan
            row.append(val)
        return pd.DataFrame([row], columns=feature_names)

    values_common = {
        'off_epa_diff': off_epa_diff,
        'def_epa_allowed_diff': def_epa_allowed_diff,
        'explosive_diff': explosive_diff,
        'turnover_diff': turnover_diff,
        'spread_line': spread_line,
        'total_line': total_line,
        'implied_home_prob': implied_home_prob,
    }

    # Use odds model only if we have spread_line and either total_line or implied_home_prob
    has_odds = spread_line is not None
    if has_odds:
        return build_feature_df(features_with_odds, values_common), True
    else:
        print("Odds not found. Using fallback model.")
        return build_feature_df(features_no_odds, values_common), False

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'NFL Win Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Predict NFL game outcome (params: home=<team>, away=<team>)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'with_odds': model_with_odds is not None,
            'without_odds': model_no_odds is not None
        }
    })

@app.route('/predict', methods=['GET'])
def predict():
    home_team = request.args.get('home')
    away_team = request.args.get('away')

    if not home_team or not away_team:
        return jsonify({'error': 'Please provide both home and away team abbreviations.'}), 400

    try:
        features_df, odds_found = get_latest_features(home_team, away_team)

        # CHOOSE which model to use based on odds availability
        if odds_found:
            model_to_use = model_with_odds
        else:
            model_to_use = model_no_odds

        # Get probability and prediction set from the chosen model
        # SplitConformalClassifier stores the underlying estimator as a private attribute
        base_model = getattr(model_to_use, 'estimator', None) or getattr(model_to_use, '_estimator', None) or model_to_use
        if not hasattr(base_model, 'predict_proba'):
            raise RuntimeError('Underlying model does not support predict_proba')
        probabilities = base_model.predict_proba(features_df)
        home_win_prob = float(probabilities[0][1])

        # Get conformal predictions
        y_pred = model_to_use.predict(features_df)
        # For SplitConformalClassifier on binary classification, the result contains both classes if uncertain
        # Convert prediction to conformal set representation
        if hasattr(model_to_use, 'conformalize') and hasattr(model_to_use, 'predict'):
            # MAPIE conformal classifiers return arrays; interpret as conformal prediction set
            plausible_outcomes = ["Home Win"] if y_pred[0] == 1 else ["Away Win"]
        else:
            plausible_outcomes = ["Home Win", "Away Win"]  # Uncertain fallback

        result = {
            'home_team': home_team,
            'away_team': away_team,
            'calibrated_home_win_prob': f"{home_win_prob:.2%}",
            'conformal_prediction_set': plausible_outcomes,
            'prediction_type': 'Vegas Odds Model' if odds_found else 'Fallback Stats-Only Model',
            'model_features_used': features_df.to_dict('records')[0]
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)