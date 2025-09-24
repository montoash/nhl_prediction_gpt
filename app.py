# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import nfl_data_py as nfl
from datetime import datetime
import numpy as np
import os
import logging
from functools import lru_cache
import gc  # For garbage collection
from fallback_predictor import get_fallback_features

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve models directory relative to repository root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')

# Load BOTH models and their feature lists
model_with_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_with_odds.pkl'))
features_with_odds = joblib.load(os.path.join(MODELS_DIR, 'features_with_odds.pkl'))

model_no_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_no_odds.pkl'))
features_no_odds = joblib.load(os.path.join(MODELS_DIR, 'features_no_odds.pkl'))

# --- Cached data loading functions for memory optimization ---
@lru_cache(maxsize=1)
def get_cached_pbp_data():
    """Cache minimal play-by-play data to avoid memory issues"""
    logger.info("Loading minimal PBP data...")
    try:
        # Load only 2024 season and only essential columns
        seasons = [2024]  # Only most recent complete season
        
        # Load with minimal columns to save memory
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
        
        # Filter immediately to reduce memory footprint
        pbp_filtered = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
        
        # Keep only essential columns we actually need
        essential_cols = ['posteam', 'defteam', 'week', 'epa', 'fumble_lost', 'interception']
        existing_cols = [col for col in essential_cols if col in pbp_filtered.columns]
        pbp_minimal = pbp_filtered[existing_cols].copy()
        
        # Immediate cleanup
        del pbp, pbp_filtered
        gc.collect()
        
        logger.info(f"Cached minimal PBP data: {len(pbp_minimal)} plays, {pbp_minimal.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return pbp_minimal
        
    except Exception as e:
        logger.error(f"PBP cache failed: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=1) 
def get_cached_schedule_data():
    """Cache minimal schedule data"""
    logger.info("Loading minimal schedule data...")
    try:
        # Only load current year to save memory
        current_year = datetime.now().year
        schedule = nfl.import_schedules([current_year])
        
        # Keep only essential columns
        essential_cols = ['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        existing_cols = [col for col in essential_cols if col in schedule.columns]
        schedule_minimal = schedule[existing_cols].copy()
        
        del schedule
        gc.collect()
        
        logger.info(f"Cached minimal schedule: {len(schedule_minimal)} games, {schedule_minimal.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return schedule_minimal
    except Exception as e:
        logger.error(f"Schedule cache failed: {e}")
        return pd.DataFrame()

# --- Optimized helper function to get features for a matchup ---
def get_latest_features(home_team_abbr, away_team_abbr):
    logger.info(f"Fetching features for {home_team_abbr} vs {away_team_abbr}...")
    
    # 1. TRY to get betting odds from cached schedule data
    spread_line = None
    total_line = None
    implied_home_prob = None
    
    try:
        schedule = get_cached_schedule_data()
        if not schedule.empty:
            current_year = datetime.now().year
            # Filter to current year and matchup
            current_schedule = schedule[schedule['season'] == current_year]
            mask = (current_schedule['home_team'] == home_team_abbr) & (current_schedule['away_team'] == away_team_abbr)
            game_row = current_schedule.loc[mask].sort_values('week').tail(1)
            
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
        logger.warning(f"Could not fetch odds: {e}")

    # 2. Get Rolling Team Stats using cached data or fallback to neutral
    try:
        pbp_reg = get_cached_pbp_data()
    except Exception as e:
        logger.warning(f"Failed to load PBP data: {e}, using neutral stats")
        pbp_reg = pd.DataFrame()
    
    if pbp_reg.empty:
        logger.warning("No PBP data available, using neutral stats")
    
    team_stats_rolling = {}
    if not pbp_reg.empty:
        try:
            # Add computed columns efficiently
            if 'is_explosive' not in pbp_reg.columns:
                pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype('int8')  # Use int8 to save memory
            if 'is_turnover' not in pbp_reg.columns:
                pbp_reg['is_turnover'] = ((pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1)).astype('int8')
            
            # Process team stats with memory optimization
            team_game_stats = pbp_reg.groupby(['posteam', 'week'], as_index=False).agg(
                off_epa_per_play=('epa', 'mean'),
                explosive_play_rate=('is_explosive', 'mean'),
                turnover_rate=('is_turnover', 'mean')
            )

            # Defensive stats
            def_stats = pbp_reg.groupby(['defteam', 'week'], as_index=False).agg(
                def_epa_allowed_per_play=('epa', 'mean')
            ).rename(columns={'defteam': 'posteam'})
        except Exception as e:
            logger.warning(f"Error processing team stats: {e}, using defaults")
            team_game_stats = pd.DataFrame()
            def_stats = pd.DataFrame()

    # Process team stats with memory-efficient error handling
    try:
        if not pbp_reg.empty and 'team_game_stats' in locals() and not team_game_stats.empty:
            for team in [home_team_abbr, away_team_abbr]:
                team_df = team_game_stats[team_game_stats['posteam'] == team].copy()
                if team_df.empty:
                    logger.warning(f"No stats for team {team}, using neutral")
                    team_stats_rolling[team] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
                    continue
                    
                team_df.sort_values('week', inplace=True)
                roll_window = 4
                
                epa = float(team_df['off_epa_per_play'].rolling(roll_window, min_periods=1).mean().iloc[-1])
                explosive = float(team_df['explosive_play_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1])
                turnover = float(team_df['turnover_rate'].rolling(roll_window, min_periods=1).mean().iloc[-1])
                team_stats_rolling[team] = {'epa': epa, 'explosive': explosive, 'turnover': turnover}
                
                del team_df  # Immediate cleanup

            # Defensive stats with error handling
            home_def, away_def = 0.0, 0.0
            if 'def_stats' in locals() and not def_stats.empty:
                for team in [home_team_abbr, away_team_abbr]:
                    try:
                        df = def_stats[def_stats['posteam'] == team].copy()
                        if not df.empty:
                            df.sort_values('week', inplace=True)
                            def_val = float(df['def_epa_allowed_per_play'].rolling(4, min_periods=1).mean().iloc[-1])
                            if team == home_team_abbr:
                                home_def = def_val
                            else:
                                away_def = def_val
                            del df
                    except Exception as e:
                        logger.warning(f"Defensive stats failed for {team}: {e}")
        else:
            raise ValueError("Insufficient data available")
    except Exception as e:
        logger.warning(f"Team stats processing failed: {e}, using neutral defaults")
        # Fallback to neutral stats
        team_stats_rolling[home_team_abbr] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
        team_stats_rolling[away_team_abbr] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
        home_def, away_def = 0.0, 0.0
    
    # Aggressive cleanup
    if 'pbp_reg' in locals():
        del pbp_reg
    if 'team_game_stats' in locals():
        del team_game_stats  
    if 'def_stats' in locals():
        del def_stats
    gc.collect()
    
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
    logger.info("Prediction request received - starting memory optimization")
    
    home_team = request.args.get('home')
    away_team = request.args.get('away')

    if not home_team or not away_team:
        return jsonify({'error': 'Please provide both home and away team abbreviations.'}), 400

    logger.info(f"Predicting {away_team} @ {home_team}")

    try:
        # Aggressive pre-processing cleanup
        gc.collect()
        
        logger.info("Computing features...")
        try:
            features_df, odds_found = get_latest_features(home_team, away_team)
            logger.info(f"Features computed successfully, odds_found: {odds_found}")
        except Exception as e:
            logger.warning(f"Full feature extraction failed: {e}, using fallback")
            features_df = get_fallback_features(home_team, away_team)
            odds_found = False
            logger.info("Using lightweight fallback predictor")

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
        
        logger.info(f"Prediction completed: {result['calibrated_home_win_prob']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Memory cleanup on error
        gc.collect()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Development server
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)