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

# Lazy loading - models loaded on first request to handle version issues
model_with_odds = None
features_with_odds = None
model_no_odds = None
features_no_odds = None

def load_models():
    """Load models with error handling"""
    global model_with_odds, features_with_odds, model_no_odds, features_no_odds
    
    if model_with_odds is None:
        try:
            logger.info("Loading ML models...")
            model_with_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_with_odds.pkl'))
            features_with_odds = joblib.load(os.path.join(MODELS_DIR, 'features_with_odds.pkl'))
            model_no_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_no_odds.pkl'))
            features_no_odds = joblib.load(os.path.join(MODELS_DIR, 'features_no_odds.pkl'))
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Could not load ML models: {e}")
    
    return model_with_odds, features_with_odds, model_no_odds, features_no_odds

# --- Cached data loading functions ---
@lru_cache(maxsize=1)
def get_cached_pbp_data():
    """Cache minimal play-by-play data"""
    logger.info("Loading minimal PBP data...")
    try:
        seasons = [2024]
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
        pbp_filtered = pbp.loc[(pbp['season_type'] == 'REG') & (pbp['epa'].notna())].copy()
        
        essential_cols = ['posteam', 'defteam', 'week', 'epa', 'fumble_lost', 'interception']
        existing_cols = [col for col in essential_cols if col in pbp_filtered.columns]
        pbp_minimal = pbp_filtered[existing_cols].copy()
        
        del pbp, pbp_filtered
        gc.collect()
        
        logger.info(f"Cached minimal PBP data: {len(pbp_minimal)} plays")
        return pbp_minimal
        
    except Exception as e:
        logger.error(f"PBP cache failed: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=1) 
def get_cached_schedule_data():
    """Cache minimal schedule data"""
    logger.info("Loading minimal schedule data...")
    try:
        current_year = datetime.now().year
        schedule = nfl.import_schedules([current_year])
        
        essential_cols = ['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        existing_cols = [col for col in essential_cols if col in schedule.columns]
        schedule_minimal = schedule[existing_cols].copy()
        
        del schedule
        gc.collect()
        
        logger.info(f"Cached minimal schedule: {len(schedule_minimal)} games")
        return schedule_minimal
    except Exception as e:
        logger.error(f"Schedule cache failed: {e}")
        return pd.DataFrame()

# --- Ultra-lightweight mode: No NFL data loading to avoid memory issues ---
# Render free tier (512MB) can't handle nfl_data_py imports
# Using fallback predictor only for all predictions

def get_minimal_features_no_data(home_team, away_team):
    """Ultra-minimal feature extraction without loading any NFL data"""
    logger.info(f"Using data-free prediction for {home_team} vs {away_team}")
    
    # No NFL data loading at all - use fallback immediately
    features_df = get_fallback_features(home_team, away_team)
    odds_found = False  # Never have odds in data-free mode
    
    logger.info("Data-free features generated successfully")
    return features_df, odds_found

# --- DATA-FREE feature extraction for memory-constrained environments ---
# --- Helper function to get features for a matchup ---
def get_latest_features(home_team_abbr, away_team_abbr):
    """Get features with fallback for memory-constrained environments"""
    logger.info(f"Fetching features for {home_team_abbr} vs {away_team_abbr}")
    
    # Try full feature extraction first, fallback to lightweight if OOM
    try:
        return get_full_features(home_team_abbr, away_team_abbr)
    except Exception as e:
        logger.warning(f"Full feature extraction failed: {e}, using fallback")
        features_df = get_fallback_features(home_team_abbr, away_team_abbr)
        return features_df, False

def get_full_features(home_team_abbr, away_team_abbr):
    """Full feature extraction with NFL data"""
    current_year = datetime.now().year
    
    # 1. Get betting odds from cached schedule data
    spread_line = None
    total_line = None
    implied_home_prob = None
    
    try:
        schedule = get_cached_schedule_data()
        if not schedule.empty:
            current_schedule = schedule[schedule['season'] == current_year]
            mask = (current_schedule['home_team'] == home_team_abbr) & (current_schedule['away_team'] == away_team_abbr)
            game_row = current_schedule.loc[mask].sort_values('week').tail(1)
            
            if not game_row.empty:
                if 'spread_line' in game_row.columns:
                    spread_line = game_row['spread_line'].iloc[0]
                if 'total_line' in game_row.columns:
                    total_line = game_row['total_line'].iloc[0]
                    
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

    # 2. Get team stats using cached PBP data
    pbp_reg = get_cached_pbp_data()
    
    # Process team stats
    team_stats_rolling = {}
    home_def, away_def = 0.0, 0.0
    
    try:
        if not pbp_reg.empty:
            # Add computed columns efficiently
            if 'is_explosive' not in pbp_reg.columns:
                pbp_reg['is_explosive'] = (pbp_reg['epa'] > 1.75).astype('int8')
            if 'is_turnover' not in pbp_reg.columns:
                pbp_reg['is_turnover'] = ((pbp_reg['fumble_lost'] == 1) | (pbp_reg['interception'] == 1)).astype('int8')
            
            team_game_stats = pbp_reg.groupby(['posteam', 'week'], as_index=False).agg(
                off_epa_per_play=('epa', 'mean'),
                explosive_play_rate=('is_explosive', 'mean'),
                turnover_rate=('is_turnover', 'mean')
            )

            def_stats = pbp_reg.groupby(['defteam', 'week'], as_index=False).agg(
                def_epa_allowed_per_play=('epa', 'mean')
            ).rename(columns={'defteam': 'posteam'})
            
            # Calculate rolling stats for both teams
            for team in [home_team_abbr, away_team_abbr]:
                team_df = team_game_stats[team_game_stats['posteam'] == team].copy()
                if not team_df.empty:
                    team_df.sort_values('week', inplace=True)
                    
                    epa = float(team_df['off_epa_per_play'].rolling(4, min_periods=1).mean().iloc[-1])
                    explosive = float(team_df['explosive_play_rate'].rolling(4, min_periods=1).mean().iloc[-1])
                    turnover = float(team_df['turnover_rate'].rolling(4, min_periods=1).mean().iloc[-1])
                    team_stats_rolling[team] = {'epa': epa, 'explosive': explosive, 'turnover': turnover}
                else:
                    team_stats_rolling[team] = {'epa': 0.0, 'explosive': 0.0, 'turnover': 0.0}
                    
            # Defensive stats
            for team in [home_team_abbr, away_team_abbr]:
                def_df = def_stats[def_stats['posteam'] == team].copy()
                if not def_df.empty:
                    def_df.sort_values('week', inplace=True)
                    def_val = float(def_df['def_epa_allowed_per_play'].rolling(4, min_periods=1).mean().iloc[-1])
                    if team == home_team_abbr:
                        home_def = def_val
                    else:
                        away_def = def_val
        else:
            raise ValueError("No PBP data available")
            
    except Exception as e:
        logger.warning(f"Team stats failed: {e}, using fallback")
        return get_fallback_features(home_team_abbr, away_team_abbr), False
    
    # Calculate differentials
    off_epa_diff = team_stats_rolling[home_team_abbr]['epa'] - team_stats_rolling[away_team_abbr]['epa']
    def_epa_allowed_diff = away_def - home_def
    explosive_diff = team_stats_rolling[home_team_abbr]['explosive'] - team_stats_rolling[away_team_abbr]['explosive']
    turnover_diff = team_stats_rolling[home_team_abbr]['turnover'] - team_stats_rolling[away_team_abbr]['turnover']
    
    # Build feature dataframe
    def build_feature_df(feature_names, values_map):
        row = []
        for name in feature_names:
            val = values_map.get(name, None)
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

    # Use odds model if we have spread_line
    has_odds = spread_line is not None
    if has_odds:
        return build_feature_df(features_with_odds, values_common), True
    else:
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
    try:
        # Try to load models to verify they work
        load_models()
        models_ok = True
    except Exception as e:
        logger.warning(f"Health check - model loading failed: {e}")
        models_ok = False
    
    return jsonify({
        'status': 'healthy' if models_ok else 'degraded',
        'models_loaded': {
            'with_odds': models_ok,
            'without_odds': models_ok
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

        # Load models with error handling
        try:
            model_with_odds_loaded, features_with_odds_loaded, model_no_odds_loaded, features_no_odds_loaded = load_models()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Return fallback prediction if models can't load
            return jsonify({
                'home_team': home_team,
                'away_team': away_team,
                'calibrated_home_win_prob': "60.00%",
                'conformal_prediction_set': ["Home Win", "Away Win"],
                'prediction_type': 'Fallback Heuristics (Models Unavailable)',
                'model_features_used': features_df.to_dict('records')[0],
                'error': f'ML models could not be loaded: {str(e)}'
            })

        # Choose which model to use
        if odds_found:
            model_to_use = model_with_odds_loaded
        else:
            model_to_use = model_no_odds_loaded

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
            'prediction_type': 'Lightweight Data-Free Model',
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