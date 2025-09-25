# app.py

from flask import Flask, request, jsonify, send_from_directory
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
from odds_fetcher import get_game_odds, get_live_odds

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS headers to all responses for GPT Action compatibility
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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
            
            # Load each component individually with detailed logging
            logger.info("Loading model with odds...")
            model_with_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_with_odds.pkl'))
            
            logger.info("Loading features with odds...")
            features_with_odds = joblib.load(os.path.join(MODELS_DIR, 'features_with_odds.pkl'))
            
            logger.info("Loading model without odds...")
            model_no_odds = joblib.load(os.path.join(MODELS_DIR, 'nfl_win_predictor_no_odds.pkl'))
            
            logger.info("Loading features without odds...")
            features_no_odds = joblib.load(os.path.join(MODELS_DIR, 'features_no_odds.pkl'))
            
            logger.info(f"Models loaded successfully - Features with odds: {len(features_with_odds)}, without odds: {len(features_no_odds)}")
            
        except Exception as e:
            import traceback
            logger.error(f"Model loading failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
    """Get features with fallback for memory-constrained environments.
    By default, use ultra-lightweight features to ensure fast responses for Actions.
    Set ENABLE_FULL_FEATURES=1 to attempt heavier nfl_data_py features.
    """
    logger.info(f"Fetching features for {home_team_abbr} vs {away_team_abbr}")

    use_full = os.environ.get('ENABLE_FULL_FEATURES', '0') == '1'
    if not use_full:
        return get_minimal_features_no_data(home_team_abbr, away_team_abbr)

    # Attempt full feature extraction when explicitly enabled
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
                    logger.info(f"{team} stats: EPA={epa:.3f}, Explosive={explosive:.3f}, Turnover={turnover:.3f}")
                else:
                    logger.warning(f"No stats found for {team} in PBP data - using neutral values")
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

# --- Market odds helper (LIVE ODDS) ---
def get_market_odds(home_team_abbr: str, away_team_abbr: str):
    """Fetch LIVE market odds with fallback to cached schedule.
    Prioritizes real-time odds for true edge computation.
    Returns dict with keys: spread_line, total_line, home_moneyline, away_moneyline, implied_home_prob
    """
    try:
        # First try live odds fetcher
        live_odds = get_game_odds(home_team_abbr, away_team_abbr)
        if live_odds and any(live_odds.get(k) is not None for k in ['spread_line', 'total_line', 'home_moneyline']):
            logger.info(f"Using LIVE odds for {home_team_abbr} vs {away_team_abbr} from {live_odds.get('source', 'unknown')}")
            return live_odds
        
        # Fallback to cached schedule data
        logger.info(f"No live odds found, falling back to cached schedule for {home_team_abbr} vs {away_team_abbr}")
        schedule = get_cached_schedule_data()
        if schedule.empty:
            return {}
        current_year = datetime.now().year
        # Prefer current season rows
        sched = schedule[schedule['season'] == current_year] if 'season' in schedule.columns else schedule
        mask = (sched.get('home_team') == home_team_abbr) & (sched.get('away_team') == away_team_abbr)
        game_row = sched.loc[mask].sort_values('week').tail(1)
        if game_row.empty:
            return {}

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

        spread_line = game_row['spread_line'].iloc[0] if 'spread_line' in game_row.columns else None
        total_line = game_row['total_line'].iloc[0] if 'total_line' in game_row.columns else None
        home_ml = game_row['home_moneyline'].iloc[0] if 'home_moneyline' in game_row.columns else None
        away_ml = game_row['away_moneyline'].iloc[0] if 'away_moneyline' in game_row.columns else None
        h = ml_to_prob(home_ml)
        a = ml_to_prob(away_ml)
        implied_home_prob = None
        if h is not None and a is not None and (h + a) > 0:
            implied_home_prob = h / (h + a)

        return {
            'spread_line': spread_line,
            'total_line': total_line,
            'home_moneyline': home_ml,
            'away_moneyline': away_ml,
            'implied_home_prob': implied_home_prob,
            'source': 'cached_schedule'
        }
    except Exception as e:
        logger.warning(f"Market odds fetch failed: {e}")
        return {}

# --- API Endpoints ---
@app.route('/', methods=['GET', 'OPTIONS'])
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

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    import traceback
    import os
    
    health_info = {
        'status': 'unknown',
        'models_loaded': {
            'with_odds': False,
            'without_odds': False
        },
        'debug_info': {}
    }
    
    try:
        # Check if model files exist
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        required_files = [
            'nfl_win_predictor_with_odds.pkl',
            'features_with_odds.pkl', 
            'nfl_win_predictor_no_odds.pkl',
            'features_no_odds.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
            else:
                health_info['debug_info'][f'{file}_size'] = os.path.getsize(file_path)
        
        if missing_files:
            health_info['status'] = 'degraded'
            health_info['debug_info']['missing_files'] = missing_files
            health_info['debug_info']['error'] = f"Missing model files: {missing_files}"
        else:
            # Try to load models
            try:
                load_models()
                health_info['status'] = 'healthy'
                health_info['models_loaded']['with_odds'] = True
                health_info['models_loaded']['without_odds'] = True
                health_info['debug_info']['models_loaded'] = 'success'
            except Exception as model_error:
                health_info['status'] = 'degraded'
                health_info['debug_info']['model_error'] = str(model_error)
                health_info['debug_info']['model_traceback'] = traceback.format_exc()
        
        health_info['debug_info']['models_dir_exists'] = os.path.exists(models_dir)
        health_info['debug_info']['cwd'] = os.getcwd()
        health_info['debug_info']['python_version'] = os.sys.version
        
    except Exception as e:
        health_info['status'] = 'error'
        health_info['debug_info']['health_check_error'] = str(e)
        health_info['debug_info']['health_check_traceback'] = traceback.format_exc()
    
    return jsonify(health_info)

@app.route('/debug', methods=['GET', 'OPTIONS'])
def debug_info():
    """Debug endpoint to check deployment status"""
    import os
    import sys
    
    debug_data = {
        'deployment_info': {
            'cwd': os.getcwd(),
            'python_version': sys.version,
            'python_path': sys.path[:3]  # First 3 entries
        },
        'file_system': {},
        'environment': {
            'PORT': os.environ.get('PORT', 'Not set'),
            'GAE_ENV': os.environ.get('GAE_ENV', 'Not set'),
            'K_SERVICE': os.environ.get('K_SERVICE', 'Not set')
        }
    }
    
    # Check key directories
    for directory in ['.', 'models', 'scripts']:
        dir_path = os.path.join(os.getcwd(), directory)
        if os.path.exists(dir_path):
            debug_data['file_system'][directory] = os.listdir(dir_path)
        else:
            debug_data['file_system'][directory] = 'Does not exist'
    
    return jsonify(debug_data)

@app.route('/openapi', methods=['GET', 'OPTIONS'])
@app.route('/openapi.yaml', methods=['GET', 'OPTIONS'])
def serve_openapi_spec():
    """Serve the OpenAPI specification file for Action import."""
    import os
    repo_root = os.path.dirname(os.path.abspath(__file__))
    # Prefer explicit YAML mimetype for better compatibility with Action importers
    try:
        return send_from_directory(directory=repo_root, path='openapi.yaml', mimetype='application/yaml')
    except TypeError:
        # Fallback for older Flask versions where send_from_directory uses filename arg
        return send_from_directory(directory=repo_root, filename='openapi.yaml', mimetype='application/yaml')

@app.route('/odds', methods=['GET', 'OPTIONS'])
def get_odds():
    """Get current live odds for all NFL games"""
    try:
        home_team = request.args.get('home')
        away_team = request.args.get('away')
        
        if home_team and away_team:
            # Get odds for specific game
            odds = get_game_odds(home_team, away_team)
            if odds:
                return jsonify({
                    'game': f"{away_team} @ {home_team}",
                    'odds': odds
                })
            else:
                return jsonify({
                    'game': f"{away_team} @ {home_team}",
                    'odds': None,
                    'message': 'No odds found for this matchup'
                })
        else:
            # Get all current odds
            all_odds = get_live_odds()
            return jsonify({
                'total_games': len(all_odds),
                'games': all_odds,
                'last_updated': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Odds endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET', 'OPTIONS'])
def predict():
    logger.info("Prediction request received - starting memory optimization")
    
    home_team = request.args.get('home')
    away_team = request.args.get('away')
    # Optional market inputs for immediate edge computation
    q_spread = request.args.get('spread')
    q_total = request.args.get('total')
    q_home_ml = request.args.get('home_ml')
    q_away_ml = request.args.get('away_ml')

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
            # Return a schema-consistent fallback prediction if models can't load
            # Heuristic probability derived from feature differential
            features = features_df.iloc[0]
            off_diff = float(features.get('off_epa_diff', 0) or 0)
            def_diff = float(features.get('def_epa_allowed_diff', 0) or 0)

            # Map offensive differential to a reasonable win probability
            # Centered at 0.5, scaled and clipped to avoid extremes
            home_win_prob = max(0.05, min(0.95, 0.5 + off_diff * 0.15))
            away_win_prob = 1 - home_win_prob

            import math
            # Convert probability to implied spread
            if home_win_prob > 0.999:
                home_win_prob = 0.999
            elif home_win_prob < 0.001:
                home_win_prob = 0.001
            predicted_spread = -14 * math.log((1 - home_win_prob) / home_win_prob)

            # Predict total points based on offensive/defensive diffs
            base_total = 45
            total_adjustment = (off_diff - def_diff) * 10
            predicted_total = base_total + total_adjustment
            home_score = (predicted_total + predicted_spread) / 2
            away_score = (predicted_total - predicted_spread) / 2
            home_score = max(10, min(50, home_score))
            away_score = max(10, min(50, away_score))
            predicted_total = home_score + away_score

            # Try to enrich with cached market odds
            market = {} if (q_spread or q_total or q_home_ml or q_away_ml) else get_market_odds(home_team, away_team)
            actual_spread = None
            try:
                actual_spread = float(q_spread) if q_spread is not None else (market.get('spread_line') if market else features.get('spread_line'))
            except Exception:
                actual_spread = (market.get('spread_line') if market else features.get('spread_line'))
            actual_total = None
            try:
                actual_total = float(q_total) if q_total is not None else (market.get('total_line') if market else features.get('total_line'))
            except Exception:
                actual_total = (market.get('total_line') if market else features.get('total_line'))

            # Recommendations without models
            spread_recommendation = "No Line Available"
            spread_confidence = "N/A"
            if actual_spread is not None and not pd.isna(actual_spread):
                spread_diff = predicted_spread - actual_spread
                if abs(spread_diff) > 3:
                    spread_recommendation = f"Take {'Home' if spread_diff > 0 else 'Away'} ({spread_diff:+.1f} point edge)"
                    spread_confidence = "Medium"
                else:
                    spread_recommendation = "No Strong Edge"
                    spread_confidence = "Low"

            total_recommendation = "No Line Available"
            total_confidence = "N/A"
            if actual_total is not None and not pd.isna(actual_total):
                total_diff = predicted_total - actual_total
                if abs(total_diff) > 3:
                    total_recommendation = f"{'Over' if total_diff > 0 else 'Under'} ({total_diff:+.1f} point edge)"
                    total_confidence = "Medium"
                else:
                    total_recommendation = "No Strong Edge"
                    total_confidence = "Low"

            try:
                if q_home_ml is not None and q_away_ml is not None:
                    hml = float(q_home_ml)
                    aml = float(q_away_ml)
                    def ml_to_prob(ml):
                        return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)
                    h = ml_to_prob(hml)
                    a = ml_to_prob(aml)
                    implied_home_prob_odds = h / (h + a) if (h + a) > 0 else None
                else:
                    implied_home_prob_odds = (market.get('implied_home_prob') if market else features.get('implied_home_prob'))
            except Exception:
                implied_home_prob_odds = (market.get('implied_home_prob') if market else features.get('implied_home_prob'))
            moneyline_recommendation = "No Odds Available"
            moneyline_value = 0
            if implied_home_prob_odds is not None and not pd.isna(implied_home_prob_odds):
                moneyline_value = home_win_prob - float(implied_home_prob_odds)
                if abs(moneyline_value) > 0.05:
                    moneyline_recommendation = f"{'Home' if moneyline_value > 0 else 'Away'} (+{abs(moneyline_value)*100:.1f}% edge)"
                else:
                    moneyline_recommendation = "Fair Value"

            plausible_outcomes = ["Home Win"] if home_win_prob >= 0.55 else (["Away Win"] if home_win_prob <= 0.45 else ["Home Win", "Away Win"])

            # Compute edges when we have market lines
            spread_edge_points = None
            total_edge_points = None
            if actual_spread is not None and not pd.isna(actual_spread):
                spread_edge_points = predicted_spread - actual_spread
            if actual_total is not None and not pd.isna(actual_total):
                total_edge_points = predicted_total - actual_total

            result = {
                'matchup': f"{away_team} @ {home_team}",
                'game_prediction': {
                    'winner': home_team if home_win_prob > 0.5 else away_team,
                    'home_win_probability': f"{home_win_prob:.1%}",
                    'away_win_probability': f"{away_win_prob:.1%}",
                    'confidence_level': plausible_outcomes[0] if len(plausible_outcomes) == 1 else 'Low Confidence'
                },
                'edges': {
                    'spread_points': round(spread_edge_points, 1) if spread_edge_points is not None else 'N/A',
                    'total_points': round(total_edge_points, 1) if total_edge_points is not None else 'N/A',
                    'moneyline_percent': f"{moneyline_value*100:+.1f}%" if moneyline_value != 0 else 'N/A'
                },
                'score_prediction': {
                    'home_team_score': round(home_score, 1),
                    'away_team_score': round(away_score, 1),
                    'predicted_margin': f"{home_team} by {abs(predicted_spread):.1f}" if predicted_spread > 0 else f"{away_team} by {abs(predicted_spread):.1f}",
                    'predicted_total_points': round(predicted_total, 1)
                },
                'betting_analysis': {
                    'spread': {
                        'vegas_line': actual_spread if actual_spread is not None and not pd.isna(actual_spread) else 'N/A',
                        'predicted_spread': round(predicted_spread, 1),
                        'recommendation': spread_recommendation,
                        'confidence': spread_confidence
                    },
                    'total': {
                        'vegas_total': actual_total if actual_total is not None and not pd.isna(actual_total) else 'N/A',
                        'predicted_total': round(predicted_total, 1),
                        'recommendation': total_recommendation,
                        'confidence': total_confidence
                    },
                    'moneyline': {
                        'recommendation': moneyline_recommendation,
                        'model_edge': f"{moneyline_value*100:+.1f}%" if moneyline_value != 0 else 'N/A'
                    }
                },
                'model_details': {
                    'prediction_type': 'Fallback Heuristics (Models Unavailable)',
                    'key_factors': {
                        'offensive_advantage': f"{home_team if off_diff > 0 else away_team} (+{abs(off_diff):.3f} EPA/play)",
                        'defensive_advantage': f"{home_team if def_diff > 0 else away_team} (+{abs(def_diff):.3f} EPA/play allowed)",
                        'home_field_advantage': "+2.5 points (assumed)"
                    },
                    'data_sources': 'Heuristics (no model available)',
                    'features_used': features_df.to_dict('records')[0]
                },
                'error': f"ML models could not be loaded: {str(e)}"
            }

            logger.info(f"Fallback prediction (no models): {result['game_prediction']['winner']} ({result['game_prediction']['home_win_probability']})")
            return jsonify(result)

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

        # Enhanced predictions with scores, spreads, and betting analysis
        away_win_prob = 1 - home_win_prob
        
        # Score prediction based on win probability (ensures consistency)
        # Convert win probability to point spread using logistic regression inverse
        # P(win) = 1 / (1 + exp(-spread/14)) -> spread = -14 * ln((1-p)/p)
        import math
        if home_win_prob > 0.999:
            home_win_prob = 0.999  # Prevent division by zero
        elif home_win_prob < 0.001:
            home_win_prob = 0.001
        
        predicted_spread = -14 * math.log((1 - home_win_prob) / home_win_prob)
        
        # Predict total points based on team features
        features = features_df.iloc[0]
        off_diff = features.get('off_epa_diff', 0)
        def_diff = features.get('def_epa_allowed_diff', 0)
        
        # Total points: league average ~45, adjust based on offensive/defensive strength
        base_total = 45
        total_adjustment = (off_diff - def_diff) * 10  # More offense = higher total
        predicted_total = base_total + total_adjustment
        
        # Calculate individual scores from spread and total
        home_score = (predicted_total + predicted_spread) / 2
        away_score = (predicted_total - predicted_spread) / 2
        
        # Ensure reasonable score bounds
        home_score = max(10, min(50, home_score))
        away_score = max(10, min(50, away_score))
        predicted_total = home_score + away_score
        
        # Betting analysis with LIVE ODDS
        # Prefer explicit query params; else get LIVE market odds; else feature-derived lines
        market = {} if (q_spread or q_total or q_home_ml or q_away_ml) else get_market_odds(home_team, away_team)
        
        # Log odds source for transparency
        if market and market.get('source'):
            logger.info(f"Using {market['source']} odds for edge computation")

        actual_spread = None
        try:
            actual_spread = float(q_spread) if q_spread is not None else (market.get('spread_line') if market else features.get('spread_line'))
        except Exception:
            actual_spread = (market.get('spread_line') if market else features.get('spread_line'))

        actual_total = None
        try:
            actual_total = float(q_total) if q_total is not None else (market.get('total_line') if market else features.get('total_line'))
        except Exception:
            actual_total = (market.get('total_line') if market else features.get('total_line'))
        
        # Compute edges
        spread_edge_points = None
        total_edge_points = None

        # Spread betting recommendation
        spread_recommendation = "No Line Available"
        spread_confidence = "N/A"
        if actual_spread is not None and not pd.isna(actual_spread):
            spread_diff = predicted_spread - actual_spread
            spread_edge_points = spread_diff
            if abs(spread_diff) > 3:
                spread_recommendation = f"Take {'Home' if spread_diff > 0 else 'Away'} ({spread_diff:+.1f} point edge)"
                spread_confidence = "High" if abs(spread_diff) > 6 else "Medium"
            else:
                spread_recommendation = "No Strong Edge"
                spread_confidence = "Low"
        
        # Over/Under recommendation  
        total_recommendation = "No Line Available"
        total_confidence = "N/A"
        if actual_total is not None and not pd.isna(actual_total):
            total_diff = predicted_total - actual_total
            total_edge_points = total_diff
            if abs(total_diff) > 3:
                total_recommendation = f"{'Over' if total_diff > 0 else 'Under'} ({total_diff:+.1f} point edge)"
                total_confidence = "High" if abs(total_diff) > 6 else "Medium"
            else:
                total_recommendation = "No Strong Edge" 
                total_confidence = "Low"
        
        # Moneyline value analysis
        # Derive implied probability from moneylines if provided
        implied_home_prob_odds = None
        try:
            if q_home_ml is not None and q_away_ml is not None:
                hml = float(q_home_ml)
                aml = float(q_away_ml)
                def ml_to_prob(ml):
                    return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)
                h = ml_to_prob(hml)
                a = ml_to_prob(aml)
                if (h + a) > 0:
                    implied_home_prob_odds = h / (h + a)
            else:
                implied_home_prob_odds = (market.get('implied_home_prob') if market else features.get('implied_home_prob'))
        except Exception:
            implied_home_prob_odds = (market.get('implied_home_prob') if market else features.get('implied_home_prob'))
        moneyline_recommendation = "No Odds Available"
        moneyline_value = 0
        if implied_home_prob_odds is not None and not pd.isna(implied_home_prob_odds):
            moneyline_value = home_win_prob - implied_home_prob_odds
            if abs(moneyline_value) > 0.05:  # 5% edge
                moneyline_recommendation = f"{'Home' if moneyline_value > 0 else 'Away'} (+{abs(moneyline_value)*100:.1f}% edge)"
            else:
                moneyline_recommendation = "Fair Value"

        result = {
            'matchup': f"{away_team} @ {home_team}",
            'game_prediction': {
                'winner': home_team if home_win_prob > 0.5 else away_team,
                'home_win_probability': f"{home_win_prob:.1%}",
                'away_win_probability': f"{away_win_prob:.1%}",
                'confidence_level': plausible_outcomes[0] if len(plausible_outcomes) == 1 else 'Low Confidence'
            },
            'edges': {
                'spread_points': round(spread_edge_points, 1) if spread_edge_points is not None else 'N/A',
                'total_points': round(total_edge_points, 1) if total_edge_points is not None else 'N/A',
                'moneyline_percent': f"{moneyline_value*100:+.1f}%" if moneyline_value != 0 else 'N/A'
            },
            'score_prediction': {
                'home_team_score': round(home_score, 1),
                'away_team_score': round(away_score, 1),
                'predicted_margin': f"{home_team} by {abs(predicted_spread):.1f}" if predicted_spread > 0 else f"{away_team} by {abs(predicted_spread):.1f}",
                'predicted_total_points': round(predicted_total, 1)
            },
            'betting_analysis': {
                'spread': {
                    'vegas_line': actual_spread if actual_spread is not None and not pd.isna(actual_spread) else 'N/A',
                    'predicted_spread': round(predicted_spread, 1),
                    'recommendation': spread_recommendation,
                    'confidence': spread_confidence
                },
                'total': {
                    'vegas_total': actual_total if actual_total is not None and not pd.isna(actual_total) else 'N/A', 
                    'predicted_total': round(predicted_total, 1),
                    'recommendation': total_recommendation,
                    'confidence': total_confidence
                },
                'moneyline': {
                    'recommendation': moneyline_recommendation,
                    'model_edge': f"{moneyline_value*100:+.1f}%" if moneyline_value != 0 else 'N/A'
                }
            },
            'model_details': {
                'prediction_type': 'Advanced Analytics Model' if odds_found else 'Team Performance Model',
                'key_factors': {
                    'offensive_advantage': f"{home_team if off_diff > 0 else away_team} (+{abs(off_diff):.3f} EPA/play)",
                    'defensive_advantage': f"{home_team if def_diff > 0 else away_team} (+{abs(def_diff):.3f} EPA/play allowed)",
                    'home_field_advantage': "+2.5 points (included in model)"
                },
                'data_sources': 'NFL play-by-play data (2024 season)' + (' + Vegas odds' if odds_found else ''),
                'features_used': features_df.to_dict('records')[0]
            }
        }
        
        logger.info(f"Prediction completed: {result['game_prediction']['winner']} ({result['game_prediction']['home_win_probability']})")
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