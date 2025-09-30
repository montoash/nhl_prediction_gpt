# app_nhl.py - NHL prediction API

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import requests
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

# Log every incoming request
@app.before_request
def log_request_info():
    try:
        ua = request.headers.get('User-Agent', 'unknown')
        rid = request.headers.get('X-Request-ID') or request.headers.get('X-Correlation-ID')
        logger.info(f"Incoming request: {request.method} {request.path} args={dict(request.args)} ua='{ua}' rid={rid}")
    except Exception:
        pass

# Add CORS headers for GPT Action compatibility
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    requested_headers = request.headers.get('Access-Control-Request-Headers', 'Content-Type,Authorization')
    response.headers['Access-Control-Allow-Headers'] = requested_headers or 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    response.headers['Access-Control-Max-Age'] = '600'
    if not response.headers.get('X-Request-ID'):
        response.headers['X-Request-ID'] = request.headers.get('X-Request-ID', request.headers.get('X-Correlation-ID', 'no-id'))
    return response

# Resolve models directory
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')

# Lazy loading - models loaded on first request
model_with_odds = None
features_with_odds = None
model_no_odds = None
features_no_odds = None

def load_models():
    """Load NHL models with error handling"""
    global model_with_odds, features_with_odds, model_no_odds, features_no_odds
    
    if model_with_odds is None:
        try:
            logger.info("Loading NHL ML models...")
            
            logger.info("Loading model with odds...")
            model_with_odds = joblib.load(os.path.join(MODELS_DIR, 'nhl_win_predictor_with_odds.pkl'))
            
            logger.info("Loading features with odds...")
            features_with_odds = joblib.load(os.path.join(MODELS_DIR, 'features_with_odds.pkl'))
            
            logger.info("Loading model without odds...")
            model_no_odds = joblib.load(os.path.join(MODELS_DIR, 'nhl_win_predictor_no_odds.pkl'))
            
            logger.info("Loading features without odds...")
            features_no_odds = joblib.load(os.path.join(MODELS_DIR, 'features_no_odds.pkl'))
            
            logger.info(f"NHL models loaded successfully - Features with odds: {len(features_with_odds)}, without odds: {len(features_no_odds)}")
            
        except Exception as e:
            import traceback
            logger.error(f"Model loading failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Could not load NHL ML models: {e}")
    
    return model_with_odds, features_with_odds, model_no_odds, features_no_odds

@lru_cache(maxsize=1)
def get_cached_nhl_schedule():
    """Cache current NHL schedule data"""
    logger.info("Loading NHL schedule data...")
    try:
        current_season = "20242025"  # 2024-25 NHL season
        url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={current_season}&gameType=R"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for date_data in data['dates']:
            for game in date_data['games']:
                games.append({
                    'game_id': game['gamePk'],
                    'date': game['gameDate'],
                    'home_team': game['teams']['home']['team']['abbreviation'],
                    'away_team': game['teams']['away']['team']['abbreviation'],
                    'game_state': game['status']['detailedState']
                })
        
        schedule_df = pd.DataFrame(games)
        logger.info(f"Cached NHL schedule: {len(schedule_df)} games")
        return schedule_df
    except Exception as e:
        logger.error(f"NHL schedule cache failed: {e}")
        return pd.DataFrame()

def get_team_stats_from_api(team_abbr, season="20242025"):
    """Fetch team statistics from NHL API"""
    try:
        # Get team stats
        url = f"https://statsapi.web.nhl.com/api/v1/teams?season={season}&stats=statsSingleSeason"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        for team_data in data['teams']:
            if team_data['abbreviation'] == team_abbr:
                stats = team_data['teamStats'][0]['splits'][0]['stat']
                return {
                    'goals_per_game': stats.get('goalsPerGame', 0),
                    'goals_against_per_game': stats.get('goalsAgainstPerGame', 0),
                    'shots_per_game': stats.get('shotsPerGame', 0),
                    'shots_allowed_per_game': stats.get('shotsAllowedPerGame', 0),
                    'powerplay_percentage': stats.get('powerPlayPercentage', 0),
                    'penalty_kill_percentage': stats.get('penaltyKillPercentage', 0),
                    'wins': stats.get('wins', 0),
                    'losses': stats.get('losses', 0),
                    'ot': stats.get('ot', 0)
                }
        return {}
    except Exception as e:
        logger.warning(f"Could not fetch stats for {team_abbr}: {e}")
        return {}

def get_nhl_features(home_team, away_team):
    """Get NHL-specific features for prediction"""
    logger.info(f"Fetching NHL features for {home_team} vs {away_team}")
    
    # Try to get team statistics
    home_stats = get_team_stats_from_api(home_team)
    away_stats = get_team_stats_from_api(away_team)
    
    if not home_stats or not away_stats:
        logger.warning("Could not fetch team stats, using fallback features")
        return get_fallback_features(home_team, away_team), False
    
    # Calculate differentials
    goals_for_diff = home_stats.get('goals_per_game', 0) - away_stats.get('goals_per_game', 0)
    goals_against_diff = away_stats.get('goals_against_per_game', 0) - home_stats.get('goals_against_per_game', 0)
    shots_diff = home_stats.get('shots_per_game', 0) - away_stats.get('shots_per_game', 0)
    powerplay_diff = home_stats.get('powerplay_percentage', 0) - away_stats.get('powerplay_percentage', 0)
    
    # Calculate giveaway differential (proxy using defensive stats)
    penalty_kill_diff = home_stats.get('penalty_kill_percentage', 0) - away_stats.get('penalty_kill_percentage', 0)
    
    features_df = pd.DataFrame([{
        'goals_for_diff': goals_for_diff,
        'goals_against_diff': goals_against_diff,
        'shots_diff': shots_diff,
        'powerplay_diff': powerplay_diff / 100.0,  # Convert percentage to decimal
        'giveaway_diff': penalty_kill_diff / 100.0  # Using PK% as proxy
    }])
    
    logger.info(f"Generated NHL features: goals_for_diff={goals_for_diff:.2f}, goals_against_diff={goals_against_diff:.2f}")
    return features_df, False

# --- API Endpoints ---
@app.route('/', methods=['GET', 'OPTIONS'])
def root():
    return jsonify({
        'message': 'NHL Win Prediction API 🏒',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Predict NHL game outcome (params: home=<team>, away=<team>)',
            '/teams': 'List all NHL teams'
        }
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    health_info = {
        'status': 'unknown',
        'models_loaded': {
            'with_odds': False,
            'without_odds': False
        },
        'memory_usage_mb': None,
        'model_files': []
    }
    
    try:
        # Check model loading
        load_models()
        health_info['models_loaded']['with_odds'] = model_with_odds is not None
        health_info['models_loaded']['without_odds'] = model_no_odds is not None
        
        # Check model files exist
        model_files = []
        for filename in ['nhl_win_predictor_with_odds.pkl', 'nhl_win_predictor_no_odds.pkl', 
                        'features_with_odds.pkl', 'features_no_odds.pkl']:
            path = os.path.join(MODELS_DIR, filename)
            model_files.append({
                'filename': filename,
                'exists': os.path.exists(path),
                'size_mb': round(os.path.getsize(path) / 1024 / 1024, 2) if os.path.exists(path) else 0
            })
        health_info['model_files'] = model_files
        
        # Memory usage
        import psutil
        process = psutil.Process(os.getpid())
        health_info['memory_usage_mb'] = round(process.memory_info().rss / 1024 / 1024, 2)
        
        health_info['status'] = 'healthy'
    except Exception as e:
        health_info['status'] = 'error'
        health_info['error'] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return jsonify(health_info)

@app.route('/teams', methods=['GET', 'OPTIONS'])
def teams():
    """List all NHL teams"""
    nhl_teams = {
        'Atlantic': ['BOS', 'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TBL', 'TOR'],
        'Metropolitan': ['CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WSH'],
        'Central': ['ARI', 'CHI', 'COL', 'DAL', 'MIN', 'NSH', 'STL', 'WPG'],
        'Pacific': ['ANA', 'CGY', 'EDM', 'LAK', 'SEA', 'SJS', 'VAN', 'VGK']
    }
    return jsonify(nhl_teams)

@app.route('/predict', methods=['GET', 'OPTIONS'])
def predict():
    """Predict NHL game outcome"""
    try:
        home_team = request.args.get('home', '').upper()
        away_team = request.args.get('away', '').upper()
        
        if not home_team or not away_team:
            return jsonify({'error': 'Missing required parameters: home and away team abbreviations'}), 400
        
        # Validate team abbreviations
        valid_teams = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 
                      'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 
                      'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 
                      'TOR', 'VAN', 'VGK', 'WSH', 'WPG']
        
        if home_team not in valid_teams or away_team not in valid_teams:
            return jsonify({'error': f'Invalid team abbreviation. Valid teams: {valid_teams}'}), 400
        
        # Load models
        model_with_odds, features_with_odds, model_no_odds, features_no_odds = load_models()
        
        # Get features
        features_df, has_odds = get_nhl_features(home_team, away_team)
        
        # Choose appropriate model
        if has_odds and model_with_odds is not None:
            model = model_with_odds
            feature_names = features_with_odds
            logger.info("Using model with odds")
        else:
            model = model_no_odds
            feature_names = features_no_odds
            logger.info("Using model without odds")
        
        # Ensure features match model expectations
        for feature in feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0.0
        
        features_df = features_df[feature_names]
        
        # Make prediction
        prediction_prob = model.predict_proba(features_df)[0]
        
        home_win_prob = prediction_prob[1] if len(prediction_prob) > 1 else prediction_prob[0]
        away_win_prob = 1 - home_win_prob
        
        # Calculate confidence interval (simplified for mock models)
        try:
            prediction_set = model.predict_set(features_df)
            confidence_interval = prediction_set[0] if len(prediction_set) > 0 else [home_win_prob]
        except AttributeError:
            # Fallback for non-conformal models
            confidence_interval = [max(0.0, home_win_prob - 0.1), min(1.0, home_win_prob + 0.1)]
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': {
                'home_win_probability': round(float(home_win_prob), 4),
                'away_win_probability': round(float(away_win_prob), 4),
                'confidence_interval': [round(float(x), 4) for x in confidence_interval],
                'prediction': home_team if home_win_prob > 0.5 else away_team
            },
            'features_used': feature_names,
            'model_type': 'with_odds' if has_odds else 'without_odds'
        }
        
        logger.info(f"Prediction complete: {home_team} {home_win_prob:.3f} vs {away_team} {away_win_prob:.3f}")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/openapi.yaml')
def openapi_spec():
    """Serve OpenAPI specification"""
    return send_from_directory('.', 'openapi.yaml')

# Optional: Serve a JSON OpenAPI spec directly from the app to avoid stale/static file issues
@app.route('/openapi.json', methods=['GET', 'OPTIONS'])
def openapi_json():
    """Return a minimal OpenAPI JSON schema matching this NHL API.
    Useful for Custom GPT Actions to import directly via URL.
    """
    base_url = request.host_url.rstrip('/')
    schema = {
        "openapi": "3.1.1",
        "info": {
            "title": "NHL Prediction API",
            "version": "1.0.0",
            "description": "Predict NHL game outcomes with model probabilities and simple analysis. This spec is tailored for use as a Custom GPT Action."
        },
        "servers": [{"url": "https://nhl-prediction-gpt-455856529947.us-central1.run.app"}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check for the model service",
                    "operationId": "getHealth",
                    "responses": {
                        "200": {
                            "description": "Service status",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HealthResponse"}}}
                        }
                    }
                }
            },
            "/teams": {
                "get": {
                    "summary": "Get all NHL teams organized by division",
                    "operationId": "getNHLTeams",
                    "responses": {
                        "200": {
                            "description": "NHL teams by division",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TeamsResponse"}}}
                        }
                    }
                }
            },
            "/predict": {
                "get": {
                    "summary": "Predict outcome for an NHL matchup",
                    "operationId": "predictMatchup",
                    "parameters": [
                        {"in": "query", "name": "home", "required": True, "description": "Home team NHL abbreviation (e.g., TOR, BOS)", "schema": {"$ref": "#/components/schemas/TeamAbbr"}},
                        {"in": "query", "name": "away", "required": True, "description": "Away team NHL abbreviation (e.g., MTL, NYR)", "schema": {"$ref": "#/components/schemas/TeamAbbr"}}
                    ],
                    "responses": {
                        "200": {"description": "Prediction result", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PredictResponse"}}}},
                        "400": {"description": "Missing or invalid parameters"},
                        "500": {"description": "Internal error"}
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "TeamAbbr": {
                    "type": "string",
                    "description": "Official NHL team abbreviation",
                    "example": "TOR",
                    "pattern": "^(ANA|ARI|BOS|BUF|CGY|CAR|CHI|COL|CBJ|DAL|DET|EDM|FLA|LAK|MIN|MTL|NSH|NJD|NYI|NYR|OTT|PHI|PIT|SJS|SEA|STL|TBL|TOR|VAN|VGK|WSH|WPG)$"
                },
                "TeamsResponse": {
                    "type": "object",
                    "properties": {
                        "Atlantic": {"type": "array", "items": {"type": "string"}},
                        "Metropolitan": {"type": "array", "items": {"type": "string"}},
                        "Central": {"type": "array", "items": {"type": "string"}},
                        "Pacific": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Overall service status"},
                        "models_loaded": {"type": "object", "properties": {"with_odds": {"type": "boolean"}, "without_odds": {"type": "boolean"}}},
                        "memory_usage_mb": {"type": "number", "format": "float"},
                        "model_files": {"type": "array", "items": {"type": "object", "properties": {"filename": {"type": "string"}, "exists": {"type": "boolean"}, "size_mb": {"type": "number", "format": "float"}}}}
                    }
                },
                "PredictResponse": {
                    "type": "object",
                    "properties": {
                        "home_team": {"$ref": "#/components/schemas/TeamAbbr"},
                        "away_team": {"$ref": "#/components/schemas/TeamAbbr"},
                        "predictions": {
                            "type": "object",
                            "properties": {
                                "home_win_probability": {"type": "number", "format": "float"},
                                "away_win_probability": {"type": "number", "format": "float"},
                                "confidence_interval": {"type": "array", "items": {"type": "number", "format": "float"}},
                                "prediction": {"type": "string", "description": "Team predicted to win"}
                            }
                        },
                        "features_used": {"type": "array", "items": {"type": "string"}},
                        "model_type": {"type": "string", "description": "Model used for prediction (with_odds or without_odds)"}
                    }
                }
            }
        }
    }
    return jsonify(schema)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)