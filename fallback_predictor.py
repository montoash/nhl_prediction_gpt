# fallback_predictor.py
"""
Ultra-lightweight fallback predictor for memory-constrained environments
Uses only team abbreviation-based heuristics when full NFL data can't be loaded
"""
import pandas as pd
import numpy as np

def get_fallback_features(home_team, away_team):
    """Generate features using only team heuristics when NFL data unavailable"""
    
    # Simple team strength heuristics based on recent performance
    team_strength = {
        # AFC powerhouses
        'KC': 0.15, 'BUF': 0.12, 'CIN': 0.08, 'BAL': 0.10,
        # AFC contenders  
        'MIA': 0.05, 'PIT': 0.03, 'LAC': 0.02, 'TEN': 0.01,
        # AFC middle
        'JAX': 0.0, 'HOU': -0.02, 'IND': -0.01, 'CLE': -0.03,
        # AFC bottom
        'NYJ': -0.08, 'LV': -0.05, 'DEN': -0.06, 'NE': -0.10,
        
        # NFC powerhouses
        'SF': 0.14, 'PHI': 0.11, 'DAL': 0.09, 'DET': 0.07,
        # NFC contenders
        'MIN': 0.04, 'GB': 0.06, 'SEA': 0.03, 'TB': 0.02,
        # NFC middle
        'ATL': 0.0, 'NO': -0.01, 'LAR': 0.01, 'NYG': -0.04,
        # NFC bottom
        'WAS': -0.07, 'CHI': -0.09, 'CAR': -0.11, 'ARI': -0.12
    }
    
    home_strength = team_strength.get(home_team, 0.0)
    away_strength = team_strength.get(away_team, 0.0)
    
    # Home field advantage ~3 points = ~0.05 EPA
    home_advantage = 0.05
    
    # Calculate differentials
    off_epa_diff = home_strength - away_strength + home_advantage
    def_epa_allowed_diff = away_strength - home_strength  # Defense is inverse
    explosive_diff = (home_strength - away_strength) * 0.5  # Explosive plays correlate
    turnover_diff = (away_strength - home_strength) * 0.3   # Better teams turn ball over less
    
    return pd.DataFrame([{
        'off_epa_diff': off_epa_diff,
        'def_epa_allowed_diff': def_epa_allowed_diff, 
        'explosive_diff': explosive_diff,
        'turnover_diff': turnover_diff
    }])