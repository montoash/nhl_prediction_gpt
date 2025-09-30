# fallback_predictor.py
"""
Ultra-lightweight fallback predictor for memory-constrained environments
Uses only team abbreviation-based heuristics when full NHL data can't be loaded
"""
import pandas as pd
import numpy as np

def get_fallback_features(home_team, away_team):
    """Generate features using only team heuristics when NHL data unavailable"""
    
    # Simple team strength heuristics based on recent performance (2023-24 season)
    team_strength = {
        # Atlantic Division
        'FLA': 0.15, 'TOR': 0.10, 'BOS': 0.12, 'TBL': 0.08,
        'BUF': -0.05, 'DET': 0.03, 'OTT': -0.08, 'MTL': -0.12,
        
        # Metropolitan Division
        'NYR': 0.11, 'CAR': 0.09, 'WSH': 0.02, 'PHI': -0.01,
        'PIT': 0.05, 'NYI': -0.03, 'NJD': 0.07, 'CBJ': -0.10,
        
        # Central Division  
        'DAL': 0.13, 'COL': 0.06, 'WPG': 0.04, 'NSH': 0.01,
        'MIN': -0.02, 'STL': -0.04, 'ARI': -0.06, 'CHI': -0.09,
        
        # Pacific Division
        'VAN': 0.08, 'EDM': 0.14, 'LAK': 0.02, 'VGK': 0.07,
        'SEA': -0.01, 'CGY': -0.07, 'SJS': -0.11, 'ANA': -0.13
    }
    
    home_strength = team_strength.get(home_team, 0.0)
    away_strength = team_strength.get(away_team, 0.0)
    
    # Home ice advantage in NHL ~0.55 win percentage = ~0.1 goal differential
    home_advantage = 0.08
    
    # Calculate differentials (NHL-specific metrics)
    goals_for_diff = home_strength - away_strength + home_advantage
    goals_against_diff = away_strength - home_strength  # Defense is inverse
    powerplay_diff = (home_strength - away_strength) * 0.4  # PP correlates with team strength
    penalty_kill_diff = (home_strength - away_strength) * 0.3   # PK correlates with team strength
    
    return pd.DataFrame([{
        'goals_for_diff': goals_for_diff,
        'goals_against_diff': goals_against_diff, 
        'powerplay_diff': powerplay_diff,
        'penalty_kill_diff': penalty_kill_diff
    }])