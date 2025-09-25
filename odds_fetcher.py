# odds_fetcher.py
"""
Live NFL odds fetcher for real-time edge computation
Uses multiple sources with fallbacks for reliability
"""
import requests
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import os
import time

logger = logging.getLogger(__name__)

# Team name mappings to standardize between different odds providers
TEAM_MAPPINGS = {
    # Common variations to our abbreviations
    'Arizona Cardinals': 'ARI', 'Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL', 'Falcons': 'ATL', 
    'Baltimore Ravens': 'BAL', 'Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Panthers': 'CAR',
    'Chicago Bears': 'CHI', 'Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Bengals': 'CIN',
    'Cleveland Browns': 'CLE', 'Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Broncos': 'DEN',
    'Detroit Lions': 'DET', 'Lions': 'DET',
    'Green Bay Packers': 'GB', 'Packers': 'GB',
    'Houston Texans': 'HOU', 'Texans': 'HOU',
    'Indianapolis Colts': 'IND', 'Colts': 'IND',
    'Jacksonville Jaguars': 'JAX', 'Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Chiefs': 'KC',
    'Los Angeles Chargers': 'LAC', 'Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'Rams': 'LAR',
    'Las Vegas Raiders': 'LV', 'Raiders': 'LV',
    'Miami Dolphins': 'MIA', 'Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'Vikings': 'MIN',
    'New England Patriots': 'NE', 'Patriots': 'NE',
    'New Orleans Saints': 'NO', 'Saints': 'NO',
    'New York Giants': 'NYG', 'Giants': 'NYG',
    'New York Jets': 'NYJ', 'Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI', 'Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT', 'Steelers': 'PIT',
    'Seattle Seahawks': 'SEA', 'Seahawks': 'SEA',
    'San Francisco 49ers': 'SF', '49ers': 'SF',
    'Tampa Bay Buccaneers': 'TB', 'Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Titans': 'TEN',
    'Washington Commanders': 'WAS', 'Commanders': 'WAS',
}

def normalize_team_name(team_name):
    """Convert team name to our standard abbreviation"""
    if team_name in TEAM_MAPPINGS:
        return TEAM_MAPPINGS[team_name]
    
    # If already an abbreviation, return as-is
    if len(team_name) <= 3 and team_name.upper() in ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']:
        return team_name.upper()
    
    # Fuzzy matching for common variations
    team_lower = team_name.lower()
    for full_name, abbr in TEAM_MAPPINGS.items():
        if team_lower in full_name.lower() or full_name.lower() in team_lower:
            return abbr
    
    logger.warning(f"Could not normalize team name: {team_name}")
    return team_name

@lru_cache(maxsize=10)
def fetch_odds_api(api_key=None, sport='americanfootball_nfl'):
    """
    Fetch odds from The Odds API (api.the-odds-api.com)
    Free tier: 500 requests/month, good for testing
    """
    if not api_key:
        api_key = os.environ.get('ODDS_API_KEY')
        if not api_key:
            logger.info("No ODDS_API_KEY found, skipping live odds")
            return {}
    
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        odds_data = response.json()
        logger.info(f"Fetched {len(odds_data)} games from Odds API")
        
        # Transform to our format
        games = {}
        for game in odds_data:
            home_team = normalize_team_name(game['home_team'])
            away_team = normalize_team_name(game['away_team'])
            
            if not home_team or not away_team:
                continue
                
            game_key = f"{away_team}@{home_team}"
            
            # Extract odds from bookmakers (use first available)
            spread_line = None
            total_line = None
            home_ml = None
            away_ml = None
            
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads' and spread_line is None:
                        for outcome in market['outcomes']:
                            if normalize_team_name(outcome['name']) == home_team:
                                spread_line = outcome['point']
                    elif market['key'] == 'totals' and total_line is None:
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                total_line = outcome['point']
                    elif market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            team = normalize_team_name(outcome['name'])
                            if team == home_team and home_ml is None:
                                home_ml = outcome['price']
                            elif team == away_team and away_ml is None:
                                away_ml = outcome['price']
            
            games[game_key] = {
                'home_team': home_team,
                'away_team': away_team,
                'spread_line': spread_line,
                'total_line': total_line,
                'home_moneyline': home_ml,
                'away_moneyline': away_ml,
                'source': 'odds_api',
                'timestamp': datetime.now().isoformat()
            }
            
        return games
        
    except Exception as e:
        logger.error(f"Odds API fetch failed: {e}")
        return {}

def fetch_espn_odds():
    """
    Fallback: Scrape ESPN NFL odds (free but less reliable)
    """
    try:
        url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        games = {}
        
        for event in data.get('events', []):
            competitors = event.get('competitions', [{}])[0].get('competitors', [])
            if len(competitors) != 2:
                continue
                
            home_competitor = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_competitor = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            
            if not home_competitor or not away_competitor:
                continue
                
            home_team = normalize_team_name(home_competitor['team']['abbreviation'])
            away_team = normalize_team_name(away_competitor['team']['abbreviation'])
            
            game_key = f"{away_team}@{home_team}"
            
            # ESPN doesn't always have complete odds, but get what we can
            odds_data = event.get('competitions', [{}])[0].get('odds', [{}])
            if odds_data:
                odds = odds_data[0]
                games[game_key] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'spread_line': odds.get('spread'),
                    'total_line': odds.get('overUnder'),
                    'home_moneyline': None,  # ESPN API doesn't consistently provide ML
                    'away_moneyline': None,
                    'source': 'espn',
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info(f"Fetched {len(games)} games from ESPN")
        return games
        
    except Exception as e:
        logger.error(f"ESPN odds fetch failed: {e}")
        return {}

@lru_cache(maxsize=1)
def get_live_odds(use_cache_minutes=10):
    """
    Get live NFL odds with caching and multiple source fallbacks
    Returns dict of game_key -> odds_data
    """
    cache_key = f"live_odds_{int(time.time() / (use_cache_minutes * 60))}"
    
    # Try primary source (Odds API) first
    odds_api_key = os.environ.get('ODDS_API_KEY')
    games = {}
    
    if odds_api_key:
        logger.info("Fetching live odds from Odds API")
        games = fetch_odds_api(odds_api_key)
    
    # Fallback to ESPN if primary failed or no API key
    if not games:
        logger.info("Falling back to ESPN for odds")
        games = fetch_espn_odds()
    
    # If still no data, return empty but log it
    if not games:
        logger.warning("No live odds available from any source")
    else:
        logger.info(f"Successfully fetched odds for {len(games)} games")
    
    return games

def get_game_odds(home_team, away_team):
    """
    Get odds for a specific matchup
    Returns dict with spread_line, total_line, home_moneyline, away_moneyline, implied_home_prob
    """
    try:
        all_odds = get_live_odds()
        game_key = f"{away_team}@{home_team}"
        
        if game_key in all_odds:
            odds = all_odds[game_key]
            
            # Calculate implied probability from moneylines if available
            implied_home_prob = None
            home_ml = odds.get('home_moneyline')
            away_ml = odds.get('away_moneyline')
            
            if home_ml is not None and away_ml is not None:
                def ml_to_prob(ml):
                    if ml < 0:
                        return (-ml) / ((-ml) + 100)
                    else:
                        return 100 / (ml + 100)
                
                try:
                    h_prob = ml_to_prob(home_ml)
                    a_prob = ml_to_prob(away_ml)
                    if (h_prob + a_prob) > 0:
                        implied_home_prob = h_prob / (h_prob + a_prob)
                except Exception as e:
                    logger.warning(f"Moneyline probability calculation failed: {e}")
            
            return {
                'spread_line': odds.get('spread_line'),
                'total_line': odds.get('total_line'),
                'home_moneyline': home_ml,
                'away_moneyline': away_ml,
                'implied_home_prob': implied_home_prob,
                'source': odds.get('source', 'unknown'),
                'timestamp': odds.get('timestamp')
            }
    
    except Exception as e:
        logger.error(f"Failed to get odds for {home_team} vs {away_team}: {e}")
    
    return {}

if __name__ == "__main__":
    # Test the odds fetcher
    logging.basicConfig(level=logging.INFO)
    odds = get_live_odds()
    print(f"Fetched odds for {len(odds)} games:")
    for game_key, data in list(odds.items())[:3]:  # Show first 3
        print(f"  {game_key}: {data}")