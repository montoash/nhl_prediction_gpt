# odds_fetcher.py
"""
Live NHL odds fetcher for real-time edge computation
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
    # NHL Teams - Common variations to our abbreviations
    'Anaheim Ducks': 'ANA', 'Ducks': 'ANA',
    'Arizona Coyotes': 'ARI', 'Coyotes': 'ARI',
    'Boston Bruins': 'BOS', 'Bruins': 'BOS',
    'Buffalo Sabres': 'BUF', 'Sabres': 'BUF',
    'Calgary Flames': 'CGY', 'Flames': 'CGY',
    'Carolina Hurricanes': 'CAR', 'Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI', 'Blackhawks': 'CHI',
    'Colorado Avalanche': 'COL', 'Avalanche': 'COL',
    'Columbus Blue Jackets': 'CBJ', 'Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL', 'Stars': 'DAL',
    'Detroit Red Wings': 'DET', 'Red Wings': 'DET',
    'Edmonton Oilers': 'EDM', 'Oilers': 'EDM',
    'Florida Panthers': 'FLA', 'Panthers': 'FLA',
    'Los Angeles Kings': 'LAK', 'Kings': 'LAK',
    'Minnesota Wild': 'MIN', 'Wild': 'MIN',
    'Montreal Canadiens': 'MTL', 'Canadiens': 'MTL',
    'Nashville Predators': 'NSH', 'Predators': 'NSH',
    'New Jersey Devils': 'NJD', 'Devils': 'NJD',
    'New York Islanders': 'NYI', 'Islanders': 'NYI',
    'New York Rangers': 'NYR', 'Rangers': 'NYR',
    'Ottawa Senators': 'OTT', 'Senators': 'OTT',
    'Philadelphia Flyers': 'PHI', 'Flyers': 'PHI',
    'Pittsburgh Penguins': 'PIT', 'Penguins': 'PIT',
    'San Jose Sharks': 'SJS', 'Sharks': 'SJS',
    'Seattle Kraken': 'SEA', 'Kraken': 'SEA',
    'St. Louis Blues': 'STL', 'Blues': 'STL',
    'Tampa Bay Lightning': 'TBL', 'Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR', 'Maple Leafs': 'TOR',
    'Vancouver Canucks': 'VAN', 'Canucks': 'VAN',
    'Vegas Golden Knights': 'VGK', 'Golden Knights': 'VGK',
    'Washington Capitals': 'WSH', 'Capitals': 'WSH',
    'Winnipeg Jets': 'WPG', 'Jets': 'WPG',
}

def normalize_team_name(team_name):
    """Convert team name to our standard abbreviation"""
    if team_name in TEAM_MAPPINGS:
        return TEAM_MAPPINGS[team_name]
    
    # If already an abbreviation, return as-is (NHL teams)
    nhl_teams = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']
    if len(team_name) <= 3 and team_name.upper() in nhl_teams:
        return team_name.upper()
    
    # Fuzzy matching for common variations
    team_lower = team_name.lower()
    for full_name, abbr in TEAM_MAPPINGS.items():
        if team_lower in full_name.lower() or full_name.lower() in team_lower:
            return abbr
    
    logger.warning(f"Could not normalize team name: {team_name}")
    return team_name

@lru_cache(maxsize=10)
def fetch_odds_api(api_key=None, sport='icehockey_nhl'):
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
        
        # Transform to our format with consensus across books
        games = {}
        for game in odds_data:
            home_team = normalize_team_name(game['home_team'])
            away_team = normalize_team_name(game['away_team'])
            if not home_team or not away_team:
                continue
            game_key = f"{away_team}@{home_team}"

            spread_points = []
            total_points = []
            home_prices = []
            away_prices = []
            book_list = []

            for bookmaker in game.get('bookmakers', []):
                book_key = bookmaker.get('key') or bookmaker.get('title')
                markets = bookmaker.get('markets', [])
                book_spread = None
                book_total = None
                book_home_price = None
                book_away_price = None

                for market in markets:
                    key = market.get('key')
                    outcomes = market.get('outcomes', [])
                    if key == 'spreads':
                        for outcome in outcomes:
                            if normalize_team_name(outcome.get('name', '')) == home_team and outcome.get('point') is not None:
                                book_spread = outcome['point']
                                break
                    elif key == 'totals':
                        # take the Over point as the line
                        for outcome in outcomes:
                            if outcome.get('name') == 'Over' and outcome.get('point') is not None:
                                book_total = outcome['point']
                                break
                    elif key == 'h2h':
                        for outcome in outcomes:
                            team = normalize_team_name(outcome.get('name', ''))
                            if team == home_team and outcome.get('price') is not None:
                                book_home_price = outcome['price']
                            elif team == away_team and outcome.get('price') is not None:
                                book_away_price = outcome['price']

                if book_spread is not None:
                    spread_points.append(book_spread)
                if book_total is not None:
                    total_points.append(book_total)
                if book_home_price is not None:
                    home_prices.append(book_home_price)
                if book_away_price is not None:
                    away_prices.append(book_away_price)
                if book_key:
                    book_list.append(book_key)

            def mean(vals):
                try:
                    return sum(vals) / len(vals) if vals else None
                except Exception:
                    return None

            # Convert American odds to probabilities and average across books to reduce vig bias
            def american_to_prob(ml):
                try:
                    ml = float(ml)
                except Exception:
                    return None
                if ml < 0:
                    return (-ml) / ((-ml) + 100)
                else:
                    return 100 / (ml + 100)

            probs_home = [american_to_prob(p) for p in home_prices if p is not None]
            probs_away = [american_to_prob(p) for p in away_prices if p is not None]
            implied_home_prob = None
            if probs_home and probs_away:
                # Normalize each book’s pair by dividing by sum, then average across books
                per_book_probs = []
                for ph, pa in zip(probs_home, probs_away):
                    if ph is None or pa is None or (ph + pa) == 0:
                        continue
                    per_book_probs.append(ph / (ph + pa))
                implied_home_prob = mean(per_book_probs) if per_book_probs else None

            games[game_key] = {
                'home_team': home_team,
                'away_team': away_team,
                'spread_line': mean(spread_points),
                'total_line': mean(total_points),
                'home_moneyline': mean(home_prices),
                'away_moneyline': mean(away_prices),
                'implied_home_prob': implied_home_prob,
                'source': 'odds_api_consensus',
                'book_count': len(book_list),
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
        url = "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
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
    Get live NHL odds with caching and multiple source fallbacks
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