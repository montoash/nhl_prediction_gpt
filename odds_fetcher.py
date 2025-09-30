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
    if not team_name:
        return team_name
    if team_name in TEAM_MAPPINGS:
        return TEAM_MAPPINGS[team_name]
    
    # If already an abbreviation, return as-is (NHL teams)
    nhl_teams = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']
    if len(team_name) <= 3 and team_name.upper() in nhl_teams:
        return team_name.upper()
    
    # Handle common ESPN short codes and variants
    short_map = {
        'SJ': 'SJS', 'LA': 'LAK', 'NJ': 'NJD', 'TB': 'TBL', 'WSH': 'WSH', 'WAS': 'WSH', 'MON': 'MTL'
    }
    up = team_name.upper()
    if up in short_map:
        return short_map[up]

    # Fuzzy matching for common variations
    team_lower = team_name.lower()
    for full_name, abbr in TEAM_MAPPINGS.items():
        if team_lower in full_name.lower() or full_name.lower() in team_lower:
            return abbr
    
    logger.warning(f"Could not normalize team name: {team_name}")
    return team_name

@lru_cache(maxsize=10)
def fetch_odds_api(api_key=None, sports=None):
    """Fetch odds from The Odds API (api.the-odds-api.com) across one or more sport keys.
    Free tier: 500 requests/month. Supports preseason by trying multiple sport keys.
    """
    if not api_key:
        api_key = os.environ.get('ODDS_API_KEY')
        if not api_key:
            logger.info("No ODDS_API_KEY found, skipping live odds")
            return {}

    # Default to NHL regular and preseason
    if not sports:
        # Allow override with ODDS_SPORT_KEYS="icehockey_nhl,icehockey_nhl_preseason"
        env_keys = os.environ.get('ODDS_SPORT_KEYS')
        if env_keys:
            sports = [s.strip() for s in env_keys.split(',') if s.strip()]
        else:
            sports = ['icehockey_nhl', 'icehockey_nhl_preseason']

    try:
        aggregated = {}
        datasets = []
        for sport in sports:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Odds API {sport} returned {resp.status_code}: {resp.text[:200]}")
                continue
            try:
                odds_data = resp.json()
            except Exception as je:
                logger.warning(f"Odds API JSON parse failed for {sport}: {je}")
                continue
            logger.info(f"Fetched {len(odds_data)} games from Odds API for {sport}")
            datasets.append((sport, odds_data))

        # Helpers
        def mean(vals):
            try:
                return sum(vals) / len(vals) if vals else None
            except Exception:
                return None

        def american_to_prob(ml):
            try:
                ml = float(ml)
            except Exception:
                return None
            if ml < 0:
                return (-ml) / ((-ml) + 100)
            else:
                return 100 / (ml + 100)

        # Transform to our format with consensus across books, combining datasets
        for sport, odds_data in datasets:
            for game in odds_data:
                home_team = normalize_team_name(game.get('home_team'))
                away_team = normalize_team_name(game.get('away_team'))
                if not home_team or not away_team:
                    continue
                game_key = f"{away_team}@{home_team}"

                spread_points, total_points = [], []
                home_prices, away_prices = [], []
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

                probs_home = [american_to_prob(p) for p in home_prices if p is not None]
                probs_away = [american_to_prob(p) for p in away_prices if p is not None]
                implied_home_prob = None
                if probs_home and probs_away:
                    per_book_probs = []
                    for ph, pa in zip(probs_home, probs_away):
                        if ph is None or pa is None or (ph + pa) == 0:
                            continue
                        per_book_probs.append(ph / (ph + pa))
                    implied_home_prob = mean(per_book_probs) if per_book_probs else None

                prev = aggregated.get(game_key, {})
                aggregated[game_key] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'spread_line': mean([v for v in [prev.get('spread_line'), mean(spread_points)] if v is not None]),
                    'total_line': mean([v for v in [prev.get('total_line'), mean(total_points)] if v is not None]),
                    'home_moneyline': mean([v for v in [prev.get('home_moneyline'), mean(home_prices)] if v is not None]),
                    'away_moneyline': mean([v for v in [prev.get('away_moneyline'), mean(away_prices)] if v is not None]),
                    'implied_home_prob': implied_home_prob,
                    'source': f"odds_api_consensus:{sport}",
                    'book_count': (prev.get('book_count') or 0) + len(book_list),
                    'timestamp': datetime.now().isoformat()
                }

        # If nothing aggregated, return empty dict
        return aggregated
        
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

def get_live_odds(use_cache_minutes: int = 10, force_refresh: bool = False):
    """Get live NHL odds with caching and multiple source fallbacks.
    Uses a time-bucketed cache so results refresh every `use_cache_minutes`.
    Set force_refresh=True to bypass cache.
    Returns dict of game_key -> odds_data
    """
    bucket = int(time.time() / (max(1, use_cache_minutes) * 60))
    if force_refresh:
        # Unique bucket to bypass cache without disabling decorator
        bucket = int(time.time())
    return _get_live_odds_cached(bucket, use_cache_minutes)


@lru_cache(maxsize=8)
def _get_live_odds_cached(bucket: int, use_cache_minutes: int):
    # Try primary source (Odds API) first
    odds_api_key = os.environ.get('ODDS_API_KEY')
    games = {}

    if odds_api_key:
        logger.info("Fetching live odds from Odds API (NHL + preseason)")
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