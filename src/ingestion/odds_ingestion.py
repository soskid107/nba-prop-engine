"""

Odds API Ingestion



Fetches game spreads and totals from The Odds API.

Maps team names between Odds API and NBA API formats.

"""



from collections import Counter, defaultdict
from datetime import datetime

from typing import Any, Dict, List, Optional



from ..utils.config import get_config

from ..utils.database import DatabaseManager

from ..utils.http_client import SmartHttpClient





class OddsIngestion:

    """Handles market context (spreads/totals) ingestion from The Odds API."""

    

    # Map The Odds API team names to NBA abbreviations

    TEAM_NAME_MAP = {

        # Full names from Odds API -> NBA abbreviations

        'Atlanta Hawks': 'ATL',

        'Boston Celtics': 'BOS',

        'Brooklyn Nets': 'BKN',

        'Charlotte Hornets': 'CHA',

        'Chicago Bulls': 'CHI',

        'Cleveland Cavaliers': 'CLE',

        'Dallas Mavericks': 'DAL',

        'Denver Nuggets': 'DEN',

        'Detroit Pistons': 'DET',

        'Golden State Warriors': 'GSW',

        'Houston Rockets': 'HOU',

        'Indiana Pacers': 'IND',

        'Los Angeles Clippers': 'LAC',

        'Los Angeles Lakers': 'LAL',

        'LA Clippers': 'LAC',

        'LA Lakers': 'LAL',

        'Memphis Grizzlies': 'MEM',

        'Miami Heat': 'MIA',

        'Milwaukee Bucks': 'MIL',

        'Minnesota Timberwolves': 'MIN',

        'New Orleans Pelicans': 'NOP',

        'New York Knicks': 'NYK',

        'Oklahoma City Thunder': 'OKC',

        'Orlando Magic': 'ORL',

        'Philadelphia 76ers': 'PHI',

        'Phoenix Suns': 'PHX',

        'Portland Trail Blazers': 'POR',

        'Sacramento Kings': 'SAC',

        'San Antonio Spurs': 'SAS',

        'Toronto Raptors': 'TOR',

        'Utah Jazz': 'UTA',

        'Washington Wizards': 'WAS',

    }

    

    # Reverse mapping for lookups

    ABBR_TO_NAME = {v: k for k, v in TEAM_NAME_MAP.items()}
    SUPPORTED_BOOKMAKERS = {'fanduel', 'draftkings', 'betmgm'}

    

    def __init__(self, db: Optional[DatabaseManager] = None,

                 http_client: Optional[SmartHttpClient] = None):

        """Initialize Odds ingestion.

        

        Args:

            db: Optional database manager

            http_client: Optional HTTP client (shares caching/limits)

        """

        self.config = get_config()

        self.db = db or DatabaseManager()

        self.http = http_client or SmartHttpClient(self.db)

    

    def _normalize_team_name(self, name: str) -> str:

        """Convert team name to standard abbreviation.

        

        Args:

            name: Team name from Odds API

            

        Returns:

            3-letter abbreviation

        """

        # Direct lookup

        if name in self.TEAM_NAME_MAP:

            return self.TEAM_NAME_MAP[name]

        

        # Already an abbreviation?

        name_upper = name.upper()

        if len(name_upper) <= 4 and name_upper in self.ABBR_TO_NAME:

            return name_upper

        

        # Fuzzy match by checking if team name contains key words

        name_lower = name.lower()

        for full_name, abbr in self.TEAM_NAME_MAP.items():

            # Check if last word (nickname) matches

            nickname = full_name.split()[-1].lower()

            if nickname in name_lower:

                return abbr

        

        # Return original if no match found

        print(f"  [WARN] Unknown team: {name}")

        return name[:3].upper()


    

    def fetch_todays_odds(self, bookmaker: str = 'fanduel') -> List[Dict[str, Any]]:
        """Fetch spreads, totals, AND player props for today's NBA games.
        
        Args:
            bookmaker: Preferred bookmaker (consensus used if not available)
            
        Returns:
            List of odds data (Game Lines) - props are stored in DB side-effect
        """
        print("\n[Odds API] Fetching today's NBA odds (Game Lines & Player Props)...")
        today = datetime.now().strftime('%Y-%m-%d')
        snapshot_time = datetime.now().isoformat()
        existing_summary = self.db.get_player_prop_snapshot_summary(today)
        
        # Fetch events list to get valid event IDs for per-event odds endpoints
        events = self.http.get_odds_api(
            endpoint="/sports/basketball_nba/events",
            params={'regions': 'us'},
            cache_hours=1
        )
        event_info_by_matchup = {}
        if events:
            for ev in events:
                ev_id = ev.get('id')
                h = ev.get('home_team')
                a = ev.get('away_team')
                ct = ev.get('commence_time')
                if ev_id and h and a:
                    h_abbr = self._normalize_team_name(h)
                    a_abbr = self._normalize_team_name(a)
                    event_info_by_matchup[(h_abbr, a_abbr)] = {'id': ev_id, 'commence_time': ct}

        # Fetch odds with spreads and totals (Game Lines only first)
        data = self.http.get_odds_api(
            endpoint="/sports/basketball_nba/odds",
            params={
                'regions': 'us',
                'markets': 'spreads,totals', 
                'oddsFormat': 'american'
            },
            cache_hours=1
        )
        
        if not data:
            print("   Failed to fetch odds data")
            return []
        
        games_data = []
        staged_props: List[Dict[str, Any]] = []
        props_count = 0
        processed_event_ids = set()
        diagnostics = defaultdict(lambda: {
            'fetched_props': 0,
            'drops': Counter(),
            'by_market': Counter(),
            'by_bookmaker': Counter(),
        })
        
        from datetime import timedelta

        def _game_label(home_abbr: str, away_abbr: str, game_date: str) -> str:
            return f"{game_date} | {away_abbr} @ {home_abbr}"

        def _record_drop(game_label: str, market_key: Optional[str], book_key: Optional[str], reason: str) -> None:
            diag = diagnostics[game_label]
            diag['drops'][reason] += 1
            if market_key:
                diag['by_market'][market_key] += 1
            if book_key:
                diag['by_bookmaker'][book_key] += 1
        
        for game in data:
            try:
                game_id_api = game.get('id')
                game_time = game.get('commence_time', '')

                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')

                home_abbr = self._normalize_team_name(home_team)
                away_abbr = self._normalize_team_name(away_team)

                event_info = event_info_by_matchup.get((home_abbr, away_abbr)) or event_info_by_matchup.get((away_abbr, home_abbr))
                event_id_api = event_info.get('id') if event_info else None
                if event_info and event_info.get('commence_time'):
                    game_time = event_info['commence_time']

                # Determine Game Date (Approx ET)
                try:
                    dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                    # Shift to ET (UTC-5) to handle late games correctly
                    game_date = (dt - timedelta(hours=5)).strftime('%Y-%m-%d')
                except (ValueError, TypeError) as e:
                    game_date = today
                game_label = _game_label(self._normalize_team_name(ev.get('home_team', '')), self._normalize_team_name(ev.get('away_team', '')), game_date)
                game_label = _game_label(home_abbr, away_abbr, game_date)
                
                # --- GAME LINES ---
                spread_home = None
                spread_away = None
                total = None
                over_odds = None
                under_odds = None
                
                bookmakers = game.get('bookmakers', [])
                selected_book = None
                
                # Find preferred book for game lines
                for book in bookmakers:
                    if book.get('key') == bookmaker:
                        selected_book = book
                        break
                if not selected_book and bookmakers:
                    selected_book = bookmakers[0]
                
                if selected_book:
                    for market in selected_book.get('markets', []):
                        market_key = market.get('key')
                        outcomes = market.get('outcomes', [])
                        
                        if market_key == 'spreads':
                            for outcome in outcomes:
                                if outcome.get('name') == home_team:
                                    spread_home = outcome.get('point')
                                elif outcome.get('name') == away_team:
                                    spread_away = outcome.get('point')
                        elif market_key == 'totals':
                             for outcome in outcomes:
                                if outcome.get('name') == 'Over':
                                    total = outcome.get('point')
                                    over_odds = outcome.get('price')
                                elif outcome.get('name') == 'Under':
                                    under_odds = outcome.get('price')
                
                odds_record = {
                    'game_id': None, # Internal DB ID (optional link)
                    'game_date': game_date,
                    'home_team': home_abbr,
                    'away_team': away_abbr,
                    'spread_home': spread_home,
                    'spread_away': spread_away,
                    'total': total,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'moneyline_home': None,
                    'moneyline_away': None,
                    'bookmaker': selected_book.get('key', 'unknown') if selected_book else 'none',
                    'snapshot_time': snapshot_time
                }
                self.db.insert_odds_snapshot(odds_record)
                games_data.append(odds_record)
                
                # --- PLAYER PROPS (Fetch per event) ---
                if event_id_api:
                    processed_event_ids.add(event_id_api)
                    # random sleep to be nice to API if not already throttled
                    import time
                    time.sleep(0.5) 
                    
                    # [PHASE 13] Requesting Expanded Markets
                    props_data = self.http.get_odds_api(
                        endpoint=f"/sports/basketball_nba/events/{event_id_api}/odds",
                        params={
                            'regions': 'us',
                            'markets': 'player_points,player_assists,player_rebounds,player_threes,player_blocks,player_steals,player_field_goals',
                            'oddsFormat': 'american'
                        },
                        cache_hours=1
                    )
                    
                    if props_data and 'bookmakers' in props_data:
                        prop_bookmakers = props_data.get('bookmakers', [])
                        
                        # Iterate ALL bookmakers to find props
                        for book in prop_bookmakers:
                            book_key = book.get('key')
                            if book_key not in self.SUPPORTED_BOOKMAKERS: 
                                _record_drop(game_label, None, book_key or 'unknown', 'unsupported_bookmaker')
                                continue 
                                
                            for market in book.get('markets', []):
                                market_key = market.get('key')
                                # [PHASE 13] Allow new market keys
                                allowed_markets = [
                                    'player_points', 'player_assists', 'player_rebounds',
                                    'player_threes', 'player_blocks', 'player_steals',
                                    'player_field_goals' # FGM usually
                                ]
                                if market_key not in allowed_markets:
                                    _record_drop(game_label, market_key, book_key, 'unsupported_market')
                                    continue
                                    
                                for outcome in market.get('outcomes', []):
                                    p_name = outcome.get('description')
                                    line = outcome.get('point')
                                    odds_mk = outcome.get('price')
                                    label = outcome.get('name') # Over/Under
                                    
                                    if not p_name or line is None:
                                        _record_drop(game_label, market_key, book_key, 'missing_player_or_line')
                                        continue
                                    
                                    # Resolve Player ID
                                    player_rec = self._resolve_player_id(
                                        p_name,
                                        valid_teams=[home_abbr, away_abbr]
                                    )
                                    player_id = player_rec['player_id'] if player_rec else None
                                    if player_id is None:
                                        _record_drop(game_label, market_key, book_key, 'unresolved_player')
                                    
                                    prop_record = {
                                        'game_id': event_id_api,
                                        'game_date': game_date,
                                        'player_id': player_id,
                                        'player_name': p_name,
                                        'market_key': market_key,
                                        'bookmaker': book_key,
                                        'line': line,
                                        'odds_over': odds_mk if label == 'Over' else None,
                                        'odds_under': odds_mk if label == 'Under' else None,
                                        'snapshot_time': snapshot_time
                                    }
                                    staged_props.append(prop_record)
                                    props_count += 1
                                    diagnostics[game_label]['fetched_props'] += 1

            except Exception as e:
                print(f"  [WARN] Error parsing game: {e}")
                continue

        # Ensure props are attempted for any events not present in /odds response
        if events:
            from datetime import timedelta
            for ev in events:
                ev_id = ev.get('id')
                if not ev_id or ev_id in processed_event_ids:
                    continue

                commence = ev.get('commence_time', '')
                try:
                    dt = datetime.fromisoformat(commence.replace('Z', '+00:00'))
                    game_date = (dt - timedelta(hours=5)).strftime('%Y-%m-%d')
                except (ValueError, TypeError) as e:
                    game_date = today

                import time
                time.sleep(0.5)

                # [PHASE 13] Requesting Expanded Markets (Fallback Loop)
                props_data = self.http.get_odds_api(
                    endpoint=f"/sports/basketball_nba/events/{ev_id}/odds",
                    params={
                        'regions': 'us',
                        'markets': 'player_points,player_assists,player_rebounds,player_threes,player_blocks,player_steals,player_field_goals',
                        'oddsFormat': 'american'
                    },
                    cache_hours=1
                )

                if not props_data or 'bookmakers' not in props_data:
                    _record_drop(game_label, None, 'all', 'missing_props_payload')
                    continue

                for book in props_data.get('bookmakers', []):
                    book_key = book.get('key')
                    if book_key not in self.SUPPORTED_BOOKMAKERS:
                        _record_drop(game_label, None, book_key or 'unknown', 'unsupported_bookmaker')
                        continue

                    for market in book.get('markets', []):
                        market_key = market.get('key')
                        # [PHASE 13] Allow new market keys
                        allowed_markets = [
                            'player_points', 'player_assists', 'player_rebounds',
                            'player_threes', 'player_blocks', 'player_steals',
                            'player_field_goals'
                        ]
                        if market_key not in allowed_markets:
                            _record_drop(game_label, market_key, book_key, 'unsupported_market')
                            continue

                        for outcome in market.get('outcomes', []):
                            p_name = outcome.get('description')
                            line = outcome.get('point')
                            odds_mk = outcome.get('price')
                            label = outcome.get('name')

                            if not p_name or line is None:
                                _record_drop(game_label, market_key, book_key, 'missing_player_or_line')
                                continue

                            player_rec = self._resolve_player_id(
                                p_name,
                                valid_teams=[
                                    self._normalize_team_name(ev.get('home_team', '')),
                                    self._normalize_team_name(ev.get('away_team', ''))
                                ]
                            )
                            player_id = player_rec['player_id'] if player_rec else None
                            if player_id is None:
                                _record_drop(game_label, market_key, book_key, 'unresolved_player')

                            prop_record = {
                                'game_id': ev_id,
                                'game_date': game_date,
                                'player_id': player_id,
                                'player_name': p_name,
                                'market_key': market_key,
                                'bookmaker': book_key,
                                'line': line,
                                'odds_over': odds_mk if label == 'Over' else None,
                                'odds_under': odds_mk if label == 'Under' else None,
                                'snapshot_time': snapshot_time
                            }
                            staged_props.append(prop_record)
                            props_count += 1
                            diagnostics[game_label]['fetched_props'] += 1

        if props_count > 0:
            self.db.clear_player_prop_odds_for_date(today)
            for prop_record in staged_props:
                self.db.insert_player_prop_odds(prop_record)
            print(f"   Fetched {len(games_data)} games, {props_count} player props")
        else:
            fallback_count = int(existing_summary.get('prop_count') or 0)
            latest_snapshot = existing_summary.get('latest_snapshot_time')
            if fallback_count > 0:
                print(
                    f"   [FALLBACK] Live prop refresh failed; reusing {fallback_count} stored same-day props "
                    f"(latest snapshot: {latest_snapshot})"
                )
            else:
                print(f"   Fetched {len(games_data)} games, 0 player props")

        if diagnostics:
            print("   [PROP DIAGNOSTICS]")
            for game_label in sorted(diagnostics.keys()):
                diag = diagnostics[game_label]
                fetched = diag['fetched_props']
                drops = diag['drops']
                if fetched == 0 and not drops:
                    continue
                bits = []
                if drops:
                    bits.append("drops=" + ", ".join(f"{k}:{v}" for k, v in drops.most_common()))
                if diag['by_market']:
                    bits.append("markets=" + ", ".join(f"{k}:{v}" for k, v in diag['by_market'].most_common()))
                if diag['by_bookmaker']:
                    bits.append("books=" + ", ".join(f"{k}:{v}" for k, v in diag['by_bookmaker'].most_common()))
                suffix = f" | {' ; '.join(bits)}" if bits else ""
                print(f"   - {game_label}: fetched={fetched}{suffix}")
        return games_data

    def _resolve_player_id(self, name: str, valid_teams: Optional[List[str]] = None) -> Optional[Dict]:
        """Resolve player name to DB record using fuzzy match and optional team validation."""
        valid_team_set = {
            str(team).upper()
            for team in (valid_teams or [])
            if team
        }

        def team_ok(player_row: Dict[str, Any]) -> bool:
            if not valid_team_set:
                return True
            return str(player_row.get('team_abbreviation', '')).upper() in valid_team_set

        # Exact match first
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM players WHERE full_name = ?", (name,))
            row = cursor.fetchone()
            if row:
                player = dict(row)
                if team_ok(player):
                    return player
            
            # Simple fuzzy (contains)
            cursor.execute("SELECT * FROM players WHERE full_name LIKE ?", (f"%{name.split()[-1]}%",))
            rows = [dict(r) for r in cursor.fetchall()]
            valid_rows = [r for r in rows if team_ok(r)]

            if len(valid_rows) == 1:
                return valid_rows[0]
            if len(valid_rows) > 1:
                first = name.split()[0].lower()
                narrowed = [r for r in valid_rows if (r['full_name'] or '').lower().startswith(first)]
                if len(narrowed) == 1:
                    return narrowed[0]

            # If strictly one match on last name, take it (risky for 'Green' etc but okay for 'Wembanyama')
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                first = name.split()[0].lower()
                narrowed = [r for r in rows if (r['full_name'] or '').lower().startswith(first)]
                if len(narrowed) == 1:
                    return narrowed[0]
                
        return None

    

    def get_odds_for_game(self, home_team: str, away_team: str, 

                          game_date: str = None) -> Optional[Dict[str, Any]]:

        """Get odds for a specific game.

        

        Args:

            home_team: Home team abbreviation

            away_team: Away team abbreviation

            game_date: Date string (default: today)

            

        Returns:

            Odds dict or None if not found

        """

        game_date = game_date or datetime.now().strftime('%Y-%m-%d')

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM odds_snapshots 

                WHERE game_date = ? AND home_team = ? AND away_team = ?

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, home_team, away_team))

            

            row = cursor.fetchone()

            return dict(row) if row else None

    

    def get_spread_for_team(self, team_abbr: str, game_date: str = None) -> Optional[float]:

        """Get spread for a team's game.

        

        Positive spread = underdog, Negative spread = favorite

        

        Args:

            team_abbr: Team abbreviation

            game_date: Date string (default: today)

            

        Returns:

            Spread value or None

        """

        game_date = game_date or datetime.now().strftime('%Y-%m-%d')

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            # Check if team is home

            cursor.execute("""

                SELECT spread_home FROM odds_snapshots 

                WHERE game_date = ? AND home_team = ?

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, team_abbr))

            row = cursor.fetchone()

            if row and row['spread_home'] is not None:

                return row['spread_home']

            

            # Check if team is away

            cursor.execute("""

                SELECT spread_away FROM odds_snapshots 

                WHERE game_date = ? AND away_team = ?

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, team_abbr))

            row = cursor.fetchone()

            if row and row['spread_away'] is not None:

                return row['spread_away']

        

        return None

    

    def get_total_for_game(self, team_abbr: str, game_date: str = None) -> Optional[float]:

        """Get game total (Over/Under) for a team's game.

        

        Args:

            team_abbr: Team abbreviation (home or away)

            game_date: Date string (default: today)

            

        Returns:

            Total points line or None

        """

        game_date = game_date or datetime.now().strftime('%Y-%m-%d')

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT total FROM odds_snapshots 

                WHERE game_date = ? AND (home_team = ? OR away_team = ?)

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, team_abbr, team_abbr))

            

            row = cursor.fetchone()

            return row['total'] if row else None

    

    def get_api_usage_today(self) -> int:

        """Get number of Odds API calls made today.

        

        Returns:

            Call count

        """

        return self.db.get_api_usage_today("the_odds_api")





# Convenience function

def get_odds_ingestion() -> OddsIngestion:

    """Get Odds ingestion instance."""

    return OddsIngestion()

