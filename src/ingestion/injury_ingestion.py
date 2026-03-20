"""

Injury Pipeline



Fetches and parses NBA injury reports.

Uses fuzzy matching to map player names to database IDs.

Calculates p_play (probability of playing) based on status.

"""



import re

from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple



try:

    from rapidfuzz import fuzz, process

    FUZZY_AVAILABLE = True

except ImportError:

    FUZZY_AVAILABLE = False

    print("[WARN] rapidfuzz not installed, using basic string matching")



from ..utils.config import get_config
from ..utils.database import DatabaseManager
from ..utils.http_client import SmartHttpClient
from .web_scraper import WebScraper





class InjuryIngestion:

    """Handles injury report ingestion and player status tracking."""

    

    # Alternative sources for injury data (since official PDF is complex)

    INJURY_SOURCES = [

        # ESPN has a public injury page

        "https://www.espn.com/nba/injuries",

        # CBS Sports

        "https://www.cbssports.com/nba/injuries/",

    ]

    

    # Status to probability mapping

    STATUS_PROBABILITY = {

        'OUT': 0.0,

        'DOUBTFUL': 0.25,

        'QUESTIONABLE': 0.50,

        'PROBABLE': 0.90,

        'AVAILABLE': 1.0,

        'DAY-TO-DAY': 0.60,

        'GTD': 0.50,  # Game Time Decision

    }

    

    def __init__(self, db: Optional[DatabaseManager] = None,

                 http_client: Optional[SmartHttpClient] = None):

        """Initialize injury ingestion.

        

        Args:

            db: Optional database manager

            http_client: Optional HTTP client

        """

        self.config = get_config()

        self.db = db or DatabaseManager()

        self.http = http_client or SmartHttpClient(self.db)

        self._player_cache: Dict[str, int] = {}  # name -> player_id

        self._build_player_cache()

    

    def _build_player_cache(self) -> None:

        """Build cache of player names to IDs for fuzzy matching."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("SELECT player_id, full_name FROM players WHERE is_active = 1")

            for row in cursor.fetchall():

                name = row['full_name'].lower().strip()

                self._player_cache[name] = row['player_id']

    

    def _normalize_status(self, status: str) -> str:

        """Normalize injury status string.

        

        Args:

            status: Raw status string

            

        Returns:

            Normalized status (OUT, DOUBTFUL, QUESTIONABLE, etc.)

        """

        status = status.upper().strip()

        

        # Handle common variations

        if 'OUT' in status:

            return 'OUT'

        if 'DOUBT' in status:

            return 'DOUBTFUL'

        if 'QUESTION' in status or status == 'Q':

            return 'QUESTIONABLE'

        if 'PROBABLE' in status or status == 'P':

            return 'PROBABLE'

        if 'GTD' in status or 'GAME TIME' in status:

            return 'GTD'

        if 'DAY' in status:

            return 'DAY-TO-DAY'

        

        return status

    

    def _get_play_probability(self, status: str) -> float:

        """Get probability of playing given status.

        

        Args:

            status: Normalized status string

            

        Returns:

            Probability between 0 and 1

        """

        normalized = self._normalize_status(status)
        
        # Try to get from config first
        if hasattr(self.config, 'injury_probabilities'):
            # simple lookup
            if normalized in self.config.injury_probabilities:
                return self.config.injury_probabilities[normalized]
                
            # Dictionary lookup check (if it's a dict object)
            if isinstance(self.config.injury_probabilities, dict):
                 return self.config.injury_probabilities.get(normalized, 1.0)
                 
        return self.STATUS_PROBABILITY.get(normalized, 1.0)

    

    def _fuzzy_match_player(self, name: str, threshold: int = 80) -> Optional[Tuple[int, str]]:

        """Fuzzy match a player name to database.

        

        Args:

            name: Player name to match

            threshold: Minimum match score (0-100)

            

        Returns:

            Tuple of (player_id, matched_name) or None

        """

        name_lower = name.lower().strip()

        

        # Direct match first

        if name_lower in self._player_cache:

            return (self._player_cache[name_lower], name_lower)

        

        if not FUZZY_AVAILABLE:

            # Basic substring matching

            for cached_name, player_id in self._player_cache.items():

                if name_lower in cached_name or cached_name in name_lower:

                    return (player_id, cached_name)

            return None

        

        # Fuzzy matching with rapidfuzz

        result = process.extractOne(

            name_lower,

            list(self._player_cache.keys()),

            scorer=fuzz.token_sort_ratio

        )

        

        if result and result[1] >= threshold:

            matched_name = result[0]

            return (self._player_cache[matched_name], matched_name)

        

        return None

    

    def parse_injury_data(self, injuries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        """Parse and enrich injury data with player IDs and probabilities.

        

        Args:

            injuries: List of raw injury dicts with 'name', 'team', 'status', 'reason'

            

        Returns:

            List of enriched injury dicts

        """

        today = datetime.now().strftime('%Y-%m-%d')

        enriched = []

        

        for injury in injuries:

            player_name = injury.get('name', '')

            status = injury.get('status', 'UNKNOWN')

            

            # Fuzzy match player

            match = self._fuzzy_match_player(player_name)

            player_id = match[0] if match else None

            

            # Calculate probability

            p_play = self._get_play_probability(status)

            

            record = {

                'player_id': player_id,

                'player_name': player_name,

                'team_abbreviation': injury.get('team'),

                'status': self._normalize_status(status),

                'reason': injury.get('reason', ''),

                'report_date': today,

                'p_play': p_play

            }

            

            enriched.append(record)

        

        return enriched

    def get_latest_refresh_summary(self) -> Dict[str, Any]:
        """Summarize latest stored injury refresh for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        records = self.get_todays_injuries() or []
        if not records:
            return {
                'report_date': today,
                'records': 0,
                'matched_players': 0,
                'latest_source': None,
                'latest_fetched_at': None,
                'is_fresh': False,
            }

        latest_row = max(records, key=lambda row: row.get('fetched_at') or '')
        matched_players = len([row for row in records if row.get('player_id')])
        return {
            'report_date': today,
            'records': len(records),
            'matched_players': matched_players,
            'latest_source': latest_row.get('source_name'),
            'latest_fetched_at': latest_row.get('fetched_at'),
            'is_fresh': bool(latest_row.get('fetched_at')),
        }

    

    def add_manual_injuries(self, injuries: List[Dict[str, Any]]) -> int:

        """Add injury reports manually (for testing or manual entry).

        

        Args:

            injuries: List of injury dicts

            

        Returns:

            Number of injuries stored

        """

        enriched = self.parse_injury_data(injuries)

        

        for injury in enriched:

            self.db.insert_injury_snapshot(injury)

        

        return len(enriched)

    

    def get_todays_injuries(self) -> List[Dict[str, Any]]:

        """Get all injury reports for today.

        

        Returns:

            List of injury records

        """

        today = datetime.now().strftime('%Y-%m-%d')

        return self.db.get_injuries_for_date(today)

    

    def get_player_status(self, player_id: int) -> Tuple[str, float]:

        """Get current injury status and play probability for a player.

        

        Args:

            player_id: NBA player ID

            

        Returns:

            Tuple of (status, p_play). Returns ('AVAILABLE', 1.0) if healthy.

        """

        today = datetime.now().strftime('%Y-%m-%d')

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT status, p_play FROM injury_snapshots 

                WHERE player_id = ? AND report_date = ?

                ORDER BY id DESC LIMIT 1

            """, (player_id, today))

            

            row = cursor.fetchone()

            if row:

                return (row['status'], row['p_play'])

        

        return ('AVAILABLE', 1.0)

    

    def simulate_sample_injuries(self) -> List[Dict[str, Any]]:

        """Create sample injury data for testing.

        

        Returns:

            List of sample injury records

        """

        # Sample injuries for testing the pipeline

        sample_injuries = [

            {'name': 'LeBron James', 'team': 'LAL', 'status': 'Questionable', 'reason': 'Left foot soreness'},

            {'name': 'Stephen Curry', 'team': 'GSW', 'status': 'Probable', 'reason': 'Right knee management'},

            {'name': 'Kevin Durant', 'team': 'PHX', 'status': 'Out', 'reason': 'Left ankle sprain'},

            {'name': 'Joel Embiid', 'team': 'PHI', 'status': 'Doubtful', 'reason': 'Left knee injury'},

            {'name': 'Ja Morant', 'team': 'MEM', 'status': 'GTD', 'reason': 'Shoulder soreness'},

        ]

        

        return sample_injuries

    

    def fetch_injuries_from_web(self) -> List[Dict[str, Any]]:
        """Attempt to fetch injuries from multiple web sources.
        
        Order:
        1. ESPN (Best coverage)
        2. CBS Sports (Good backup)
        3. Yahoo Sports (Fallback)
        
        Returns:
            List of parsed injury records
        """
        print("\n[Injuries] Fetching injury reports...")
        today = datetime.now().strftime('%Y-%m-%d')
        
        raw_injuries = []
        sources = [
            ('ESPN', WebScraper.scrape_espn_injuries),
            ('CBS', WebScraper.scrape_cbs_injuries),
            ('Yahoo', WebScraper.scrape_yahoo_injuries)
        ]
        
        for name, scraper_func in sources:
            try:
                print(f"   Fetching from {name}...")
                injuries = scraper_func()
                if injuries and len(injuries) > 20: # Sanity check: Should find at least 20 injuries league-wide
                    print(f"   [OK] Fetched {len(injuries)} records from {name}")
                    fetched_at = datetime.now().isoformat()
                    for injury in injuries:
                        injury['source_name'] = name
                        injury['fetched_at'] = fetched_at
                    raw_injuries = injuries
                    break # Stop if successful
                else:
                    print(f"   [WARN] {name} returned few/no records ({len(injuries)}). Trying next...")
            except Exception as e:
                print(f"   [WARN] {name} fetch failed: {e}")
        
        # Fallback to sample if all fail (prevent pipeline crash, but warn heavily)
        if not raw_injuries:
             print("   [CRITICAL] All scrapers failed! Using empty list (Predictions may be inaccurate).")
             # We DO NOT use sample data anymore as it misleads the user
             # Better to have no injuries than fake ones
             
        enriched = self.parse_injury_data(raw_injuries)
        for injury in enriched:
            if raw_injuries:
                injury['source_name'] = raw_injuries[0].get('source_name')
                injury['fetched_at'] = raw_injuries[0].get('fetched_at')

        # Store in database
        if enriched and raw_injuries:
            source_name = raw_injuries[0].get('source_name')
            if source_name:
                self.db.clear_injury_snapshots(today, source_name=source_name)
        count = 0
        for injury in enriched:
            if injury['player_id']: # Only store matched players
                self.db.insert_injury_snapshot(injury)
                count += 1
        
        print(f"   Stored {count} injury records (matched to DB players)")
        return enriched





# Convenience function

def get_injury_ingestion() -> InjuryIngestion:

    """Get injury ingestion instance."""

    return InjuryIngestion()

