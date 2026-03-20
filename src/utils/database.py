"""

Database Manager



SQLite database schema and connection management for the NBA Props Engine.

Handles all CRUD operations for players, games, odds, logs, and injuries.

"""



import sqlite3
import shutil

from contextlib import contextmanager

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Generator, List, Optional, Tuple



from .config import get_config





class DatabaseManager:

    """Manages SQLite database connections and operations."""

    

    def __init__(self, db_path: Optional[Path] = None):

        """Initialize database manager.

        

        Args:

            db_path: Optional path to database. Uses config if not provided.

        """

        self.db_path = Path(db_path or get_config().database_path)

        self._ensure_directory()

        self._bootstrap_from_starter()

        self._initialize_schema()

    

    def _ensure_directory(self) -> None:

        """Ensure database directory exists."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _bootstrap_from_starter(self) -> None:

        """Copy the shipped starter DB into place on a fresh clone."""

        if self.db_path.exists():

            return

        starter_path = self.db_path.with_name(f"{self.db_path.stem}_starter{self.db_path.suffix}")

        if starter_path.exists():

            shutil.copy2(starter_path, self.db_path)

    

    @contextmanager

    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:

        """Context manager for database connections.

        

        Yields:

            SQLite connection with row factory set to sqlite3.Row

        """

        conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row

        try:

            yield conn

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            conn.close()

    

    def _initialize_schema(self) -> None:

        """Create all tables if they don't exist."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            

            # =====================

            # Players Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS players (

                    player_id INTEGER PRIMARY KEY,

                    full_name TEXT NOT NULL,

                    first_name TEXT,

                    last_name TEXT,

                    team_id INTEGER,

                    team_abbreviation TEXT,

                    position TEXT,

                    is_active INTEGER DEFAULT 1,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP

                )

            """)

            

            # =====================

            # Teams Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS teams (

                    team_id INTEGER PRIMARY KEY,

                    abbreviation TEXT NOT NULL UNIQUE,

                    full_name TEXT NOT NULL,

                    city TEXT,

                    nickname TEXT,

                    conference TEXT,

                    division TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP

                )

            """)

            

            # =====================

            # Games Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS games (

                    game_id TEXT PRIMARY KEY,

                    game_date TEXT NOT NULL,

                    season TEXT NOT NULL,

                    season_type TEXT,

                    home_team_id INTEGER,

                    away_team_id INTEGER,

                    home_team_abbr TEXT,

                    away_team_abbr TEXT,

                    home_score INTEGER,

                    away_score INTEGER,

                    status TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),

                    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)

                )

            """)

            

            # =====================

            # Player Game Logs Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS player_logs (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    player_id INTEGER NOT NULL,

                    game_id TEXT NOT NULL,

                    game_date TEXT NOT NULL,

                    season TEXT NOT NULL,

                    team_id INTEGER,

                    team_abbreviation TEXT,

                    opponent_team_id INTEGER,

                    opponent_abbreviation TEXT,

                    is_home INTEGER,

                    is_starter INTEGER,

                    minutes REAL,

                    points INTEGER,

                    rebounds INTEGER,

                    assists INTEGER,

                    steals INTEGER,

                    blocks INTEGER,

                    turnovers INTEGER,

                    fgm INTEGER,

                    fga INTEGER,

                    fg3m INTEGER,

                    fg3a INTEGER,

                    ftm INTEGER,

                    fta INTEGER,

                    plus_minus INTEGER,

                    ppm REAL,  -- Points per minute (calculated)

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(player_id, game_id),

                    FOREIGN KEY (player_id) REFERENCES players(player_id),

                    FOREIGN KEY (game_id) REFERENCES games(game_id)

                )

            """)

            

            # =====================

            # =====================
            # Player Prop Odds Table
            # =====================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_prop_odds (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     game_id TEXT,
                     game_date TEXT NOT NULL,
                     player_id INTEGER,
                     player_name TEXT NOT NULL,
                     market_key TEXT NOT NULL, -- e.g. player_points
                     bookmaker TEXT NOT NULL,
                     line REAL NOT NULL,
                     odds_over INTEGER,
                     odds_under INTEGER,
                     snapshot_time TEXT NOT NULL,
                     created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                     UNIQUE(game_date, player_id, market_key, bookmaker, snapshot_time)
                )
            """)

            # =====================
            # Odds Snapshots Table (Market Context)

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS odds_snapshots (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    game_id TEXT,

                    game_date TEXT NOT NULL,

                    home_team TEXT NOT NULL,

                    away_team TEXT NOT NULL,

                    spread_home REAL,

                    spread_away REAL,

                    total REAL,

                    over_odds REAL,

                    under_odds REAL,

                    moneyline_home INTEGER,

                    moneyline_away INTEGER,

                    bookmaker TEXT,

                    snapshot_time TEXT NOT NULL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(game_date, home_team, away_team, bookmaker, snapshot_time)

                )

            """)

            

            # =====================

            # Injury Snapshots Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS injury_snapshots (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    player_id INTEGER,

                    player_name TEXT NOT NULL,

                    team_abbreviation TEXT,

                    status TEXT NOT NULL,

                    reason TEXT,

                    source_name TEXT,

                    fetched_at TEXT,

                    report_date TEXT NOT NULL,

                    p_play REAL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (player_id) REFERENCES players(player_id)

                )

            """)

            for column_def in [
                "ALTER TABLE injury_snapshots ADD COLUMN source_name TEXT",
                "ALTER TABLE injury_snapshots ADD COLUMN fetched_at TEXT",
            ]:
                try:
                    cursor.execute(column_def)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

            

            # =====================

            # HTTP Cache Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS http_cache (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    cache_key TEXT NOT NULL UNIQUE,

                    url TEXT NOT NULL,

                    params TEXT,

                    response_data TEXT NOT NULL,

                    content_type TEXT,

                    cached_at TEXT NOT NULL,

                    expires_at TEXT NOT NULL

                )

            """)

            

            # =====================

            # Team Advanced Stats Table

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS team_advanced_stats (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    team_abbreviation TEXT,

                    window_type TEXT, -- e.g. 'Season', 'L5', 'L10'

                    stat_date TEXT,

                    off_rating REAL,

                    def_rating REAL,

                    net_rating REAL,

                    pace REAL,

                    pie REAL,

                    fg_pct REAL,

                    opp_fg_pct REAL,

                    fg3_pct REAL,

                    opp_fg3_pct REAL,

                    oreb_pct REAL,

                    opp_oreb_pct REAL,

                    dreb_pct REAL,

                    reb_pct REAL,

                    alt_pace REAL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(team_abbreviation, window_type, stat_date)

                )

            """)



            # =====================

            # API Usage Tracking

            # =====================

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS api_usage (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    api_name TEXT NOT NULL,

                    endpoint TEXT,

                    call_date TEXT NOT NULL,

                    call_time TEXT NOT NULL,

                    response_status INTEGER,

                    cached INTEGER DEFAULT 0

                )

            """)

            

            # =====================

            # Indexes for Performance

            # =====================

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_logs_player_id ON player_logs(player_id)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_logs_game_date ON player_logs(game_date)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_logs_season ON player_logs(season)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_game_date ON odds_snapshots(game_date)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_injuries_date ON injury_snapshots(report_date)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON http_cache(cache_key)")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON http_cache(expires_at)")

            

            conn.commit()

    

    # =====================

    # Player Operations

    # =====================

    def upsert_player(self, player_data: Dict[str, Any]) -> None:

        """Insert or update a player record."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT INTO players (player_id, full_name, first_name, last_name, 

                                    team_id, team_abbreviation, position, is_active, updated_at)

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(player_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    team_id = excluded.team_id,
                    team_abbreviation = excluded.team_abbreviation,
                    position = excluded.position,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at

            """, (

                player_data['player_id'],

                player_data.get('full_name'),

                player_data.get('first_name'),

                player_data.get('last_name'),

                player_data.get('team_id'),

                player_data.get('team_abbreviation'),

                player_data.get('position'),

                player_data.get('is_active', 1),

                datetime.now().isoformat()

            ))

    

    def get_player_by_id(self, player_id: int) -> Optional[Dict[str, Any]]:

        """Get player by ID."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))

            row = cursor.fetchone()

            return dict(row) if row else None

    

    def search_players(self, name_query: str) -> List[Dict[str, Any]]:

        """Search players by name (fuzzy)."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM players 

                WHERE full_name LIKE ? OR first_name LIKE ? OR last_name LIKE ?

                ORDER BY full_name

            """, (f"%{name_query}%", f"%{name_query}%", f"%{name_query}%"))

            return [dict(row) for row in cursor.fetchall()]

    

    # =====================

    # Game Log Operations

    # =====================

    def insert_player_log(self, log_data: Dict[str, Any]) -> None:

        """Insert a player game log (skip if exists)."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT OR IGNORE INTO player_logs (

                    player_id, game_id, game_date, season, team_id, team_abbreviation,

                    opponent_team_id, opponent_abbreviation, is_home, is_starter,

                    minutes, points, rebounds, assists, steals, blocks, turnovers,

                    fgm, fga, fg3m, fg3a, ftm, fta, plus_minus, ppm

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                log_data['player_id'],

                log_data['game_id'],

                log_data['game_date'],

                log_data['season'],

                log_data.get('team_id'),

                log_data.get('team_abbreviation'),

                log_data.get('opponent_team_id'),

                log_data.get('opponent_abbreviation'),

                log_data.get('is_home', 0),

                log_data.get('is_starter', 0),

                log_data.get('minutes', 0),

                log_data.get('points', 0),

                log_data.get('rebounds', 0),

                log_data.get('assists', 0),

                log_data.get('steals', 0),

                log_data.get('blocks', 0),

                log_data.get('turnovers', 0),

                log_data.get('fgm', 0),

                log_data.get('fga', 0),

                log_data.get('fg3m', 0),

                log_data.get('fg3a', 0),

                log_data.get('ftm', 0),

                log_data.get('fta', 0),

                log_data.get('plus_minus', 0),

                log_data.get('ppm', 0)

            ))

    

    def get_player_logs(self, player_id: int, limit: int = 100) -> List[Dict[str, Any]]:

        """Get recent game logs for a player."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM player_logs 

                WHERE player_id = ?

                ORDER BY game_date DESC

                LIMIT ?

            """, (player_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    

    def get_latest_game_date(self, player_id: int) -> Optional[str]:

        """Get the most recent game date for a player."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT MAX(game_date) as latest FROM player_logs WHERE player_id = ?

            """, (player_id,))

            row = cursor.fetchone()

            return row['latest'] if row else None

    

    # =====================

    # Odds Operations

    # =====================

    def insert_odds_snapshot(self, odds_data: Dict[str, Any]) -> None:

        """Insert odds snapshot."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT OR IGNORE INTO odds_snapshots (

                    game_id, game_date, home_team, away_team, spread_home, spread_away,

                    total, over_odds, under_odds, moneyline_home, moneyline_away,

                    bookmaker, snapshot_time

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                odds_data.get('game_id'),

                odds_data['game_date'],

                odds_data['home_team'],

                odds_data['away_team'],

                odds_data.get('spread_home'),

                odds_data.get('spread_away'),

                odds_data.get('total'),

                odds_data.get('over_odds'),

                odds_data.get('under_odds'),

                odds_data.get('moneyline_home'),

                odds_data.get('moneyline_away'),

                odds_data.get('bookmaker', 'consensus'),

                odds_data.get('snapshot_time', datetime.now().isoformat())

            ))

    

    def get_odds_for_date(self, game_date: str) -> List[Dict[str, Any]]:

        """Get all odds snapshots for a date."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM odds_snapshots WHERE game_date = ? ORDER BY home_team

            """, (game_date,))

            return [dict(row) for row in cursor.fetchall()]

    

    # =====================

    # Injury Operations

    # =====================

    def insert_injury_snapshot(self, injury_data: Dict[str, Any]) -> None:

        """Insert injury snapshot."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT INTO injury_snapshots (

                    player_id, player_name, team_abbreviation, status, reason,

                    source_name, fetched_at, report_date, p_play

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                injury_data.get('player_id'),

                injury_data['player_name'],

                injury_data.get('team_abbreviation'),

                injury_data['status'],

                injury_data.get('reason'),

                injury_data.get('source_name'),

                injury_data.get('fetched_at'),

                injury_data['report_date'],

                injury_data.get('p_play', 1.0)

            ))

    def clear_injury_snapshots(self, report_date: str, source_name: Optional[str] = None) -> None:
        """Clear injury snapshots for a date, optionally for one source only."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if source_name:
                cursor.execute("""
                    DELETE FROM injury_snapshots
                    WHERE report_date = ? AND source_name = ?
                """, (report_date, source_name))
            else:
                cursor.execute("""
                    DELETE FROM injury_snapshots
                    WHERE report_date = ?
                """, (report_date,))

    

    def get_injuries_for_date(self, report_date: str) -> List[Dict[str, Any]]:

        """Get injury reports for a specific date."""

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                WITH ranked AS (
                    SELECT
                        *,
                        CASE UPPER(COALESCE(source_name, ''))
                            WHEN 'ESPN' THEN 4
                            WHEN 'CBS' THEN 3
                            WHEN 'YAHOO' THEN 2
                            WHEN 'ROTOWIRE' THEN 1
                            ELSE 0
                        END AS source_priority,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(CAST(player_id AS TEXT), player_name)
                            ORDER BY
                                CASE UPPER(COALESCE(source_name, ''))
                                    WHEN 'ESPN' THEN 4
                                    WHEN 'CBS' THEN 3
                                    WHEN 'YAHOO' THEN 2
                                    WHEN 'ROTOWIRE' THEN 1
                                    ELSE 0
                                END DESC,
                                COALESCE(fetched_at, created_at) DESC,
                                id DESC
                        ) AS rn
                    FROM injury_snapshots
                    WHERE report_date = ?
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                ORDER BY COALESCE(team_abbreviation, ''), player_name
            """, (report_date,))
            return [dict(row) for row in cursor.fetchall()]

    # Cache Operations

    # =====================

    def get_cached_response(self, cache_key: str) -> Optional[str]:

        """Get cached HTTP response if not expired."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT response_data FROM http_cache 

                WHERE cache_key = ? AND expires_at > ?

            """, (cache_key, datetime.now().isoformat()))

            row = cursor.fetchone()

            return row['response_data'] if row else None

    

    def set_cached_response(self, cache_key: str, url: str, params: str, 

                           response_data: str, expires_at: str) -> None:

        """Store HTTP response in cache."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT OR REPLACE INTO http_cache (cache_key, url, params, response_data, cached_at, expires_at)

                VALUES (?, ?, ?, ?, ?, ?)

            """, (cache_key, url, params, response_data, datetime.now().isoformat(), expires_at))

    

    def clear_expired_cache(self) -> int:

        """Clear expired cache entries. Returns count deleted."""

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("DELETE FROM http_cache WHERE expires_at < ?", 

                          (datetime.now().isoformat(),))

            return cursor.rowcount

    

    # =====================

    # API Usage Tracking

    # =====================

    def log_api_call(self, api_name: str, endpoint: str, status: int, cached: bool = False) -> None:

        """Log an API call for usage tracking."""

        now = datetime.now()

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT INTO api_usage (api_name, endpoint, call_date, call_time, response_status, cached)

                VALUES (?, ?, ?, ?, ?, ?)

            """, (api_name, endpoint, now.strftime('%Y-%m-%d'), now.isoformat(), status, int(cached)))

    

    def get_api_usage_today(self, api_name: str) -> int:

        """Get count of API calls today (excluding cached)."""

        today = datetime.now().strftime('%Y-%m-%d')

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT COUNT(*) as count FROM api_usage 

                WHERE api_name = ? AND call_date = ? AND cached = 0

            """, (api_name, today))

            row = cursor.fetchone()

            return row['count'] if row else 0





    def insert_player_prop_odds(self, data: Dict[str, Any]) -> None:
        """Insert a player prop odds record.

        Args:
            data: Dictionary containing prop odds data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO player_prop_odds (
                    game_id, game_date, player_id, player_name, 
                    market_key, bookmaker, line, odds_over, odds_under,
                    snapshot_time
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                data.get('game_id'),
                data.get('game_date'),
                data.get('player_id'),
                data.get('player_name'),
                data.get('market_key'),
                data.get('bookmaker'),
                data.get('line'),
                data.get('odds_over'),
                data.get('odds_under'),
                data.get('snapshot_time')
            ))

    def clear_player_prop_odds_for_date(self, date_str: str) -> None:
        """Remove existing prop odds rows for a date before a full refresh."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM player_prop_odds
                WHERE game_date = ?
            """, (date_str,))

    def get_player_props_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get all player props for a specific date.

        Args:
            date_str: Date string YYYY-MM-DD

        Returns:
            List of prop records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM player_prop_odds 
                WHERE game_date = ?
            """, (date_str,))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_player_prop_snapshot_summary(self, date_str: str) -> Dict[str, Any]:
        """Summarize stored player props for a date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) AS prop_count,
                    COUNT(DISTINCT game_id) AS game_count,
                    MAX(snapshot_time) AS latest_snapshot_time
                FROM player_prop_odds
                WHERE game_date = ?
            """, (date_str,))
            row = cursor.fetchone()
            return dict(row) if row else {
                'prop_count': 0,
                'game_count': 0,
                'latest_snapshot_time': None,
            }

    def reconcile_player_teams_from_logs(self) -> int:
        """Update current player team assignments using the most recent game logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                WITH latest_logs AS (
                    SELECT
                        pl.player_id,
                        pl.team_id,
                        pl.team_abbreviation,
                        pl.game_date,
                        ROW_NUMBER() OVER (
                            PARTITION BY pl.player_id
                            ORDER BY pl.game_date DESC, pl.id DESC
                        ) AS rn
                    FROM player_logs pl
                    WHERE pl.team_abbreviation IS NOT NULL
                      AND TRIM(pl.team_abbreviation) != ''
                )
                UPDATE players
                SET
                    team_id = (
                        SELECT latest_logs.team_id
                        FROM latest_logs
                        WHERE latest_logs.player_id = players.player_id
                          AND latest_logs.rn = 1
                    ),
                    team_abbreviation = (
                        SELECT latest_logs.team_abbreviation
                        FROM latest_logs
                        WHERE latest_logs.player_id = players.player_id
                          AND latest_logs.rn = 1
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE EXISTS (
                    SELECT 1
                    FROM latest_logs
                    WHERE latest_logs.player_id = players.player_id
                      AND latest_logs.rn = 1
                      AND COALESCE(players.team_abbreviation, '') != COALESCE(latest_logs.team_abbreviation, '')
                )
            """)
            return cursor.rowcount if cursor.rowcount != -1 else 0


# Convenience function

def get_db() -> DatabaseManager:

    """Get database manager instance."""

    return DatabaseManager()

