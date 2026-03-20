"""
Defense vs Position (DvP) Model
===============================
Calculates defensive multipliers based on opponent position.
Example: "IND allows +10% points to Centers"
"""

from typing import Dict, Any, Optional
import pandas as pd
from ..utils.database import DatabaseManager

class DvPCalculator:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._cache = {}

    def get_dvp_multipliers(self, window: str = 'Season') -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate DvP multipliers for all teams and stats.
        
        Returns:
            Dict: { 'STAT': { 'TEAM': { 'POS': 1.05 } } }
        """
        if self._cache:
            return self._cache

        multipliers = {}
        
        # Stats to analyze
        stats_map = {
            'points': 'points',
            'assists': 'assists',
            'rebounds': 'rebounds',
            'threes': 'fg3m',
            'blocks': 'blocks',
            'steals': 'steals',
            'field_goals': 'fgm'
        }

        with self.db.get_connection() as conn:
            for stat_name, db_col in stats_map.items():
                # 1. League Averages by Position
                query_league = f"""
                    SELECT 
                        p.position,
                        AVG(pl.{db_col}) as avg_val
                    FROM player_logs pl
                    JOIN players p ON pl.player_id = p.player_id
                    WHERE pl.game_date >= date('now', '-30 days')
                    AND pl.minutes > 15
                    AND p.position IS NOT NULL
                    GROUP BY p.position
                """
                league_avgs = pd.read_sql_query(query_league, conn).set_index('position')['avg_val'].to_dict()
                
                # 2. Team Allowed by Position
                query_teams = f"""
                    SELECT 
                        pl.opponent_abbreviation as team,
                        p.position,
                        AVG(pl.{db_col}) as allowed_val,
                        COUNT(*) as games
                    FROM player_logs pl
                    JOIN players p ON pl.player_id = p.player_id
                    WHERE pl.game_date >= date('now', '-30 days')
                    AND pl.minutes > 15
                    AND p.position IS NOT NULL
                    GROUP BY pl.opponent_abbreviation, p.position
                """
                team_stats = pd.read_sql_query(query_teams, conn)

                # 3. Compute Multipliers for this stat
                multipliers[stat_name] = {}
                
                for _, row in team_stats.iterrows():
                    team = row['team']
                    pos = row['position']
                    allowed = row['allowed_val']
                    
                    league_avg = league_avgs.get(pos, 1.0)
                    if league_avg == 0: continue
                        
                    mult = allowed / league_avg
                    
                    # Dampen outliers (0.75 - 1.25 range)
                    mult = max(0.75, min(1.25, mult))
                    
                    if team not in multipliers[stat_name]:
                        multipliers[stat_name][team] = {}
                    multipliers[stat_name][team][pos] = mult

        self._cache = multipliers
        return multipliers

    def get_multiplier(self, team_abbr: str, position: str, stat_type: str = 'points') -> float:
        """Get multiplier for specific matchup/stat."""
        if not self._cache:
            self.get_dvp_multipliers()
            
        # Get stat specific cache
        # Map 'three_pointers_made' to 'threes' context
        if stat_type == 'three_pointers_made': stat_type = 'threes'
        
        stat_cache = self._cache.get(stat_type, self._cache.get('points', {}))
        team_data = stat_cache.get(team_abbr, {})
        
        # position mapping strategy (Specific -> Broad)
        if position in team_data:
            return team_data[position]
            
        MAPPING = {
            'PG': ['G', 'G-F'],
            'SG': ['G', 'G-F', 'F-G'],
            'SF': ['F', 'F-G', 'G-F', 'F-C'],
            'PF': ['F', 'F-C', 'C-F'],
            'C':  ['C', 'C-F', 'F-C']
        }
        
        if position in MAPPING:
            for candidate in MAPPING[position]:
                if candidate in team_data:
                    return team_data[candidate]
        
        if position and '-' in position:
            parts = position.split('-')
            for p in parts:
                if p in team_data:
                    return team_data[p]
                    
        return 1.0
