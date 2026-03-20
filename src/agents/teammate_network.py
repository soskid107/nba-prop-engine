"""
Teammate Usage Network
========================
Maps usage redistribution when teammates are absent, cold, or in foul trouble.

Key Insight: "Who gets the shots when X doesn't?"
  - When a star sits, who absorbs their usage?
  - When teammate Y is cold, does player X take more?
  - What's the historical usage split in specific lineups?

Data Source: player_logs (same team, same game) + injury_snapshots
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("USAGE_NETWORK")


class TeammateUsageNetwork:
    """
    Builds a teammate usage graph from historical game logs.
    
    For each player, tracks how their usage changes when
    key teammates are absent or underperforming.
    """
    
    def __init__(self, db=None):
        self.db = db
        self._usage_cache = {}
    
    def analyze_usage_impact(self,
                             player_id: int,
                             player_name: str,
                             team_abbr: str,
                             missing_players: List[Dict] = None,
                             game_date: str = None) -> Dict[str, Any]:
        """
        Analyze how missing/injured teammates affect this player's usage.
        
        Args:
            player_id: Target player ID
            player_name: Target player name
            team_abbr: Team abbreviation
            missing_players: List of dicts with player_id, player_name, status
            game_date: Target game date
            
        Returns:
            Dict with usage impact analysis
        """
        if not self.db or not missing_players:
            return self._no_impact_result()
        
        impacts = []
        total_usage_boost = 0
        total_minutes_boost = 0
        total_assists_boost = 0
        total_rebounds_boost = 0
        
        for missing in missing_players:
            missing_id = missing.get('player_id')
            missing_name = missing.get('player_name', 'Unknown')
            
            if not missing_id:
                continue
            
            # Get games where the missing player was OUT
            impact = self._calculate_absence_impact(
                player_id, missing_id, team_abbr, game_date
            )
            
            if impact and impact['sample_size'] >= 3:
                impacts.append({
                    'missing_player': missing_name,
                    'games_without': impact['sample_size'],
                    'usage_boost': impact['usage_boost'],
                    'points_boost': impact['points_boost'],
                    'minutes_boost': impact['minutes_boost'],
                    'assists_boost': impact['assists_boost'],
                    'rebounds_boost': impact['rebounds_boost'],
                })
                total_usage_boost += impact['usage_boost']
                total_minutes_boost += impact['minutes_boost']
                total_assists_boost += impact['assists_boost']
                total_rebounds_boost += impact['rebounds_boost']
        
        # Get team usage hierarchy
        hierarchy = self._get_team_hierarchy(team_abbr, game_date)
        
        return {
            'player_name': player_name,
            'teammate_impacts': impacts,
            'total_usage_boost': round(total_usage_boost, 3),
            'total_minutes_boost': round(total_minutes_boost, 1),
            'total_assists_boost': round(total_assists_boost, 1),
            'total_rebounds_boost': round(total_rebounds_boost, 1),
            'expected_points_boost': round(total_usage_boost * 30, 1),  # Rough: usage * possessions
            'expected_minutes_boost': round(total_minutes_boost, 1),
            'usage_pressure_score': round(min(1.0, abs(total_usage_boost) * 10.0 + abs(total_minutes_boost) / 10.0), 2),
            'team_hierarchy': hierarchy,
            'has_significant_impact': total_usage_boost > 0.02 or total_minutes_boost > 2 or total_assists_boost > 1.5,
        }
    
    def _calculate_absence_impact(self, 
                                   target_id: int,
                                   missing_id: int,
                                   team_abbr: str,
                                   game_date: str = None) -> Optional[Dict]:
        """
        Calculate how target player's stats change when missing player is absent.
        
        Compares:
          - Games where BOTH played (baseline)
          - Games where missing player was OUT (impact)
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get target player's games
                cutoff = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
                
                cursor.execute("""
                    SELECT game_id, game_date, minutes, points, assists, rebounds,
                           fga, fta, turnovers
                    FROM player_logs
                    WHERE player_id = ? AND team_abbreviation = ?
                          AND game_date >= ? AND minutes > 0
                    ORDER BY game_date DESC
                    LIMIT 40
                """, (target_id, team_abbr, cutoff))
                target_games = {row['game_id']: dict(row) for row in cursor.fetchall()}
                
                if not target_games:
                    return None
                
                # Get missing player's games
                cursor.execute("""
                    SELECT game_id, minutes
                    FROM player_logs
                    WHERE player_id = ? AND team_abbreviation = ?
                          AND game_date >= ?
                    ORDER BY game_date DESC
                    LIMIT 40
                """, (missing_id, team_abbr, cutoff))
                missing_games = {row['game_id']: row['minutes'] for row in cursor.fetchall()}
                
                # Split target's games into "both played" vs "missing was out"
                both_played = []
                missing_out = []
                
                for game_id, stats in target_games.items():
                    if game_id in missing_games and missing_games[game_id] > 5:
                        both_played.append(stats)
                    else:
                        missing_out.append(stats)
                
                if len(both_played) < 3 or len(missing_out) < 3:
                    return None
                
                # Calculate impact
                def avg_stat(games, stat):
                    vals = [g.get(stat, 0) for g in games]
                    return np.mean(vals) if vals else 0
                
                def calc_usage(games):
                    """Usage proxy: (FGA + 0.44*FTA + TOV) / Minutes."""
                    usages = []
                    for g in games:
                        mins = g.get('minutes', 0)
                        if mins > 5:
                            u = (g.get('fga', 0) + 0.44 * g.get('fta', 0) + g.get('turnovers', 0)) / mins
                            usages.append(u)
                    return np.mean(usages) if usages else 0
                
                baseline_usage = calc_usage(both_played)
                impact_usage = calc_usage(missing_out)
                
                return {
                    'sample_size': len(missing_out),
                    'baseline_games': len(both_played),
                    'usage_boost': round(impact_usage - baseline_usage, 4),
                    'points_boost': round(avg_stat(missing_out, 'points') - avg_stat(both_played, 'points'), 1),
                    'minutes_boost': round(avg_stat(missing_out, 'minutes') - avg_stat(both_played, 'minutes'), 1),
                    'assists_boost': round(avg_stat(missing_out, 'assists') - avg_stat(both_played, 'assists'), 1),
                    'rebounds_boost': round(avg_stat(missing_out, 'rebounds') - avg_stat(both_played, 'rebounds'), 1),
                }
                
        except Exception as e:
            logger.warning(f"Failed to calculate absence impact: {e}")
            return None
    
    def _get_team_hierarchy(self, team_abbr: str, 
                            game_date: str = None) -> List[Dict]:
        """
        Get team's usage hierarchy (top 5 players by usage).
        """
        if not self.db:
            return []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                cursor.execute("""
                    SELECT p.full_name, 
                           AVG(pl.points) as avg_pts,
                           AVG(pl.minutes) as avg_min,
                           AVG(pl.fga) as avg_fga,
                           COUNT(*) as games
                    FROM player_logs pl
                    JOIN players p ON pl.player_id = p.player_id
                    WHERE pl.team_abbreviation = ? 
                          AND pl.game_date >= ?
                          AND pl.minutes > 10
                    GROUP BY pl.player_id
                    HAVING games >= 3
                    ORDER BY avg_fga DESC
                    LIMIT 5
                """, (team_abbr, cutoff))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get team hierarchy: {e}")
            return []
    
    def _no_impact_result(self) -> Dict[str, Any]:
        return {
            'player_name': '',
            'teammate_impacts': [],
            'total_usage_boost': 0,
            'total_minutes_boost': 0,
            'total_assists_boost': 0,
            'total_rebounds_boost': 0,
            'expected_points_boost': 0,
            'expected_minutes_boost': 0,
            'usage_pressure_score': 0,
            'team_hierarchy': [],
            'has_significant_impact': False,
        }
