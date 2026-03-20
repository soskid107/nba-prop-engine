"""
Defensive Scheme Analyzer
==========================
Enriches the StatFragilityAgent and edge pipeline with defensive scheme data.

Uses team_advanced_stats to classify opponent defensive tendencies:
  1. Paint Protection (opp_fg_pct near rim, blocks)
  2. Perimeter Defense (opp_fg3_pct)
  3. Pace Forcing (opponent's pace vs league average)
  4. Rebounding Style (OREB% vs DREB%)
  5. Turnover Forcing (high deflections teams)

Data Source: team_advanced_stats table
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DEF_SCHEME")


class DefensiveSchemeAnalyzer:
    """
    Classifies opponent defensive schemes to enhance kill-script accuracy.
    
    Instead of just using DEF RTG (a single number), this breaks defense
    into specific scheme components that differently affect each stat.
    """
    
    # League averages (2025-26 season approximate)
    LEAGUE_AVG = {
        'def_rating': 112.0,
        'pace': 100.0,
        'opp_fg_pct': 0.465,
        'opp_fg3_pct': 0.360,
        'oreb_pct': 0.27,
        'dreb_pct': 0.73,
    }
    
    # Scheme classifications
    SCHEME_THRESHOLDS = {
        'paint_protector': {'opp_fg_pct': -0.015},    # Allow 1.5% less than avg
        'perimeter_lock': {'opp_fg3_pct': -0.015},
        'pace_pusher': {'pace': 3.0},                  # 3+ above avg
        'pace_grinder': {'pace': -3.0},                 # 3+ below avg
        'turnover_forcer': {'def_rating': -4.0},        # Well below avg DEF RTG
        'glass_cleaner': {'dreb_pct': 0.04},            # 4% above avg
    }
    
    def __init__(self, db=None):
        self.db = db
    
    def analyze_defense(self, 
                        opponent_abbr: str,
                        game_date: str = None) -> Dict[str, Any]:
        """
        Analyze opponent's defensive scheme with trend detection.
        """
        stats_windows = self._get_team_stats(opponent_abbr, game_date)
        
        if not stats_windows or 'Season' not in stats_windows:
            return self._default_analysis(opponent_abbr)
        
        # Use L10 or L5 if available for current form, else Season
        current_stats = stats_windows.get('L10') or stats_windows.get('Season')
        season_stats = stats_windows['Season']
        
        schemes = []
        stat_impacts = {}
        
        def_rating = current_stats.get('def_rating', self.LEAGUE_AVG['def_rating'])
        pace = current_stats.get('pace', self.LEAGUE_AVG['pace'])
        opp_fg_pct = current_stats.get('opp_fg_pct', self.LEAGUE_AVG['opp_fg_pct'])
        opp_fg3_pct = current_stats.get('opp_fg3_pct', self.LEAGUE_AVG['opp_fg3_pct'])
        dreb_pct = current_stats.get('dreb_pct', self.LEAGUE_AVG['dreb_pct'])
        oreb_pct = current_stats.get('oreb_pct', self.LEAGUE_AVG['oreb_pct'])
        
        # 1. Paint Protection
        fg_diff = opp_fg_pct - self.LEAGUE_AVG['opp_fg_pct']
        if fg_diff < self.SCHEME_THRESHOLDS['paint_protector']['opp_fg_pct']:
            schemes.append('paint_protector')
            stat_impacts['points'] = stat_impacts.get('points', 0) - 2.0
            stat_impacts['fga'] = stat_impacts.get('fga', 0) - 1.0
        elif fg_diff > 0.015:
            stat_impacts['points'] = stat_impacts.get('points', 0) + 1.5
        
        # 2. Perimeter Lock
        fg3_diff = opp_fg3_pct - self.LEAGUE_AVG['opp_fg3_pct']
        if fg3_diff < self.SCHEME_THRESHOLDS['perimeter_lock']['opp_fg3_pct']:
            schemes.append('perimeter_lock')
            stat_impacts['fg3m'] = stat_impacts.get('fg3m', 0) - 0.5
            stat_impacts['points'] = stat_impacts.get('points', 0) - 1.5
        elif fg3_diff > 0.015:
            stat_impacts['fg3m'] = stat_impacts.get('fg3m', 0) + 0.5
            stat_impacts['points'] = stat_impacts.get('points', 0) + 1.0
        
        # 3. Pace Classification (Trend Aware)
        # Check if pace is trending up/down significantly
        pace_season = season_stats.get('pace', self.LEAGUE_AVG['pace'])
        pace_trend = pace - pace_season
        
        pace_diff = pace - self.LEAGUE_AVG['pace']
        if pace_diff >= self.SCHEME_THRESHOLDS['pace_pusher']['pace']:
            schemes.append('pace_pusher')
            stat_impacts['points'] = stat_impacts.get('points', 0) + 2.0
            stat_impacts['rebounds'] = stat_impacts.get('rebounds', 0) + 0.5
            stat_impacts['assists'] = stat_impacts.get('assists', 0) + 0.5
        elif pace_diff <= self.SCHEME_THRESHOLDS['pace_grinder']['pace']:
            schemes.append('pace_grinder')
            stat_impacts['points'] = stat_impacts.get('points', 0) - 2.0
            stat_impacts['rebounds'] = stat_impacts.get('rebounds', 0) - 0.5
        
        # 4. Turnover Forcing
        def_diff = def_rating - self.LEAGUE_AVG['def_rating']
        if def_diff < self.SCHEME_THRESHOLDS['turnover_forcer']['def_rating']:
            schemes.append('turnover_forcer')
            stat_impacts['turnovers'] = stat_impacts.get('turnovers', 0) + 0.5
            stat_impacts['assists'] = stat_impacts.get('assists', 0) - 0.5
            stat_impacts['points'] = stat_impacts.get('points', 0) - 1.0
        elif def_diff > 4.0:
            stat_impacts['points'] = stat_impacts.get('points', 0) + 2.0
            stat_impacts['assists'] = stat_impacts.get('assists', 0) + 1.0
        
        # 5. Glass Cleaning
        dreb_diff = dreb_pct - self.LEAGUE_AVG['dreb_pct']
        if dreb_diff >= self.SCHEME_THRESHOLDS['glass_cleaner']['dreb_pct']:
            schemes.append('glass_cleaner')
            stat_impacts['rebounds'] = stat_impacts.get('rebounds', 0) - 1.0
        elif dreb_diff < -0.03:
            stat_impacts['rebounds'] = stat_impacts.get('rebounds', 0) + 1.0
        
        # Determine Quality
        if def_rating < 108: quality = 'elite'
        elif def_rating < 112: quality = 'good'
        elif def_rating < 116: quality = 'average'
        else: quality = 'poor'
        
        return {
            'opponent': opponent_abbr,
            'defensive_quality': quality,
            'def_rating': def_rating,
            'pace': pace,
            'schemes': schemes,
            'stat_impacts': stat_impacts,
            'raw_stats': {
                'opp_fg_pct': opp_fg_pct,
                'opp_fg3_pct': opp_fg3_pct,
                'pace': pace,
                'dreb_pct': dreb_pct,
                'oreb_pct': oreb_pct,
                'pace_trend': pace_trend
            },
            'kill_script_boost': self._get_kill_script_relevance(schemes),
        }

    def _get_kill_script_relevance(self, schemes: List[str]) -> Dict[str, float]:
        """Map schemes to kill script probability boosts."""
        boosts = {}
        if 'paint_protector' in schemes:
            boosts['defensive_adjustment'] = 1.3
            boosts['blowout'] = 0.9
        if 'pace_grinder' in schemes:
            boosts['blowout'] = 1.2
            boosts['tight_game'] = 1.3
        if 'pace_pusher' in schemes:
            boosts['blowout'] = 0.8
            boosts['hot_teammate'] = 1.2
        if 'turnover_forcer' in schemes:
            boosts['foul_trouble'] = 1.2
            boosts['defensive_adjustment'] = 1.4
        if 'glass_cleaner' in schemes:
            boosts['hot_teammate'] = 0.9
        return boosts
    
    def _get_team_stats(self, team_abbr: str,
                        game_date: str = None) -> Dict[str, Dict]:
        """Get ALL team defensive stats windows (Season, L10, L5)."""
        if not self.db: return {}
        
        results = {}
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # If game_date provided, filter before that date
                date_filter = ""
                params = [team_abbr]
                if game_date:
                    date_filter = "AND stat_date <= ?"
                    params.append(game_date)
                
                for window in ['Season', 'L10', 'L5']:
                    # We need a fresh cursor execute for each window
                    # and we must respect game_date to prevent data leakage in backtests
                    current_params = params + [window]
                    
                    cursor.execute(f"""
                        SELECT def_rating, pace, opp_fg_pct, opp_fg3_pct,
                               dreb_pct, opp_oreb_pct as oreb_pct
                        FROM team_advanced_stats
                        WHERE team_abbreviation = ? 
                        {date_filter}
                        AND window_type = ?
                        ORDER BY stat_date DESC
                        LIMIT 1
                    """, current_params if game_date else [team_abbr, window])
                    
                    row = cursor.fetchone()
                    if row:
                        results[window] = dict(row)
                        
            return results
        except Exception as e:
            logger.warning(f"Failed to get team stats for {team_abbr}: {e}")
            return {}
    
    def _default_analysis(self, opponent_abbr: str) -> Dict[str, Any]:
        """Return default when no data available."""
        return {
            'opponent': opponent_abbr,
            'defensive_quality': 'average',
            'def_rating': self.LEAGUE_AVG['def_rating'],
            'pace': self.LEAGUE_AVG['pace'],
            'schemes': [],
            'stat_impacts': {},
            'raw_stats': {},
            'kill_script_boost': {},
        }
