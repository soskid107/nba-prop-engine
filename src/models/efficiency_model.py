"""
Efficiency Model

Predicts scoring efficiency based on:
1. Shot profile (rim/midrange/3PT distribution)
2. Opponent defense vs shot zones
3. Free throw contribution (modeled separately)

Key insight: Stop thinking FG%. Think shot profile × opponent weakness.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..utils.database import DatabaseManager
from ..utils.config import get_config
from .dvp_model import DvPCalculator


# League averages for normalization (2024-25 season estimates)
LEAGUE_AVERAGES = {
    'fg_pct': 0.470,
    'fg3_pct': 0.365,
    'rim_fg_pct': 0.650,      # FG% at the rim
    'mid_fg_pct': 0.420,       # Midrange FG%
    'fta_per_min': 0.12,       # FTA per minute
    'ft_pct': 0.785,           # League FT%
    'opp_rim_protection': 0.62, # Opponent FG% allowed at rim
    'opp_three_pct': 0.365,    # Opponent 3P% allowed
    'foul_rate': 0.21,         # Fouls per possession
}


class EfficiencyModel:
    """
    Predicts scoring efficiency from shot profile × opponent matchup.
    
    Components:
    1. Shot profile (where the player shoots from)
    2. Opponent zone defense (how opponent defends each zone)
    3. Matchup efficiency edge
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize efficiency model."""
        self.db = db or DatabaseManager()
        self.config = get_config()
        
        # Cache for player shot profiles
        self._profile_cache: Dict[int, Dict] = {}
        
        # Initialize DvP Calculator
        self.dvp_calc = DvPCalculator(self.db)
    
    def get_player_shot_profile(self, player_id: int, window: int = 10) -> Dict[str, float]:
        """
        Calculate player's shot distribution and efficiency by zone.
        
        Returns:
            Dict with rim_rate, mid_rate, three_rate, and zone-specific efficiency
        """
        if player_id in self._profile_cache:
            return self._profile_cache[player_id]
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get raw game logs for weighted aggregation
            cursor.execute("""
                SELECT 
                    fga, fgm, fg3a, fg3m, fta, ftm, minutes, points, game_date
                FROM player_logs
                WHERE player_id = ?
                AND minutes > 0
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            rows = cursor.fetchall()
            
        if not rows:
            # Return league average profile
            return {
                'three_rate': 0.40,
                'mid_rate': 0.20,
                'rim_rate': 0.40,
                'three_pct': LEAGUE_AVERAGES['fg3_pct'],
                'mid_pct': LEAGUE_AVERAGES['mid_fg_pct'],
                'rim_pct': LEAGUE_AVERAGES['rim_fg_pct'],
                'overall_fg_pct': LEAGUE_AVERAGES['fg_pct'],
                'fta_per_minute': LEAGUE_AVERAGES['fta_per_min'],
                'ft_pct': LEAGUE_AVERAGES['ft_pct'],
            }
            
        # Weighted Aggregation Logic
        total_weight = 0.0
        w_fga, w_fgm = 0.0, 0.0
        w_3pa, w_3pm = 0.0, 0.0
        w_fta, w_ftm = 0.0, 0.0
        w_minutes = 0.0
        
        latest_date = datetime.strptime(rows[0]['game_date'], '%Y-%m-%d')
        decay_factor = 0.95  # Strict decay
        
        for row in rows:
            game_date = datetime.strptime(row['game_date'], '%Y-%m-%d')
            days_ago = (latest_date - game_date).days
            weight = decay_factor ** days_ago
            
            # Accumulate weighted sums
            w_fga += (row['fga'] or 0) * weight
            w_fgm += (row['fgm'] or 0) * weight
            w_3pa += (row['fg3a'] or 0) * weight
            w_3pm += (row['fg3m'] or 0) * weight
            w_fta += (row['fta'] or 0) * weight
            w_ftm += (row['ftm'] or 0) * weight
            w_minutes += (row['minutes'] or 0) * weight
            
            total_weight += weight

        # Normalize back to "per game" equivalents (though ratios don't need it, it helps debugging)
        # We actually just need the ratios from the weighted sums
        
        # Avoid division by zero
        safe_fga = w_fga if w_fga > 0.1 else 1.0
        safe_3pa = w_3pa if w_3pa > 0.1 else 1.0
        safe_fta = w_fta if w_fta > 0.1 else 1.0
        safe_minutes = w_minutes if w_minutes > 0.1 else 1.0
        
        # Estimate shot distribution
        two_pt_attempts = max(0, w_fga - w_3pa)
        two_pt_makes = max(0, w_fgm - w_3pm)
        
        two_pct = two_pt_makes / two_pt_attempts if two_pt_attempts > 0.1 else 0.50
        rim_fraction = min(0.75, max(0.30, (two_pct - 0.35) / 0.30))
        
        rim_attempts = two_pt_attempts * rim_fraction
        mid_attempts = two_pt_attempts * (1 - rim_fraction)
        
        profile = {
            # Shot distribution rates
            'three_rate': w_3pa / safe_fga,
            'mid_rate': mid_attempts / safe_fga,
            'rim_rate': rim_attempts / safe_fga,
            
            # Zone efficiencies
            'three_pct': w_3pm / safe_3pa if w_3pa > 0.1 else LEAGUE_AVERAGES['fg3_pct'],
            'rim_pct': min(0.75, two_pct * 1.2) if two_pct > 0 else 0.60,
            'mid_pct': max(0.35, two_pct * 0.85) if two_pct > 0 else 0.42,
            
            # Overall
            'overall_fg_pct': w_fgm / safe_fga,
            
            # Free throw profile
            'fta_per_minute': w_fta / safe_minutes,
            'ft_pct': w_ftm / safe_fta if safe_fta > 0.1 else 0.78,
        }
        
        self._profile_cache[player_id] = profile
        self._profile_cache[player_id] = profile
        self._profile_cache[player_id] = profile
        return profile

    def get_player_ppm_profile(self, player_id: int, window: int = 20) -> float:
        """
        Calculate player's actual Points Per Minute (PPM) from recent games.
        Weighted decay to favor recent performance.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT points, minutes, game_date
                FROM player_logs
                WHERE player_id = ?
                AND minutes > 5
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            rows = cursor.fetchall()
            
        if not rows:
            return 0.55 # League average fallback
            
        w_points = 0.0
        w_minutes = 0.0
        total_weight = 0.0
        
        latest_date = datetime.strptime(rows[0]['game_date'], '%Y-%m-%d')
        decay_factor = 0.96
        
        for row in rows:
            game_date = datetime.strptime(row['game_date'], '%Y-%m-%d')
            days_ago = (latest_date - game_date).days
            weight = decay_factor ** days_ago
            
            w_points += (row['points'] or 0) * weight
            w_minutes += (row['minutes'] or 0) * weight
            total_weight += weight
            
        if w_minutes < 5.0:
            return 0.55
            
        return w_points / w_minutes

    def get_player_fg_ppm_profile(self, player_id: int, window: int = 20) -> float:
        """
        Calculate player's Field Goal PPM (Points per minute EXCLUDING FTs).
        Crucial to avoid double-counting FTs in composition.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT points, ftm, minutes, game_date
                FROM player_logs
                WHERE player_id = ?
                AND minutes > 5
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            rows = cursor.fetchall()
            
        if not rows:
            return 0.45 # Conservative fallback (approx 0.55 total - 0.10 FT)
            
        w_fg_points = 0.0
        w_minutes = 0.0
        
        latest_date = datetime.strptime(rows[0]['game_date'], '%Y-%m-%d')
        # [FIX] Increased decay for faster response to breakouts/slumps
        # 0.90 decay = game from 7 days ago has ~48% weight (vs 75% with 0.96)
        decay_factor = 0.90
        
        for row in rows:
            game_date = datetime.strptime(row['game_date'], '%Y-%m-%d')
            days_ago = (latest_date - game_date).days
            weight = decay_factor ** days_ago
            
            fg_points = (row['points'] or 0) - (row['ftm'] or 0)
            fg_points = max(0, fg_points) # Safety
            
            w_fg_points += fg_points * weight
            w_minutes += (row['minutes'] or 0) * weight
            
        if w_minutes < 5.0:
            return 0.45
            
        return w_fg_points / w_minutes

    def _get_player_position(self, player_id: int) -> str:
        """Get player's primary position."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT position FROM players WHERE player_id = ?", (player_id,))
            row = cursor.fetchone()
            if row and row['position']:
                return row['position']
        return "G" # Default to Guard if unknown
    
    def get_opponent_defense_profile(self, opponent_abbr: str) -> Dict[str, float]:
        """
        Get opponent's defensive profile by zone.
        
        Returns:
            Dict with rim protection, perimeter defense, etc.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get opponent's recent defensive stats
            cursor.execute("""
                SELECT 
                    opp_fg_pct, opp_fg3_pct, def_rating, pace
                FROM team_advanced_stats
                WHERE team_abbreviation = ?
                AND window_type = 'L10'
                ORDER BY stat_date DESC
                LIMIT 1
            """, (opponent_abbr,))
            
            row = cursor.fetchone()
        
        if not row:
            # Return league average defense
            return {
                'rim_protection': LEAGUE_AVERAGES['opp_rim_protection'],
                'perimeter_defense': LEAGUE_AVERAGES['opp_three_pct'],
                'overall_defense': 0.465,
                'def_rating': 110.0,
                'pace': 100.0,
                'foul_rate': LEAGUE_AVERAGES['foul_rate'],
            }
        
        opp_fg_pct = row['opp_fg_pct'] or 0.465
        opp_fg3_pct = row['opp_fg3_pct'] or 0.365
        
        # Estimate rim protection from overall FG% allowed
        # Better defense (lower FG%) → better rim protection
        rim_protection = opp_fg_pct * 1.35  # Rim % is ~35% higher than overall
        
        return {
            'rim_protection': rim_protection,
            'perimeter_defense': opp_fg3_pct,
            'overall_defense': opp_fg_pct,
            'def_rating': row['def_rating'] or 110.0,
            'pace': row['pace'] or 100.0,
            # Estimate foul rate from def rating (worse defense → more fouls)
            'foul_rate': LEAGUE_AVERAGES['foul_rate'] * (row['def_rating'] / 110.0),
        }
    
    def calculate_matchup_efficiency(self, 
                                     player_profile: Dict,
                                     opponent_defense: Dict) -> Dict[str, float]:
        """
        Calculate efficiency edge from player style vs opponent weakness.
        
        Returns:
            Dict with expected efficiency metrics
        """
        # Calculate expected FG% by zone given opponent
        expected_rim_pct = player_profile['rim_pct'] * (
            opponent_defense['rim_protection'] / LEAGUE_AVERAGES['opp_rim_protection']
        )
        expected_three_pct = player_profile['three_pct'] * (
            opponent_defense['perimeter_defense'] / LEAGUE_AVERAGES['opp_three_pct']
        )
        expected_mid_pct = player_profile['mid_pct'] * (
            opponent_defense['overall_defense'] / LEAGUE_AVERAGES['fg_pct']
        )
        
        # Weighted expected efficiency
        expected_fg_pct = (
            player_profile['rim_rate'] * expected_rim_pct +
            player_profile['mid_rate'] * expected_mid_pct +
            player_profile['three_rate'] * expected_three_pct
        )
        
        # Calculate points per shot attempt (TS proxy)
        expected_pts_per_fga = (
            player_profile['rim_rate'] * expected_rim_pct * 2 +
            player_profile['mid_rate'] * expected_mid_pct * 2 +
            player_profile['three_rate'] * expected_three_pct * 3
        )
        
        # League average for comparison
        league_avg_pts_per_fga = (
            0.40 * LEAGUE_AVERAGES['rim_fg_pct'] * 2 +
            0.20 * LEAGUE_AVERAGES['mid_fg_pct'] * 2 +
            0.40 * LEAGUE_AVERAGES['fg3_pct'] * 3
        )
        
        # Efficiency edge (positive = advantage, negative = disadvantage)
        efficiency_edge = expected_pts_per_fga - league_avg_pts_per_fga
        
        # Zone-specific edges
        three_pt_edge = (expected_three_pct - LEAGUE_AVERAGES['fg3_pct'])
        rim_edge = (expected_rim_pct - LEAGUE_AVERAGES['rim_fg_pct'])
        
        return {
            'expected_fg_pct': expected_fg_pct,
            'expected_pts_per_fga': expected_pts_per_fga,
            'efficiency_edge': efficiency_edge,
            'three_pt_edge': three_pt_edge,
            'rim_edge': rim_edge,
            'expected_rim_pct': expected_rim_pct,
            'expected_three_pct': expected_three_pct,
        }
    
    def _get_team_playmakers(self, team_abbr: str) -> List[int]:
        """Identify primary playmakers (Avg AST > 5.0)."""
        if hasattr(self, '_playmaker_cache') and team_abbr in self._playmaker_cache:
            return self._playmaker_cache[team_abbr]
            
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Use full season stats to identify playmakers (handle long injuries)
            cursor.execute("""
                SELECT player_id, AVG(assists) as avg_ast
                FROM player_logs
                WHERE team_abbreviation = ?
                AND season = ?
                GROUP BY player_id
                HAVING avg_ast > 5.0
                ORDER BY avg_ast DESC
            """, (team_abbr, self.config.current_season))
            playmakers = [row['player_id'] for row in cursor.fetchall()]
            
        if not hasattr(self, '_playmaker_cache'):
            self._playmaker_cache = {}
        self._playmaker_cache[team_abbr] = playmakers
        return playmakers

    def predict_efficiency_multiplier(self,
                                      player_id: int,
                                      opponent_abbr: str,
                                      team_abbr: str = None,
                                      injuries: Dict[int, float] = None) -> Tuple[float, float]:
        """
        Calculate efficiency multiplier for PPM adjustment.
        
        Args:
            player_id: Player ID
            opponent_abbr: Opponent Team
            team_abbr: Player Team (Required for playmaker check)
            injuries: Injury map
            
        Returns:
            Tuple of (multiplier, variance)
        """
        profile = self.get_player_shot_profile(player_id)
        defense = self.get_opponent_defense_profile(opponent_abbr)
        matchup = self.calculate_matchup_efficiency(profile, defense)
        
        # Convert efficiency edge to multiplier
        # Edge of ±0.10 pts/shot → ±5% multiplier on PPM
        multiplier = 1.0 + (matchup['efficiency_edge'] * 0.5)
        
        # Apply DvP Adjustment
        position = self._get_player_position(player_id)
        dvp_multiplier = self.dvp_calc.get_multiplier(opponent_abbr, position)
        
        # Combine multipliers (Base * DvP)
        multiplier = multiplier * dvp_multiplier
        
        # [NEW] Playmaker Dependency Adjustment
        if team_abbr and injuries:
            playmakers = self._get_team_playmakers(team_abbr)
            # Check if ANY major playmaker is out
            playmaker_out = any(
                pid != player_id and injuries.get(pid, 1.0) < 0.5 
                for pid in playmakers
            )
            
            if playmaker_out:
                # If I am NOT a playmaker myself, I might suffer
                if player_id not in playmakers:
                    # Centers and heavy rim finishers rely on lobs/dimes
                    is_dependent = profile['rim_rate'] > 0.45 or profile['three_rate'] > 0.60
                    
                    if is_dependent:
                        # Penalty for losing floor general
                        multiplier *= 0.96 # -4% efficiency
        
        multiplier = np.clip(multiplier, 0.85, 1.15)  # Cap at ±15%
        
        # Variance based on shot profile
        # 3PT shooters have higher variance
        base_variance = 0.02
        if profile['three_rate'] > 0.50:
            variance = base_variance * 1.3  # High variance for shooters
        elif profile['rim_rate'] > 0.50:
            variance = base_variance * 0.8  # Lower variance for rim finishers
        else:
            variance = base_variance
        
        return multiplier, variance


class FreeThrowModel:
    """
    Models free throw contribution separately.
    
    FTs are:
    - Most stable source of points
    - Matchup-dependent (how opponent fouls)
    - High value per possession
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize free throw model."""
        self.db = db or DatabaseManager()
    
    def get_player_ft_profile(self, player_id: int, window: int = 10) -> Dict[str, float]:
        """Get player's free throw tendency and efficiency."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    SUM(fta) as total_fta,
                    SUM(ftm) as total_ftm,
                    SUM(minutes) as total_minutes,
                    SUM(points) as total_points
                FROM player_logs
                WHERE player_id = ?
                AND minutes > 0
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            row = cursor.fetchone()
        
        if not row or not row['total_minutes'] or row['total_minutes'] == 0:
            return {
                'fta_per_minute': LEAGUE_AVERAGES['fta_per_min'],
                'ft_pct': LEAGUE_AVERAGES['ft_pct'],
                'ft_pts_per_min': LEAGUE_AVERAGES['fta_per_min'] * LEAGUE_AVERAGES['ft_pct'],
            }
        
        total_fta = row['total_fta'] or 0
        total_ftm = row['total_ftm'] or 0
        total_minutes = row['total_minutes']
        
        fta_per_min = total_fta / total_minutes
        ft_pct = total_ftm / total_fta if total_fta > 0 else 0.78
        
        return {
            'fta_per_minute': fta_per_min,
            'ft_pct': ft_pct,
            'ft_pts_per_min': fta_per_min * ft_pct,
        }
    
    def predict_ft_points(self,
                          player_id: int,
                          predicted_minutes: float,
                          opponent_abbr: str = None) -> Tuple[float, float]:
        """
        Predict free throw contribution to points.
        
        Returns:
            Tuple of (expected_ft_points, variance)
        """
        profile = self.get_player_ft_profile(player_id)
        
        # Expected FTA from minutes
        expected_fta = predicted_minutes * profile['fta_per_minute']
        
        # Opponent foul rate adjustment
        foul_multiplier = 1.0
        if opponent_abbr:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT def_rating FROM team_advanced_stats
                    WHERE team_abbreviation = ?
                    AND window_type = 'L10'
                    ORDER BY stat_date DESC LIMIT 1
                """, (opponent_abbr,))
                row = cursor.fetchone()
                
                if row and row['def_rating']:
                    # Worse defensive teams foul more
                    foul_multiplier = row['def_rating'] / 110.0
        
        adjusted_fta = expected_fta * foul_multiplier
        expected_ft_pts = adjusted_fta * profile['ft_pct']
        
        # FT variance is very low (most stable scoring source)
        variance = 0.3 * profile['fta_per_minute']
        
        return expected_ft_pts, variance


# Convenience functions
def get_efficiency_model() -> EfficiencyModel:
    """Get efficiency model instance."""
    return EfficiencyModel()


def get_ft_model() -> FreeThrowModel:
    """Get free throw model instance."""
    return FreeThrowModel()
