"""
Variance Model

Models player-specific scoring variance based on:
1. Player archetype (microwave scorer, catch-and-shoot, volume star, etc.)
2. Shot profile (3PT shooters have higher variance)
3. Pace interaction (as multiplier for possessions)

Key insight: Books price means well. They misprice tails.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from ..utils.database import DatabaseManager
from ..utils.config import get_config


# Player archetype definitions with variance multipliers
PLAYER_ARCHETYPES = {
    'volume_star': {
        'description': 'High usage star player (30+ min, 25%+ usage)',
        'variance_multiplier': 1.0,   # Normal variance (stable)
        'min_usage': 0.25,
        'min_minutes': 28,
    },
    'secondary_star': {
        'description': 'Second scoring option (28+ min, 20-25% usage)',
        'variance_multiplier': 1.1,   # Slightly higher variance
        'min_usage': 0.20,
        'min_minutes': 26,
    },
    'microwave_scorer': {
        'description': 'Bench scorer who gets hot/cold (15-25 min, high PPM when on)',
        'variance_multiplier': 1.5,   # High variance - boom or bust
        'max_minutes': 25,
        'min_ppm': 0.50,
    },
    'catch_and_shoot': {
        'description': '3PT specialist, dependent on looks (high 3PA rate)',
        'variance_multiplier': 1.4,   # High variance from 3PT volume
        'min_three_rate': 0.50,
    },
    'rim_runner': {
        'description': 'Primarily rim finisher (high rim rate, lower variance)',
        'variance_multiplier': 0.85,  # Lower variance - efficient finisher
        'min_rim_rate': 0.55,
    },
    'floor_general': {
        'description': 'Pass-first guard with low usage',
        'variance_multiplier': 0.9,   # Lower variance - consistent role
        'max_usage': 0.18,
        'min_ast_rate': 0.25,
    },
    'role_player': {
        'description': 'Default - limited offensive role',
        'variance_multiplier': 0.8,   # Low variance - predictable
    }
}

# League average PPM and standard deviation
LEAGUE_AVG_PPM = 0.50
LEAGUE_STD_PPM = 0.18


class VarianceModel:
    """
    Models player-specific scoring variance for better tail predictions.
    
    The core insight: Books price means well but misprice tails.
    - Microwave scorers have higher upside variance
    - Catch-and-shoot players have boom-or-bust potential
    - Rim runners are more predictable
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize variance model."""
        self.db = db or DatabaseManager()
        self.config = get_config()
        
        # Cache for player archetypes
        self._archetype_cache: Dict[int, str] = {}
        self._variance_cache: Dict[int, float] = {}
    
    def get_player_stats_summary(self, player_id: int, window: int = 15) -> Dict[str, float]:
        """Get player's key stats for archetype classification."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get aggregate stats (no STDEV - not supported in SQLite)
            cursor.execute("""
                SELECT 
                    AVG(minutes) as avg_minutes,
                    AVG(points) as avg_points,
                    AVG(assists) as avg_assists,
                    SUM(fga) as total_fga,
                    SUM(fgm) as total_fgm,
                    SUM(fg3a) as total_3pa,
                    SUM(fg3m) as total_3pm,
                    SUM(points) as total_points,
                    SUM(minutes) as total_minutes,
                    SUM(assists) as total_assists,
                    COUNT(*) as games
                FROM player_logs
                WHERE player_id = ?
                AND minutes > 5
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            row = cursor.fetchone()
            
            # Get points list for std calculation
            cursor.execute("""
                SELECT points FROM player_logs
                WHERE player_id = ?
                AND minutes > 5
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            points_list = [r['points'] for r in cursor.fetchall() if r['points']]
        
        if not row or not row['games'] or row['games'] < 3:
            return {
                'avg_minutes': 20.0,
                'avg_points': 10.0,
                'usage': 0.18,
                'ppm': LEAGUE_AVG_PPM,
                'three_rate': 0.35,
                'rim_rate': 0.40,
                'ast_rate': 0.15,
                'points_std': 5.0,
            }
        
        # Calculate std in Python
        points_std = np.std(points_list) if len(points_list) >= 3 else 5.0
        
        total_fga = row['total_fga'] or 1
        total_3pa = row['total_3pa'] or 0
        total_minutes = row['total_minutes'] or 1
        total_points = row['total_points'] or 0
        total_assists = row['total_assists'] or 0
        
        # Calculate usage proxy
        usage = (total_fga + 0.44 * (row['total_fgm'] or 0)) / total_minutes / 1.5
        usage = min(0.40, max(0.10, usage))
        
        # Calculate rates
        three_rate = total_3pa / total_fga if total_fga > 0 else 0.35
        two_pt_fga = total_fga - total_3pa
        
        # Estimate rim rate from 2P%
        two_pt_fgm = (row['total_fgm'] or 0) - (row['total_3pm'] or 0)
        two_pct = two_pt_fgm / two_pt_fga if two_pt_fga > 0 else 0.50
        rim_rate = min(0.70, max(0.25, (two_pct - 0.35) / 0.30)) * (1 - three_rate)
        
        # Assist rate
        ast_rate = total_assists / total_minutes if total_minutes > 0 else 0.10
        
        return {
            'avg_minutes': row['avg_minutes'] or 20.0,
            'avg_points': row['avg_points'] or 10.0,
            'usage': usage,
            'ppm': total_points / total_minutes if total_minutes > 0 else 0.50,
            'three_rate': three_rate,
            'rim_rate': rim_rate,
            'ast_rate': ast_rate,
            'points_std': points_std,
        }
    
    def classify_archetype(self, player_id: int) -> str:
        """
        Classify player into an archetype based on their playing style.
        
        Returns:
            Archetype string (e.g., 'volume_star', 'microwave_scorer', etc.)
        """
        if player_id in self._archetype_cache:
            return self._archetype_cache[player_id]
        
        stats = self.get_player_stats_summary(player_id)
        
        # Classification logic (order matters - check most specific first)
        
        # Volume Star: High usage, high minutes
        if stats['usage'] >= 0.25 and stats['avg_minutes'] >= 28:
            archetype = 'volume_star'
        
        # Secondary Star
        elif stats['usage'] >= 0.20 and stats['avg_minutes'] >= 26:
            archetype = 'secondary_star'
        
        # Catch and Shoot: High 3PT rate
        elif stats['three_rate'] >= 0.50:
            archetype = 'catch_and_shoot'
        
        # Rim Runner: High rim rate, efficient
        elif stats['rim_rate'] >= 0.50 and stats['ppm'] >= 0.55:
            archetype = 'rim_runner'
        
        # Microwave Scorer: Bench player with high PPM
        elif stats['avg_minutes'] < 25 and stats['ppm'] >= 0.55:
            archetype = 'microwave_scorer'
        
        # Floor General: Low usage, high assists
        elif stats['usage'] < 0.18 and stats['ast_rate'] >= 0.20:
            archetype = 'floor_general'
        
        # Default to role player
        else:
            archetype = 'role_player'
        
        self._archetype_cache[player_id] = archetype
        return archetype
    
    def get_variance_multiplier(self, player_id: int) -> float:
        """Get the variance multiplier for a player based on their archetype."""
        archetype = self.classify_archetype(player_id)
        return PLAYER_ARCHETYPES[archetype]['variance_multiplier']
    
    def calculate_player_variance(self, player_id: int, 
                                   base_std: float = None) -> Tuple[float, str]:
        """
        Calculate player-specific variance (σ) for point predictions.
        
        Args:
            player_id: NBA player ID
            base_std: Base standard deviation (default uses historical)
            
        Returns:
            Tuple of (adjusted_std, archetype)
        """
        if player_id in self._variance_cache:
            archetype = self.classify_archetype(player_id)
            return self._variance_cache[player_id], archetype
        
        # Get player stats
        stats = self.get_player_stats_summary(player_id)
        archetype = self.classify_archetype(player_id)
        multiplier = PLAYER_ARCHETYPES[archetype]['variance_multiplier']
        
        # Base std from historical or defaults
        if base_std is None:
            base_std = stats['points_std'] if stats['points_std'] > 0 else 5.0
        
        # Apply archetype multiplier
        adjusted_std = base_std * multiplier
        
        # Additional adjustments based on 3PT rate
        # High 3PT shooters have more variance
        if stats['three_rate'] > 0.45:
            adjusted_std *= 1.1
        
        # Rim runners have less variance
        if stats['rim_rate'] > 0.50:
            adjusted_std *= 0.9
        
        # Cap at reasonable bounds
        adjusted_std = np.clip(adjusted_std, 2.0, 12.0)
        
        self._variance_cache[player_id] = adjusted_std
        return adjusted_std, archetype
    
    def get_shooting_form_reversion(self, player_id: int) -> Dict[str, Any]:
        """
        Analyze recent shooting form to detect Slumps or Hot Streaks.
        
        Key Insight: 
        If Volume (FGA) maintains but Efficiency (FG%) drops -> Expect Mean Reversion (Bounce back).
        If Volume drops -> True Role Reduction.
        
        Returns:
            Dict with 'multiplier', 'status' (normal/slump/hot), 'detail'
        """
        stats_season = self.get_player_stats_summary(player_id, window=30)
        stats_l5 = self.get_player_stats_summary(player_id, window=5)
        
        # Minimum sample
        if stats_season['avg_points'] < 8.0:
            return {'multiplier': 1.0, 'status': 'low_vol', 'detail': ''}
            
        # Calculate derived metrics
        fga_season = stats_season['avg_points'] / stats_season['ppm'] if stats_season['ppm'] > 0 else 10
        fga_l5 = stats_l5['avg_points'] / stats_l5['ppm'] if stats_l5['ppm'] > 0 else 10
        
        ppm_season = stats_season['ppm']
        ppm_l5 = stats_l5['ppm']
        
        # 1. Check Volume Maintenance
        # If taking >85% of season shots, role is safe
        volume_maintained = fga_l5 >= (fga_season * 0.85)
        
        # 2. Check Efficiency Deviation
        ppm_ratio = ppm_l5 / ppm_season if ppm_season > 0 else 1.0
        
        # Logic:
        # A. SLUMP (High Volume, Low Efficiency) -> Reversion Candidate (Boost)
        if volume_maintained and ppm_ratio < 0.80:
            # They are shooting <80% of normal efficiency but still shooting
            # Our L5-heavy baseline is punishing them too much. Correct upwards.
            # Example: 1.05x multiplier to pull mean back toward season
            return {
                'multiplier': 1.05,
                'status': 'slump_reversion',
                'detail': f"Slump (PPM {ppm_l5:.2f} vs {ppm_season:.2f}) but Vol Steady -> Reversion Boost"
            }
            
        # B. HOT STREAK (Sustainable?) -> Dampen slightly?
        elif volume_maintained and ppm_ratio > 1.25:
            # Shooting 25% better than season. Likely unsustainable.
            # L5-heavy baseline might be over-projecting. Dampen.
            return {
                'multiplier': 0.96, 
                'status': 'hot_dampen',
                'detail': f"Hot Streak (PPM {ppm_l5:.2f} vs {ppm_season:.2f}) -> Regression Dampen"
            }
            
        return {'multiplier': 1.0, 'status': 'normal', 'detail': 'Normal form'}
            
    def sample_with_archetype_variance(self, 
                                        player_id: int,
                                        predicted_mean: float,
                                        n_samples: int = 1000,
                                        market: str = 'points') -> np.ndarray:
        """
        Generate point samples with player-specific variance.
        
        [R1] Now uses FatTailedSampler for data-driven distributions
        (Student-t, Skew-Normal, or KDE) instead of hardcoded Gaussian/Gamma.
        Falls back to archetype-based sampling if FatTailedSampler unavailable.
        """
        adjusted_std, archetype = self.calculate_player_variance(player_id)
        
        # [R1] Try fat-tailed sampling first
        try:
            from .fat_tailed_sampler import FatTailedSampler
            sampler = FatTailedSampler(db=self.db)
            samples = sampler.sample(
                player_id=player_id,
                predicted_mean=predicted_mean,
                market=market,
                n_samples=n_samples,
                base_std=adjusted_std
            )
            if len(samples) > 0:
                return samples
        except Exception:
            pass  # Fall through to legacy sampling
        
        # Legacy fallback: archetype-based sampling
        if archetype == 'microwave_scorer':
            # Right-skewed: can explode but rarely goes very low
            samples = np.random.gamma(
                shape=(predicted_mean / adjusted_std) ** 2,
                scale=adjusted_std ** 2 / predicted_mean,
                size=n_samples
            )
        else:
            # Normal distribution for most players
            samples = np.random.normal(predicted_mean, adjusted_std, n_samples)
        
        return np.maximum(0, samples)


class PaceAdjuster:
    """
    Adjusts predictions based on game pace.
    
    Key insight: Use pace as a multiplier for possessions, NOT direct points.
    """
    
    LEAGUE_AVG_PACE = 100.0  # Possessions per 48 minutes
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize pace adjuster."""
        self.db = db or DatabaseManager()
    
    def get_expected_pace(self, team_abbr: str, opponent_abbr: str) -> float:
        """
        Get expected game pace from both teams.
        
        Returns:
            Expected game pace (possessions per 48 min)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get team pace
            cursor.execute("""
                SELECT pace FROM team_advanced_stats
                WHERE team_abbreviation = ?
                AND window_type = 'L10'
                ORDER BY stat_date DESC LIMIT 1
            """, (team_abbr,))
            team_row = cursor.fetchone()
            team_pace = team_row['pace'] if team_row and team_row['pace'] else self.LEAGUE_AVG_PACE
            
            # Get opponent pace
            cursor.execute("""
                SELECT pace FROM team_advanced_stats
                WHERE team_abbreviation = ?
                AND window_type = 'L10'
                ORDER BY stat_date DESC LIMIT 1
            """, (opponent_abbr,))
            opp_row = cursor.fetchone()
            opp_pace = opp_row['pace'] if opp_row and opp_row['pace'] else self.LEAGUE_AVG_PACE
        
        # Expected pace is average of both teams
        expected_pace = (team_pace + opp_pace) / 2
        return expected_pace
    
    def calculate_pace_multiplier(self, team_abbr: str, opponent_abbr: str) -> float:
        """
        Calculate pace-based possession multiplier.
        
        Returns:
            Multiplier relative to league average (1.0 = average pace)
        """
        expected_pace = self.get_expected_pace(team_abbr, opponent_abbr)
        
        # More possessions = more scoring opportunities
        # But don't overweight - cap at ±10%
        raw_multiplier = expected_pace / self.LEAGUE_AVG_PACE
        return np.clip(raw_multiplier, 0.90, 1.10)


# Convenience functions
def get_variance_model() -> VarianceModel:
    """Get variance model instance."""
    return VarianceModel()


def get_pace_adjuster() -> PaceAdjuster:
    """Get pace adjuster instance."""
    return PaceAdjuster()
