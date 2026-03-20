"""
Player-vs-Team Matchup Model (R4)
===================================
3-level matchup specificity cascade:

  Level 1: Direct H2H (player vs exact team, needs 5+ games)
  Level 2: Archetype-vs-Scheme (e.g., PnR guard vs switching defense)
  Level 3: Position-vs-DvP (guard vs team's guard defense rank) [fallback]

Uses Bayesian shrinkage: raw H2H multiplier is pulled toward 1.0 
based on sample size (5 games → small pull, 15+ → full trust).
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("MATCHUP")


class MatchupModel:
    """
    Player-vs-opponent matchup model with 3-level specificity cascade.
    
    Instead of using team-level DvP (which treats all guards the same),
    this model tries specific matchup data first, then falls back to
    progressively less specific levels.
    """
    
    # Archetype-vs-scheme adjustments (empirical)
    # Key: (archetype, scheme) → scoring multiplier
    ARCHETYPE_SCHEME_TABLE = {
        # PnR-heavy players struggle vs switching defenses
        ('volume_star', 'SWITCH_HEAVY'):       0.94,
        ('secondary_star', 'SWITCH_HEAVY'):    0.95,
        # Shooters thrive vs drop coverage (open off screens)
        ('catch_and_shoot', 'DROP_COVERAGE'):   1.08,
        ('catch_and_shoot', 'SWITCH_HEAVY'):    0.96,
        # Rim runners feast vs poor paint protection
        ('rim_runner', 'WEAK_PAINT'):           1.10,
        ('rim_runner', 'ELITE_PAINT'):          0.90,
        # Microwave scorers thrive vs weak bench units
        ('microwave_scorer', 'WEAK_BENCH_D'):   1.12,
        # Volume scorers thrive vs slow pace teams (more iso opportunity)
        ('volume_star', 'SLOW_PACE_D'):         1.03,
        # Floor generals limited against pressing teams
        ('floor_general', 'PRESS_HEAVY'):       0.93,
    }
    
    def __init__(self, db=None):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self._cache: Dict[str, float] = {}
    
    def get_matchup_multiplier(self, player_id: int, opponent_abbr: str,
                                market: str = 'points') -> Dict[str, Any]:
        """
        Get matchup-adjusted scoring multiplier using 3-level cascade.
        
        Returns:
            Dict with 'multiplier', 'level', 'confidence', 'detail'
        """
        cache_key = f"{player_id}_{opponent_abbr}_{market}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Level 1: Direct H2H
        h2h = self._get_h2h_stats(player_id, opponent_abbr, market)
        if h2h['sample_size'] >= 5:
            ratio = h2h['avg_vs_opponent'] / max(h2h['overall_avg'], 1.0)
            # Bayesian shrinkage toward 1.0 based on sample size
            # 5 games → 33% trust, 15+ games → 100% trust
            shrinkage = min(1.0, h2h['sample_size'] / 15.0)
            multiplier = 1.0 + (ratio - 1.0) * shrinkage
            # Bound to avoid extreme adjustments
            multiplier = np.clip(multiplier, 0.80, 1.25)
            
            result = {
                'multiplier': multiplier,
                'level': 'H2H',
                'confidence': shrinkage,
                'detail': (f"H2H: {h2h['avg_vs_opponent']:.1f} {market} vs {opponent_abbr} "
                          f"(n={h2h['sample_size']}) vs overall {h2h['overall_avg']:.1f}"),
            }
            self._cache[cache_key] = result
            return result
        
        # Level 2: Archetype-vs-Scheme
        archetype = self._get_player_archetype(player_id)
        schemes = self._get_opponent_schemes(opponent_abbr)
        
        # [NEW] Dynamic Lookup: Check for learned biases first
        # We look for the MOST significant scheme match
        best_dynamic_mult = None
        best_dynamic_scheme = None
        
        for scheme in schemes:
            learned_bias = self._get_learned_bias(archetype, scheme)
            if learned_bias:
                # learned_bias is a float adjustment (e.g. +1.5 pts)
                # Convert to multiplier roughly: (Base + Adj) / Base
                # Assuming base points ~15.0 for meaningful calculation
                # This is a bit rough, but better than static rule
                est_base = 15.0 
                mult = (est_base + learned_bias) / est_base
                # Clip reasonable bounds
                mult = np.clip(mult, 0.85, 1.15)
                
                if best_dynamic_mult is None or abs(mult - 1.0) > abs(best_dynamic_mult - 1.0):
                    best_dynamic_mult = mult
                    best_dynamic_scheme = scheme
        
        if best_dynamic_mult:
            result = {
                'multiplier': best_dynamic_mult,
                'level': 'ARCHETYPE_SCHEME_LEARNED',
                'confidence': 0.7, # Higher confidence because it's data-driven
                'detail': f"Learned Bias [{archetype}] vs [{best_dynamic_scheme}] → {best_dynamic_mult:.2f}x",
            }
            self._cache[cache_key] = result
            return result
        
        # Fallback to Hardcoded Table if no learned data
        for scheme in schemes:
            key = (archetype, scheme)
            if key in self.ARCHETYPE_SCHEME_TABLE:
                multiplier = self.ARCHETYPE_SCHEME_TABLE[key]
                result = {
                    'multiplier': multiplier,
                    'level': 'ARCHETYPE_SCHEME_STATIC',
                    'confidence': 0.5,
                    'detail': f"Static Rule [{archetype}] vs [{scheme}] → {multiplier:.2f}x",
                }
                self._cache[cache_key] = result
                return result
        
        # Level 3: Position-vs-DvP (fallback)
        position = self._get_player_position(player_id)
        dvp_mult = self._get_position_dvp(position, opponent_abbr, market)
        
        result = {
            'multiplier': dvp_mult,
            'level': 'DvP',
            'confidence': 0.3,
            'detail': f"DvP: {position} vs {opponent_abbr} → {dvp_mult:.2f}x",
        }
        self._cache[cache_key] = result
        return result
    
    def _get_h2h_stats(self, player_id: int, opponent_abbr: str,
                        market: str = 'points') -> Dict[str, Any]:
        """Get player's stats against a specific opponent."""
        col_map = {'points': 'points', 'assists': 'assists', 'rebounds': 'rebounds'}
        col = col_map.get(market, 'points')
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Stats vs this opponent
                cursor.execute(f"""
                    SELECT AVG({col}) as avg_stat, COUNT(*) as games
                    FROM player_logs
                    WHERE player_id = ? AND opponent_abbreviation = ?
                    AND minutes > 10
                    ORDER BY game_date DESC
                    LIMIT 20
                """, (player_id, opponent_abbr))
                h2h_row = cursor.fetchone()
                
                # Overall stats (for comparison)
                cursor.execute(f"""
                    SELECT AVG({col}) as avg_stat
                    FROM player_logs
                    WHERE player_id = ? AND minutes > 10
                    ORDER BY game_date DESC
                    LIMIT 30
                """, (player_id,))
                overall_row = cursor.fetchone()
                
            return {
                'avg_vs_opponent': (h2h_row['avg_stat'] or 0) if h2h_row else 0,
                'overall_avg': (overall_row['avg_stat'] or 0) if overall_row else 0,
                'sample_size': (h2h_row['games'] or 0) if h2h_row else 0,
            }
        except Exception as e:
            logger.debug(f"H2H lookup failed for {player_id} vs {opponent_abbr}: {e}")
            return {'avg_vs_opponent': 0, 'overall_avg': 0, 'sample_size': 0}
    
    def _get_player_archetype(self, player_id: int) -> str:
        """Get archetype from VarianceModel (cached)."""
        try:
            from .variance_model import VarianceModel
            vm = VarianceModel(db=self.db)
            return vm.classify_archetype(player_id)
        except Exception:
            return 'role_player'
    
    def _get_opponent_schemes(self, opponent_abbr: str) -> list:
        """Get defensive scheme classifications from DefensiveSchemeAnalyzer."""
        try:
            from ..agents.defensive_schemes import DefensiveSchemeAnalyzer
            analyzer = DefensiveSchemeAnalyzer(db=self.db)
            result = analyzer.analyze_defense(opponent_abbr)
            return result.get('schemes', [])
        except Exception:
            return []
    
    def _get_player_position(self, player_id: int) -> str:
        """Get player position."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT position FROM players WHERE player_id = ?
                """, (player_id,))
                row = cursor.fetchone()
                return (row['position'] or 'G') if row else 'G'
        except Exception:
            return 'G'
    
    def _get_position_dvp(self, position: str, opponent_abbr: str,
                           market: str = 'points') -> float:
        """Get DvP multiplier by position (fallback)."""
        try:
            # Use existing DvP infrastructure
            from .dvp_model import DvPCalculator
            dvp = DvPCalculator(db=self.db)
            result = dvp.get_dvp_data(opponent_abbr, position, market)
            if result and 'multiplier' in result:
                return np.clip(result['multiplier'], 0.85, 1.15)
        except Exception:
            pass
        return 1.0  # Neutral if no data
    
    def _get_learned_bias(self, archetype: str, scheme: str) -> Optional[float]:
        """
        Query bias_tracker for specific scheme adjustments.
        Returns: Adjustment in points (e.g. +1.5, -0.8) or None.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                segment = f"{archetype}_vs_{scheme}"
                
                # Get latest significant bias
                cursor.execute("""
                    SELECT adjustment_applied
                    FROM bias_tracker
                    WHERE component = 'matchup'
                    AND segment = ?
                    AND is_significant = 1
                    ORDER BY analysis_date DESC
                    LIMIT 1
                """, (segment,))
                
                row = cursor.fetchone()
                if row:
                    return row['adjustment_applied']
        except Exception:
            pass
        return None

    def clear_cache(self):
        """Clear cached multipliers."""
        self._cache.clear()


# Convenience function
def get_matchup_model(db=None):
    """Get MatchupModel instance."""
    return MatchupModel(db=db)
