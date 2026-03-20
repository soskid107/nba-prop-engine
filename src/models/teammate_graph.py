"""
Teammate Impact Graph (R8)
============================
Lightweight alternative to full GNN.

Captures: "How does Player A's scoring change when Player B is on/off?"

Built from historical game logs:
- If both Player A and B play → record both their stats
- If B is out → record A's stats separately
- Compare: A_with_B vs A_without_B = teammate impact coefficient

This captures interaction effects that traditional features miss:
- Luka + Kyrie: both score LESS together (usage competition)
- Jokic + Murray: Murray scores MORE with Jokic (assist creation)
- LeBron + AD: AD scores SAME, but LeBron scores LESS (usage sharing)
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("TEAMMATE_GRAPH")


class TeammateImpactGraph:
    """
    Builds a per-team coefficient matrix capturing teammate interaction effects.
    
    The matrix stores: "When Teammate B is OUT, Player A's scoring changes by Nx"
    coefficient > 1.0 = player scores MORE without teammate (usage absorbed)
    coefficient < 1.0 = player scores LESS without teammate (lost synergy)
    coefficient = 1.0 = no effect
    """
    
    def __init__(self, db=None):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self._matrix_cache: Dict[str, Dict[int, Dict[int, float]]] = {}
    
    def build_impact_matrix(self, team_abbr: str,
                             lookback_games: int = 50) -> Dict[int, Dict[int, float]]:
        """
        Build teammate impact coefficients for a team.
        
        Returns:
            Dict[player_id → Dict[teammate_id → impact_coefficient]]
            
            impact_coefficient = avg_points_without_teammate / avg_points_with_teammate
        """
        if team_abbr in self._matrix_cache:
            return self._matrix_cache[team_abbr]
        
        matrix: Dict[int, Dict[int, float]] = {}
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get team roster
                cursor.execute("""
                    SELECT DISTINCT player_id FROM player_logs
                    WHERE team_abbreviation = ?
                    AND game_date >= date('now', '-60 days')
                    AND minutes > 5
                """, (team_abbr,))
                roster = [r['player_id'] for r in cursor.fetchall()]
                
                if len(roster) < 5:
                    self._matrix_cache[team_abbr] = matrix
                    return matrix
                
                # For each player, get their game-by-game stats
                player_game_data: Dict[int, Dict[str, Dict]] = {}
                
                for pid in roster:
                    cursor.execute("""
                        SELECT game_date, points, assists, rebounds, minutes
                        FROM player_logs
                        WHERE player_id = ? AND team_abbreviation = ?
                        AND minutes > 5
                        ORDER BY game_date DESC LIMIT ?
                    """, (pid, team_abbr, lookback_games))
                    
                    games = {}
                    for row in cursor.fetchall():
                        games[row['game_date']] = {
                            'points': row['points'] or 0,
                            'assists': row['assists'] or 0,
                            'rebounds': row['rebounds'] or 0,
                            'minutes': row['minutes'] or 0,
                        }
                    player_game_data[pid] = games
                
                # Build the coefficient matrix
                for player_id in roster:
                    matrix[player_id] = {}
                    player_games = player_game_data.get(player_id, {})
                    
                    if not player_games:
                        continue
                    
                    for teammate_id in roster:
                        if teammate_id == player_id:
                            continue
                        
                        teammate_games = player_game_data.get(teammate_id, {})
                        teammate_dates = set(teammate_games.keys())
                        
                        # Split player's games by teammate presence
                        with_pts = []
                        without_pts = []
                        
                        for date, stats in player_games.items():
                            if date in teammate_dates:
                                with_pts.append(stats['points'])
                            else:
                                without_pts.append(stats['points'])
                        
                        # Need minimum samples for both conditions
                        if len(with_pts) >= 5 and len(without_pts) >= 3:
                            avg_with = np.mean(with_pts)
                            avg_without = np.mean(without_pts)
                            
                            if avg_with > 0:
                                # Raw coefficient
                                raw_coeff = avg_without / avg_with
                                
                                # Bayesian shrinkage toward 1.0
                                # More games without → more trust in coefficient
                                n_without = len(without_pts)
                                shrinkage = min(1.0, n_without / 10.0)
                                coeff = 1.0 + (raw_coeff - 1.0) * shrinkage
                                
                                # Bound to reasonable range
                                coeff = np.clip(coeff, 0.75, 1.35)
                                
                                matrix[player_id][teammate_id] = coeff
                
        except Exception as e:
            logger.warning(f"Impact matrix build failed for {team_abbr}: {e}")
        
        self._matrix_cache[team_abbr] = matrix
        return matrix
    
    def get_injury_impact_multiplier(self, player_id: int, team_abbr: str,
                                      injuries: Dict[int, float]) -> Dict[str, Any]:
        """
        Calculate net scoring multiplier from all injured teammates.
        
        Args:
            player_id: Target player
            team_abbr: Team abbreviation
            injuries: Dict of {teammate_id: probability_of_playing}
            
        Returns:
            Dict with 'multiplier', 'detail', 'significant_effects'
        """
        matrix = self.build_impact_matrix(team_abbr)
        
        if player_id not in matrix or not matrix[player_id]:
            return {
                'multiplier': 1.0,
                'detail': 'No teammate interaction data',
                'significant_effects': [],
            }
        
        player_coefficients = matrix[player_id]
        net_multiplier = 1.0
        significant_effects: List[Dict] = []
        
        for teammate_id, p_play in injuries.items():
            if teammate_id not in player_coefficients:
                continue
            
            if p_play >= 0.5:  # Likely playing, skip
                continue
            
            coeff = player_coefficients[teammate_id]
            p_out = 1.0 - p_play
            
            # Apply coefficient scaled by probability of absence
            # If p_out=1.0 and coeff=1.15, player scores 15% more
            # If p_out=0.7 and coeff=1.15, effective boost = 1 + (0.15 * 0.7) = 1.105
            effective_coeff = 1.0 + (coeff - 1.0) * p_out
            net_multiplier *= effective_coeff
            
            if abs(coeff - 1.0) > 0.05:  # >5% effect is significant
                significant_effects.append({
                    'teammate_id': teammate_id,
                    'coefficient': coeff,
                    'p_out': p_out,
                    'direction': 'BOOST' if coeff > 1.0 else 'DRAG',
                })
        
        # Cap total multiplier
        net_multiplier = np.clip(net_multiplier, 0.80, 1.30)
        
        detail_parts = []
        for eff in significant_effects:
            dir_symbol = '↑' if eff['direction'] == 'BOOST' else '↓'
            detail_parts.append(
                f"TM#{eff['teammate_id']} OUT {dir_symbol}{abs(eff['coefficient']-1)*100:.0f}%"
            )
        
        return {
            'multiplier': net_multiplier,
            'detail': ', '.join(detail_parts) if detail_parts else 'No significant effects',
            'significant_effects': significant_effects,
        }
    
    def get_synergy_report(self, team_abbr: str) -> List[Dict[str, Any]]:
        """
        Generate human-readable synergy report for a team.
        Shows strongest positive and negative teammate interactions.
        """
        matrix = self.build_impact_matrix(team_abbr)
        effects = []
        
        for pid, teammates in matrix.items():
            for tid, coeff in teammates.items():
                if abs(coeff - 1.0) > 0.08:  # >8% effect
                    effects.append({
                        'player_id': pid,
                        'teammate_id': tid,
                        'coefficient': coeff,
                        'effect': 'scores MORE without' if coeff > 1.0 else 'scores LESS without',
                        'magnitude': abs(coeff - 1.0),
                    })
        
        # Sort by magnitude (strongest effects first)
        effects.sort(key=lambda x: x['magnitude'], reverse=True)
        return effects[:20]  # Top 20 effects
    
    def clear_cache(self):
        """Clear cached matrices."""
        self._matrix_cache.clear()


# Convenience function
def get_teammate_graph(db=None):
    """Get TeammateImpactGraph instance."""
    return TeammateImpactGraph(db=db)
