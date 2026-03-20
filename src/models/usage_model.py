"""
Usage Model

Predicts player usage rate (portion of team possessions used) based on:
1. Baseline usage (season/L10 average)
2. On/off delta for missing teammates (injury adjustment)
3. Role-based behavior (starter vs bench scorer)

Key insight: Usage is NOT season average. It's roster-dependent.
When a star is out, specific players see usage jumps, not everyone.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..utils.database import DatabaseManager
from ..utils.config import get_config


class UsageModel:
    """
    Predicts player usage rate considering roster context.
    
    Usage rate = % of team possessions used by player while on floor.
    Typical range: 10-35% (role player to star).
    """
    
    # Usage caps by role (usage explosions are bounded)
    USAGE_CAPS = {
        'star': 0.45,           # Superstars can hit 45% (Luka/Giannis)
        'secondary_star': 0.38, # e.g. Jaylen Brown, Kyrie
        'starter': 0.32,        # Standard starter
        'bench_scorer': 0.35,   # Microwave scorers can get hot
        'third_option': 0.28,
        'role_player': 0.22,
        'volume_star': 0.45,    # Alias
    }
    
    # Default on/off usage bump when a star is out
    # Key: role, Value: typical usage increase
    DEFAULT_USAGE_BUMPS = {
        'secondary_star': 0.04,   # +4% usage when primary star out
        'third_option': 0.03,     # +3%
        'bench_scorer': 0.02,     # +2%
        'role_player': 0.01,      # +1%
    }
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize usage model."""
        self.db = db or DatabaseManager()
        self.config = get_config()
        
        # Cache for player baseline usage
        self._usage_cache: Dict[int, float] = {}
        
        # Team hierarchy cache (who benefits when who's out)
        self._team_hierarchy: Dict[str, Dict[int, List[int]]] = {}
    
    def get_baseline_usage(self, player_id: int, window: int = 10) -> float:
        """
        Get player's baseline usage rate from recent games.
        
        Args:
            player_id: NBA player ID
            window: Number of recent games to average
            
        Returns:
            Usage rate as decimal (e.g., 0.25 for 25%)
        """
        # Check cache first
        if player_id in self._usage_cache:
            return self._usage_cache[player_id]
        
        # Calculate from game logs
        # Usage proxy = (FGA + 0.44*FTA + TOV) / Minutes * TeamPace
        # Simplified: Points share as usage proxy
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get player's recent stats
            cursor.execute("""
                SELECT 
                    pl.minutes, pl.points, pl.fga, pl.fta, pl.turnovers,
                    pl.team_abbreviation, pl.game_date
                FROM player_logs pl
                WHERE pl.player_id = ?
                AND pl.minutes > 0
                ORDER BY pl.game_date DESC
                LIMIT ?
            """, (player_id, window))
            
            rows = cursor.fetchall()
            
            if not rows:
                return 0.18  # Default to league average
            
            # Calculate weighted usage
            total_weighted_usage = 0.0
            total_weight = 0.0
            
            latest_date = datetime.strptime(rows[0]['game_date'], '%Y-%m-%d')
            decay_factor = 0.95
            
            for row in rows:
                minutes = row['minutes'] or 0
                if minutes < 5:
                    continue
                    
                fga = row['fga'] or 0
                fta = row['fta'] or 0
                tov = row['turnovers'] or 0
                
                # Usage formula: (FGA + 0.44*FTA + TOV) per minute
                possession_used = fga + 0.44 * fta + tov
                usage_per_min = possession_used / minutes
                
                # Convert to usage rate (rough approximation)
                usage_rate = min(usage_per_min / 1.5, 0.40)
                
                # Apply weight
                game_date = datetime.strptime(row['game_date'], '%Y-%m-%d')
                days_ago = (latest_date - game_date).days
                weight = decay_factor ** days_ago
                
                total_weighted_usage += usage_rate * weight
                total_weight += weight
            
            baseline = total_weighted_usage / total_weight if total_weight > 0 else 0.18
            
            # Cache it
            self._usage_cache[player_id] = baseline
            
            return baseline
    
    def classify_player_role(self, player_id: int, team_abbr: str) -> str:
        """
        Classify player's role on their team.
        
        Returns:
            One of: 'star', 'secondary_star', 'third_option', 'bench_scorer', 'role_player'
        """
        baseline = self.get_baseline_usage(player_id)
        
        # Also check minutes to distinguish starters from bench
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(minutes) as avg_min, AVG(is_starter) as starter_pct
                FROM player_logs
                WHERE player_id = ?
                ORDER BY game_date DESC
                LIMIT 10
            """, (player_id,))
            row = cursor.fetchone()
            
            avg_min = row['avg_min'] or 0 if row else 0
            starter_pct = row['starter_pct'] or 0 if row else 0
        
        # Classification logic
        if baseline >= 0.28 and avg_min >= 30:
            return 'star'
        elif baseline >= 0.22 and avg_min >= 26:
            return 'secondary_star'
        elif baseline >= 0.18 and starter_pct >= 0.5:
            return 'third_option'
        elif baseline >= 0.20 and avg_min < 24:
            return 'bench_scorer'
        else:
            return 'role_player'
    
    def _get_team_roster(self, team_abbr: str) -> List[int]:
        """Get recent roster for team (cached for performance)."""
        if hasattr(self, '_roster_cache') and team_abbr in self._roster_cache:
            return self._roster_cache[team_abbr]
            
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT player_id 
                FROM player_logs
                WHERE team_abbreviation = ?
                AND game_date >= date('now', '-20 days')
            """, (team_abbr,))
            roster = [row['player_id'] for row in cursor.fetchall()]
            
        if not hasattr(self, '_roster_cache'):
            self._roster_cache = {}
        self._roster_cache[team_abbr] = roster
        return roster

    def _get_absorption_score(self, role: str) -> int:
        """Score determining how much vacated usage a role absorbs."""
        scores = {
            'star': 10,           # Stars eat the most
            'secondary_star': 7,  # Major beneficiaries
            'bench_scorer': 6,    # Microwave scorers get green light
            'third_option': 4,    # Starters get some
            'starter': 3,
            'role_player': 1
        }
        return scores.get(role, 1)

    def _get_on_off_splits(self, player_id: int, team_abbr: str,
                            lookback_games: int = 50) -> Dict[int, Dict[str, Any]]:
        """
        [R3] Calculate player usage when specific teammates are in vs out.
        
        Returns:
            Dict[teammate_id -> {'usage_with': float, 'usage_without': float,
                                  'points_with': float, 'points_without': float,
                                  'games_with': int, 'games_without': int}]
        """
        splits: Dict[int, Dict] = {}
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get roster (teammates)
                roster = self._get_team_roster(team_abbr)
                
                # Get this player's recent game dates + stats
                cursor.execute("""
                    SELECT game_date, points, fga, fta, turnovers, minutes
                    FROM player_logs
                    WHERE player_id = ? AND team_abbreviation = ?
                    AND minutes > 5
                    ORDER BY game_date DESC LIMIT ?
                """, (player_id, team_abbr, lookback_games))
                player_games = cursor.fetchall()
                
                if not player_games:
                    return splits
                
                # For each teammate, check which of those games they also played
                for teammate_id in roster:
                    if teammate_id == player_id:
                        continue
                    
                    # Get teammate's game dates (minutes > 0 = playing)
                    cursor.execute("""
                        SELECT game_date FROM player_logs
                        WHERE player_id = ? AND team_abbreviation = ?
                        AND minutes > 5
                        ORDER BY game_date DESC LIMIT ?
                    """, (teammate_id, team_abbr, lookback_games))
                    teammate_dates = {r['game_date'] for r in cursor.fetchall()}
                    
                    # Split player's games into WITH and WITHOUT teammate
                    with_games = []
                    without_games = []
                    
                    for g in player_games:
                        # Calculate usage from game stats
                        mins = g['minutes'] or 0
                        if mins < 5:
                            continue
                        fga = g['fga'] or 0
                        fta = g['fta'] or 0
                        tov = g['turnovers'] or 0
                        usage = min((fga + 0.44 * fta + tov) / mins / 1.5, 0.40)
                        pts = g['points'] or 0
                        
                        if g['game_date'] in teammate_dates:
                            with_games.append({'usage': usage, 'points': pts})
                        else:
                            without_games.append({'usage': usage, 'points': pts})
                    
                    # Only store if we have minimum sample for both
                    if len(with_games) >= 3 and len(without_games) >= 3:
                        splits[teammate_id] = {
                            'usage_with': np.mean([g['usage'] for g in with_games]),
                            'usage_without': np.mean([g['usage'] for g in without_games]),
                            'points_with': np.mean([g['points'] for g in with_games]),
                            'points_without': np.mean([g['points'] for g in without_games]),
                            'games_with': len(with_games),
                            'games_without': len(without_games),
                        }
        except Exception as e:
            import logging
            logging.getLogger('USAGE').warning(f"On/off splits failed for {player_id}: {e}")
        
        return splits

    def predict_adjusted_usage(self,
                               player_id: int,
                               team_abbr: str,
                               injuries: Dict[int, float]) -> Tuple[float, float]:
        """
        [R3] Predict usage with lineup-conditional on/off splits,
        falling back to Vacuum Theory if insufficient data.
        """
        baseline = self.get_baseline_usage(player_id)
        
        # ======================================
        # [R3] TRY ON/OFF SPLITS FIRST
        # ======================================
        on_off_delta = 0.0
        on_off_used = False
        absent_count = 0
        
        try:
            splits = self._get_on_off_splits(player_id, team_abbr)
            
            for teammate_id, p_play in injuries.items():
                if p_play >= 0.5:  # Likely playing, skip
                    continue
                
                p_out = 1.0 - p_play
                
                if teammate_id in splits:
                    split = splits[teammate_id]
                    if split['games_without'] >= 3:
                        # Use empirical on/off delta
                        delta = (split['usage_without'] - split['usage_with']) * p_out
                        on_off_delta += delta
                        on_off_used = True
                        absent_count += 1
            
            # Compound diminishing returns for multiple absences
            # Each additional absence has 70% of previous impact
            if absent_count > 1:
                on_off_delta *= (1.0 - 0.3 * (absent_count - 1) / absent_count)
        except Exception:
            pass
        
        # If we got meaningful on/off data, use it
        if on_off_used and abs(on_off_delta) > 0.005:
            adjusted_usage = baseline + on_off_delta
            adjusted_usage = np.clip(adjusted_usage, 0.08, 0.40)
            
            # Cap at Role Ceiling
            player_role = self.classify_player_role(player_id, team_abbr)
            cap = self.USAGE_CAPS.get(player_role, 0.35)
            adjusted_usage = min(adjusted_usage, cap)
            
            std = 0.025  # Lower uncertainty — data-driven
            if absent_count > 2:
                std = 0.035  # More chaos with many absences
            return adjusted_usage, std
        
        # ======================================
        # FALLBACK: Original Vacuum Theory
        # ======================================
        # 1. Identify Roster Status
        roster_ids = self._get_team_roster(team_abbr)
        active_players = []
        vacated_usage = 0.0
        
        for pid in roster_ids:
            p_play = injuries.get(pid, 1.0)
            if p_play < 0.5:
                u_base = self.get_baseline_usage(pid)
                vacated_usage += (u_base * 0.85)
            else:
                active_players.append(pid)
                
        if player_id not in active_players and injuries.get(player_id, 1.0) >= 0.5:
            active_players.append(player_id)
            
        if player_id not in active_players:
            return baseline, 0.0
            
        # 2. Calculate Total Absorption Score
        total_absorption = 0
        player_absorption = 0
        
        for pid in active_players:
            role = self.classify_player_role(pid, team_abbr)
            score = self._get_absorption_score(role)
            total_absorption += score
            if pid == player_id:
                player_absorption = score
                
        # 3. Distribute Vacated Usage
        if total_absorption > 0 and vacated_usage > 0:
            share = player_absorption / total_absorption
            usage_bump = vacated_usage * share
            max_bump = min(0.08, baseline * 0.35)
            usage_bump = min(usage_bump, max_bump)
            adjusted_usage = baseline + usage_bump
        else:
            adjusted_usage = baseline

        # Cap at Role Ceiling
        player_role = self.classify_player_role(player_id, team_abbr)
        cap = self.USAGE_CAPS.get(player_role, 0.35)
        
        # Special exception: If you are the LAST STAR standing, cap is higher (e.g. Harden mode)
        if player_role in ['star', 'secondary_star']:
             # Check if other stars are out
             stars_out = any(
                 self.classify_player_role(pid, team_abbr) in ['star', 'secondary_star'] 
                 and injuries.get(pid, 1.0) < 0.5 
                 for pid in roster_ids
             )
             if stars_out:
                 cap += 0.04 # Harden Rule
                 
        adjusted_usage = min(adjusted_usage, cap)
        
        # Calculate Variance
        std = 0.03 # Base
        if player_role == 'bench_scorer': std = 0.05
        if vacated_usage > 0.05: std += 0.01 # More chaos when lots of usage is missing
        
        return adjusted_usage, std

    def predict_distribution(self, 
                            player_id: int,
                            team_abbr: str,
                            injuries: Dict[int, float],
                            n_samples: int = 1000) -> np.ndarray:
        """
        Sample usage rate distribution for Monte Carlo.
        """
        mean_usage, std = self.predict_adjusted_usage(
            player_id, team_abbr, injuries
        )
        
        # Sample from truncated normal
        samples = np.random.normal(mean_usage, std, n_samples)
        samples = np.clip(samples, 0.05, 0.45) 
        
        return samples
    
    def get_team_usage_context(self, team_abbr: str, 
                                injuries: Dict[int, float]) -> Dict[str, Any]:
        """
        Get usage redistribution context for a team.
        
        Useful for debugging and understanding model behavior.
        
        Returns:
            Dict with usage redistribution details
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get team roster from recent games
            cursor.execute("""
                SELECT DISTINCT player_id 
                FROM player_logs
                WHERE team_abbreviation = ?
                AND game_date >= date('now', '-30 days')
            """, (team_abbr,))
            
            roster = [row['player_id'] for row in cursor.fetchall()]
        
        context = {
            'team': team_abbr,
            'injured_stars': [],
            'usage_bumps': {},
        }
        
        for player_id in roster:
            if player_id in injuries and injuries[player_id] < 0.5:
                role = self.classify_player_role(player_id, team_abbr)
                if role in ['star', 'secondary_star']:
                    context['injured_stars'].append({
                        'player_id': player_id,
                        'role': role,
                        'p_play': injuries[player_id]
                    })
        
        # Calculate bumps for each active player
        for player_id in roster:
            if player_id not in injuries or injuries[player_id] >= 0.5:
                usage, std = self.predict_adjusted_usage(
                    player_id, team_abbr, injuries
                )
                baseline = self.get_baseline_usage(player_id)
                if usage > baseline + 0.01:
                    context['usage_bumps'][player_id] = {
                        'baseline': baseline,
                        'adjusted': usage,
                        'bump': usage - baseline
                    }
        
        return context


# Convenience function
def get_usage_model() -> UsageModel:
    """Get usage model instance."""
    return UsageModel()
