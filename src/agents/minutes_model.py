"""
Minutes Projection Model
=========================
Dedicated minutes projector that accounts for:
  1. Recent minutes trend (L5 vs L10 vs L15)
  2. Rest days impact (back-to-back penalty, extra rest bonus)
  3. Blowout risk from spread (garbage time / bench time)
  4. Opponent pace (faster pace = more possessions = potential minutes shift)
  5. Starter vs bench role stability
  6. Foul trouble history

Data Sources:
  - player_logs: minutes, is_starter, game_date
  - odds_snapshots: spread for blowout risk
  - team_advanced_stats: pace
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import logging

logger = logging.getLogger("MINUTES_MODEL")


class MinutesProjector:
    """
    Projects player minutes with confidence intervals.
    
    This is the single most important input to any stat projection —
    get minutes wrong and everything downstream breaks.
    """
    
    # Minutes adjustments by rest scenario
    REST_ADJUSTMENTS = {
        0: -2.5,   # Back-to-back: -2.5 min
        1: -0.5,   # No extra rest: slight penalty
        2: 0.0,    # Standard rest: baseline
        3: 0.5,    # Extra rest day: small boost
        4: 0.3,    # Extended rest: slight rust concern offsets benefit
    }
    
    # Blowout risk thresholds (spread -> minutes penalty for starters)
    BLOWOUT_SPREAD_THRESHOLD = 8.0  # Double-digit favorites
    BLOWOUT_MINUTES_PENALTY = -3.0  # Typically sit 4+ min in 4th quarter
    
    # Foul trouble: if player averages 3+ fouls, risk of early benching
    FOUL_TROUBLE_THRESHOLD = 3.0
    FOUL_TROUBLE_PENALTY = -1.5
    
    def __init__(self):
        pass
    
    def project_minutes(self, 
                        player_context: Dict[str, Any],
                        match_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project minutes for a player in a specific game.
        
        Returns:
            Dict with:
                - projected_minutes: float
                - minutes_floor: float (10th percentile)
                - minutes_ceiling: float (90th percentile)
                - confidence: str
                - adjustments: list of applied adjustments
                - minutes_std: float
        """
        stats = player_context.get('stats', {})
        adjustments = []
        
        # === BASELINE: Weighted recent averages ===
        min_L5 = stats.get('minutes_L5', stats.get('l5_minutes', 0))
        min_L10 = stats.get('minutes_L10', stats.get('l10_minutes', 0))
        min_L15 = stats.get('minutes_L15', stats.get('l15_minutes', 0))
        
        # Handle missing data
        if min_L5 == 0 and min_L10 == 0 and min_L15 == 0:
            # Try from raw logs
            games = player_context.get('recent_games', [])
            if games:
                recent_mins = [g.get('minutes', 0) for g in games[:15] if g.get('minutes', 0) > 0]
                if recent_mins:
                    min_L5 = np.mean(recent_mins[:5]) if len(recent_mins) >= 5 else np.mean(recent_mins)
                    min_L10 = np.mean(recent_mins[:10]) if len(recent_mins) >= 10 else np.mean(recent_mins)
                    min_L15 = np.mean(recent_mins[:15]) if len(recent_mins) >= 15 else np.mean(recent_mins)
        
        if min_L5 == 0:
            return self._default_projection()
        
        # Weighted baseline: 50% L5, 30% L10, 20% L15
        baseline = 0.50 * min_L5 + 0.30 * min_L10 + 0.20 * min_L15
        adjustments.append(f"Baseline: {baseline:.1f} min (50% L5 + 30% L10 + 20% L15)")
        
        projected = baseline
        
        # === MINUTES TREND ===
        min_trend = min_L5 - min_L15
        if abs(min_trend) > 2.0:
            trend_adj = min_trend * 0.3  # Partial trend continuation
            projected += trend_adj
            direction = "up" if trend_adj > 0 else "down"
            adjustments.append(f"Minutes trend {direction}: {trend_adj:+.1f} min")
        
        # === REST DAYS ===
        rest_days = player_context.get('rest_days', 2)
        if rest_days is None:
            rest_days = 2
        rest_key = min(rest_days, 4)
        rest_adj = self.REST_ADJUSTMENTS.get(rest_key, 0)
        if rest_adj != 0:
            projected += rest_adj
            label = "B2B penalty" if rest_days == 0 else f"{rest_days}-day rest"
            adjustments.append(f"{label}: {rest_adj:+.1f} min")
        
        # === BLOWOUT RISK ===
        spread = abs(match_context.get('spread', 0) or 0)
        if spread >= self.BLOWOUT_SPREAD_THRESHOLD:
            is_starter = player_context.get('is_starter', True)
            if is_starter:
                blowout_prob = min(1.0, (spread - self.BLOWOUT_SPREAD_THRESHOLD) / 10.0)
                blowout_adj = self.BLOWOUT_MINUTES_PENALTY * blowout_prob
                projected += blowout_adj
                adjustments.append(f"Blowout risk (spread {spread:.1f}): {blowout_adj:+.1f} min")
        
        # === OPPONENT PACE ===
        opp_pace = match_context.get('opp_pace', 100)
        team_pace = match_context.get('team_pace', 100)
        if opp_pace and team_pace:
            avg_pace = (opp_pace + team_pace) / 2
            pace_diff = avg_pace - 100  # Deviation from league average
            # Faster pace = slightly more minutes for rotation players
            # (more stoppages, more action, coaches keep players in)
            pace_adj = pace_diff * 0.05
            if abs(pace_adj) > 0.3:
                projected += pace_adj
                adjustments.append(f"Pace effect ({avg_pace:.0f}): {pace_adj:+.1f} min")
        
        # === STARTER STABILITY ===
        starter_rate = stats.get('starter_rate', player_context.get('starter_rate', 1.0))
        if starter_rate is not None and starter_rate < 0.8:
            # Inconsistent starter/bench — higher variance
            stability_adj = -1.0
            projected += stability_adj
            adjustments.append(f"Rotation instability ({starter_rate:.0%} starter rate): {stability_adj:+.1f} min")
        
        # === FOUL TROUBLE HISTORY ===
        avg_fouls = stats.get('avg_fouls', 0)
        if avg_fouls and avg_fouls >= self.FOUL_TROUBLE_THRESHOLD:
            projected += self.FOUL_TROUBLE_PENALTY
            adjustments.append(f"Foul trouble risk ({avg_fouls:.1f} avg PF): {self.FOUL_TROUBLE_PENALTY:+.1f} min")

        # === LINEUP-AWARE ROTATION ADJUSTMENT ===
        team_injuries = player_context.get('team_injuries', match_context.get('team_injuries', {})) or {}
        teammate_impact = match_context.get('usage_impact') or match_context.get('teammate_impact') or {}
        lineup_context = match_context.get('lineup_context') or player_context.get('lineup_context') or {}
        missing_rotation = sum(1 for prob in team_injuries.values() if prob < 0.5)
        if missing_rotation:
            starter_rate = stats.get('starter_rate', player_context.get('starter_rate', 1.0)) or 0
            is_starter = bool(player_context.get('is_starter', starter_rate >= 0.5))
            injury_adj = min(3.0, 0.6 * missing_rotation)
            if not is_starter:
                injury_adj *= 0.5
            projected += injury_adj
            adjustments.append(f"Lineup absences ({missing_rotation}): {injury_adj:+.1f} min")

        if teammate_impact:
            expected_minutes_boost = float(
                teammate_impact.get('expected_minutes_boost')
                or teammate_impact.get('minutes_boost')
                or 0.0
            )
            if abs(expected_minutes_boost) > 0.1:
                capped_boost = max(-2.0, min(3.0, expected_minutes_boost))
                projected += capped_boost
                adjustments.append(f"Teammate usage impact: {capped_boost:+.1f} min")

        if lineup_context:
            volatility_score = float(lineup_context.get('volatility_score', 0.0) or 0.0)
            role = str(lineup_context.get('player_role', player_context.get('player_role', 'rotation')))
            usage_delta = float(lineup_context.get('usage_delta', 0.0) or 0.0)
            role_change = bool(lineup_context.get('role_change', False))
            rotation_tightening = bool(lineup_context.get('rotation_tightening', False))
            significant_absence_cluster = bool(lineup_context.get('significant_absence_cluster', False))

            lineup_minutes_adj = 0.0
            if significant_absence_cluster:
                if role in ('star', 'starter'):
                    lineup_minutes_adj += 1.0
                else:
                    lineup_minutes_adj += 0.5
            if role_change:
                lineup_minutes_adj += 1.0 if role in ('starter', 'star') else 0.6
            if rotation_tightening and role == 'bench':
                lineup_minutes_adj -= 1.2
            if usage_delta > 0.025 and role in ('star', 'starter'):
                lineup_minutes_adj += 0.6

            if abs(lineup_minutes_adj) > 0.1:
                projected += lineup_minutes_adj
                adjustments.append(f"Lineup role context: {lineup_minutes_adj:+.1f} min")
        
        # === FLOOR / CEILING ===
        # Calculate std from recent games
        games = player_context.get('recent_games', [])
        if games:
            recent_mins = [g.get('minutes', 0) for g in games[:10] if g.get('minutes', 0) > 0]
            if recent_mins:
                min_std = np.std(recent_mins)
            else:
                min_std = 4.0
        else:
            min_std = stats.get('minutes_std', 4.0) or 4.0

        if lineup_context and float(lineup_context.get('volatility_score', 0.0) or 0.0) >= 0.35:
            min_std *= 1.15
            adjustments.append("Lineup volatility: widened minutes uncertainty")
        
        minutes_floor = max(0, projected - 1.28 * min_std)  # P10
        minutes_ceiling = projected + 1.28 * min_std  # P90
        
        # Clamp to reasonable range
        projected = max(0, min(48, projected))
        minutes_floor = max(0, minutes_floor)
        minutes_ceiling = min(48, minutes_ceiling)
        
        # === CONFIDENCE ===
        if min_std < 3.0 and starter_rate and starter_rate > 0.9:
            confidence = 'high'
        elif min_std < 5.0:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'projected_minutes': round(projected, 1),
            'minutes_floor': round(minutes_floor, 1),
            'minutes_ceiling': round(minutes_ceiling, 1),
            'minutes_std': round(min_std, 2),
            'confidence': confidence,
            'adjustments': adjustments,
            'baseline': round(baseline, 1),
        }
    
    def _default_projection(self) -> Dict[str, Any]:
        """Return default for players without data."""
        return {
            'projected_minutes': 0,
            'minutes_floor': 0,
            'minutes_ceiling': 0,
            'minutes_std': 10.0,
            'confidence': 'none',
            'adjustments': ['No minutes data available'],
            'baseline': 0,
        }
