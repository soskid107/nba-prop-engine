"""
Edge Agent E1: Narrative Detector
==================================
Purpose: Detect where the line is shaped by human bias, not basketball.

This agent doesn't predict numbers. It asks:
- "Is this a star?" → Reputation inflation → UNDER bias
- "Is this a role player?" → Market neglect → OVER bias
- "Is this prime-time / revenge / bounce-back?" → Public overbet

Output: narrative_inflation score + bias_direction
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("NARRATIVE_DETECTOR")


class NarrativeDetector:
    """
    Detects narrative-driven line inflation.
    
    Core Insight (Azramund):
    "If the public has a story about this player tonight,
    the line already has that story priced in — and then some."
    """
    
    # Primetime games get more public action → inflated star lines
    PRIMETIME_TEAMS = {
        'LAL', 'GSW', 'BOS', 'MIL', 'PHX', 'DAL', 'NYK', 'PHI',
        'DEN', 'MIA', 'LAC', 'MEM'
    }
    
    def __init__(self, db_manager=None):
        self.db = db_manager
    
    def analyze(self,
                player_context: Dict[str, Any],
                match_context: Dict[str, Any],
                market_line: float,
                market_type: str = 'points') -> Dict[str, Any]:
        """
        Analyze narrative inflation around a player's line.
        
        Args:
            player_context: Player stats and metadata from DataGatherer
            match_context: Game context (opponent, spread, pace, etc.)
            market_line: The sportsbook line
            market_type: 'points', 'assists', 'rebounds'
            
        Returns:
            Dict with narrative analysis and score
        """
        inflation_sources = []
        inflation_score = 0  # 0-100
        bias_direction = "NEUTRAL"
        
        # Extract key stats
        stats = player_context.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_context.get('ppg_L5') or
                  player_context.get('points_L5') or 0)
        ppg_season = (stats.get('season_ppg') or
                      player_context.get('ppg_season') or ppg_L5)
        team = player_context.get('team', '')
        opponent = match_context.get('opponent', '')
        
        # ========================================
        # CHECK 1: Star Reputation Inflation
        # ========================================
        # Books inflate star lines because public always bets OVER on stars.
        # If market_line > recent performance, reputation is doing the work.
        is_star = ppg_L5 > 20
        
        if is_star and market_type == 'points':
            # How much is the line above recent performance?
            reputation_gap = market_line - ppg_L5
            
            if reputation_gap > 3.0:
                # Line is 3+ points above recent avg → heavy reputation pricing
                inflation_sources.append('star_reputation')
                inflation_score += min(30, int(reputation_gap * 5))
                bias_direction = "UNDER"
                
            elif reputation_gap > 1.0:
                inflation_sources.append('mild_star_inflation')
                inflation_score += 10
                bias_direction = "UNDER"
        
        # ========================================
        # CHECK 2: Role Player Neglect
        # ========================================
        # Books set low lines for role players based on name recognition,
        # not actual role. If a role player is getting minutes/usage, 
        # the line may be too low.
        is_role_player = ppg_L5 < 14
        
        if is_role_player and market_type == 'points':
            # Is the line below recent form?
            neglect_gap = ppg_L5 - market_line
            
            if neglect_gap > 2.0:
                inflation_sources.append('role_player_neglect')
                inflation_score += min(25, int(neglect_gap * 5))
                if bias_direction == "NEUTRAL":
                    bias_direction = "OVER"
        
        # ========================================
        # CHECK 3: Primetime / National TV Inflation
        # ========================================
        # Star players in primetime matchups get extra public action.
        is_primetime = (team in self.PRIMETIME_TEAMS and 
                        opponent in self.PRIMETIME_TEAMS)
        
        if is_primetime and is_star:
            inflation_sources.append('primetime_game')
            inflation_score += 15
            if bias_direction == "NEUTRAL":
                bias_direction = "UNDER"
        
        # ========================================
        # CHECK 4: Bounce-Back Narrative
        # ========================================
        # After a bad game, public expects "bounce back" → inflates line.
        # After a huge game, public expects repeat → also inflates.
        recent_games = player_context.get('recent_games', [])
        
        if recent_games and len(recent_games) >= 1:
            last_game_pts = recent_games[0].get('points', ppg_L5)
            
            # Bad last game + line stays high → bounce-back narrative
            if last_game_pts < ppg_L5 * 0.65 and market_line >= ppg_L5:
                inflation_sources.append('bounce_back_narrative')
                inflation_score += 15
                if bias_direction == "NEUTRAL":
                    bias_direction = "UNDER"
            
            # Monster last game + line jumped → recency bias
            if last_game_pts > ppg_L5 * 1.40 and market_line > ppg_L5 * 1.10:
                inflation_sources.append('recency_bias_hot')
                inflation_score += 12
                if bias_direction == "NEUTRAL":
                    bias_direction = "UNDER"
        
        # ========================================
        # CHECK 5: Revenge Game / Must-Win Narrative
        # ========================================
        # These stories are priced in. Public overbets them.
        h2h = match_context.get('h2h_history', [])
        if h2h:
            h2h_pts = [g.get('points', 0) for g in h2h]
            h2h_avg = sum(h2h_pts) / len(h2h_pts) if h2h_pts else 0
            
            # If player historically dominates this opponent AND line reflects it
            if h2h_avg > ppg_L5 * 1.15 and market_line > ppg_L5:
                inflation_sources.append('revenge_or_dominance_narrative')
                inflation_score += 10
        
        # ========================================
        # CHECK 6: Line vs Season Trajectory
        # ========================================
        # If a player is trending DOWN but line hasn't adjusted
        if ppg_season > 0 and ppg_L5 > 0:
            trend_gap = ppg_season - ppg_L5
            if trend_gap > 4.0 and market_line > ppg_L5 + 2:
                # Season average is much higher than recent, 
                # line anchored to season reputation
                inflation_sources.append('season_reputation_anchor')
                inflation_score += 20
                bias_direction = "UNDER"
        
        # Cap at 100
        inflation_score = min(100, inflation_score)
        
        # Classify inflation level
        if inflation_score >= 40:
            inflation_level = "high"
        elif inflation_score >= 20:
            inflation_level = "medium"
        elif inflation_score >= 10:
            inflation_level = "low"
        else:
            inflation_level = "none"
        
        return {
            'narrative_inflation': inflation_level,
            'bias_direction': bias_direction,
            'inflation_sources': inflation_sources,
            'inflation_score': inflation_score,
            'is_star': is_star,
            'is_primetime': is_primetime,
            'reputation_gap': market_line - ppg_L5 if ppg_L5 > 0 else 0,
        }
