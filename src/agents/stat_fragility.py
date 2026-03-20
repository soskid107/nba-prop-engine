"""
Edge Agent E2: Stat Fragility Agent
=====================================
Purpose: Count how many normal game scripts KILL this stat.

This is the most important edge agent.

It asks: "Can this stat disappear without anyone noticing?"
If yes → strong UNDER.

5 Game Scripts Simulated:
1. Blowout       → kills points (bench comes in)
2. Tight game    → lowers ceiling (ball sticks, fewer possessions)
3. Foul trouble  → kills all stats (minutes vanish)
4. Hot teammate  → kills usage for secondary options
5. Defensive adj → kills usage/assists (switching, zone, help D)

Kill Count → Decision:
  ≥ 3 scripts kill → GREENLIGHT (strong edge)
  2 scripts kill   → Caution
  1 or fewer       → Discard (no structural edge)
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("STAT_FRAGILITY")


class StatFragilityAgent:
    """
    Simulates game scripts and counts failure paths for a stat.
    
    Core Insight (Azramund):
    "If many normal game scripts kill this stat, that's not bad luck —
    that's structural fragility. And fragility is edge."
    """
    
    # Player roles and their stat fragility profiles
    # Higher number = more fragile (stat can disappear quietly)
    ROLE_FRAGILITY = {
        # Primary initiators — stats are structurally needed
        'volume_star': 0.2,
        'star': 0.25,
        'primary_ball_handler': 0.2,
        
        # Secondary options — stats are partly optional
        'secondary_star': 0.45,
        'third_option': 0.60,
        
        # Role-dependent — stats completely optional
        'microwave_scorer': 0.70,
        'catch_and_shoot': 0.80,
        'rim_runner': 0.75,
        'floor_general': 0.40,  # Assists less fragile for PGs
        'role_player': 0.80,
        'bench_scorer': 0.75,
        'six_man': 0.65,
        'starter': 0.55,
        'rotation': 0.85,
        'deep_bench': 0.95,
    }
    
    def __init__(self, db_manager=None):
        self.db = db_manager
    
    def analyze(self,
                player_context: Dict[str, Any],
                match_context: Dict[str, Any],
                market_line: float,
                market_type: str = 'points') -> Dict[str, Any]:
        """
        Analyze stat fragility across 5 game scripts.
        
        Returns:
            Dict with kill count, survival scripts, replaceability, and score
        """
        kill_scripts = []
        survival_scripts = []
        
        # Run each script analysis
        scripts = [
            ('blowout', self._script_blowout),
            ('tight_game', self._script_tight_game),
            ('foul_trouble', self._script_foul_trouble),
            ('hot_teammate', self._script_hot_teammate),
            ('defensive_adjustment', self._script_defensive_adj),
        ]
        
        script_details = {}
        for script_name, script_fn in scripts:
            kills, reason = script_fn(
                player_context, match_context, market_line, market_type
            )
            script_details[script_name] = {
                'kills': kills,
                'reason': reason
            }
            if kills:
                kill_scripts.append(script_name)
            else:
                survival_scripts.append(script_name)
        
        kill_count = len(kill_scripts)
        
        # Role Replaceability
        # "Can this stat disappear without anyone noticing?"
        replaceability = self._assess_replaceability(
            player_context, market_type
        )
        
        # Calculate direction
        if kill_count >= 3:
            preferred_direction = "UNDER"
        elif kill_count == 2 and replaceability in ('high', 'very_high'):
            preferred_direction = "UNDER"
        elif kill_count <= 1 and replaceability == 'low':
            preferred_direction = "OVER"
        else:
            preferred_direction = "NEUTRAL"
        
        # Calculate score (0-100)
        # Kill count drives 60% of score, replaceability drives 40%
        kill_score = min(60, kill_count * 20)
        
        repl_scores = {
            'very_high': 40, 'high': 30, 'medium': 15, 'low': 5
        }
        repl_score = repl_scores.get(replaceability, 15)
        
        total_score = kill_score + repl_score
        
        # Is the ceiling NEEDED or OPTIONAL? (Edge Question Q4)
        ceiling_needed = self._is_ceiling_needed(player_context, match_context)
        
        return {
            'scripts_analyzed': 5,
            'kill_count': kill_count,
            'kill_scripts': kill_scripts,
            'survival_scripts': survival_scripts,
            'script_details': script_details,
            'replaceability': replaceability,
            'ceiling_needed': ceiling_needed,
            'preferred_direction': preferred_direction,
            'score': min(100, total_score),
        }
    
    # ==========================================
    # GAME SCRIPT SIMULATORS
    # ==========================================
    
    def _script_blowout(self, player_ctx: Dict, match_ctx: Dict,
                        line: float, market: str) -> tuple:
        """
        Script 1: Blowout
        Kills: points, assists (starters sit in 4Q)
        Survives: rebounds (sometimes), assists (occasionally)
        """
        spread = abs(match_ctx.get('spread', 0) or 0)
        
        if spread >= 8:
            # High spread = likely blowout
            if market in ('points', 'assists'):
                return True, f"Spread {spread:.1f} → starters likely sit Q4"
            elif market == 'rebounds':
                if spread >= 12:
                    return True, f"Extreme spread {spread:.1f} → garbage time for all"
        
        # Also check if this player's team is the heavy favorite
        # (they'd be the ones sitting)
        team_spread = match_ctx.get('team_spread', 0) or 0
        if team_spread < -10:
            return True, f"Team is {abs(team_spread):.0f}pt favorite → minutes cut risk"
        
        return False, "Competitive game expected"
    
    def _script_tight_game(self, player_ctx: Dict, match_ctx: Dict,
                           line: float, market: str) -> tuple:
        """
        Script 2: Tight game
        Kills: ceiling (ball sticks, fewer shots, more cautious play)
        Doesn't kill: the stat completely, but caps upside
        
        This script kills the OVER when the line is already high.
        """
        spread = abs(match_ctx.get('spread', 0) or 0)
        pace = match_ctx.get('opp_pace', match_ctx.get('expected_pace', 100)) or 100
        
        stats = player_ctx.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_ctx.get('ppg_L5') or
                  player_ctx.get('points_L5') or 0)
        
        if spread < 3 and pace < 98:
            # Close game + slow pace → ceiling killer
            if market == 'points' and line > ppg_L5 * 1.10:
                return True, f"Close game + slow pace ({pace:.0f}) → ceiling capped"
            if market == 'assists' and line > 8:
                return True, "Tight game → ball becomes conservative"
        
        return False, "Game script doesn't restrict this stat"
    
    def _script_foul_trouble(self, player_ctx: Dict, match_ctx: Dict,
                             line: float, market: str) -> tuple:
        """
        Script 3: Foul trouble
        Kills: ALL stats (player sits for extended stretches)
        
        Based on: player's historical foul rate + opponent's style
        """
        stats = player_ctx.get('stats', {})
        fouls_per_game = stats.get('fouls_L5', stats.get('fouls_per_game', 2.5))
        minutes_L5 = player_ctx.get('minutes_L5', stats.get('l5_minutes', 30))
        
        # Foul rate per 36 minutes
        if minutes_L5 > 0:
            foul_rate_per36 = (fouls_per_game / minutes_L5) * 36
        else:
            foul_rate_per36 = 3.0
        
        # High foul rate (>4 per 36) = foul trouble risk
        if foul_rate_per36 > 4.0:
            return True, f"High foul rate ({foul_rate_per36:.1f}/36) → foul trouble risk"
        
        # Also consider: if opponent draws lots of fouls
        opp_ft_rate = match_ctx.get('opp_ft_rate', 0.25)
        if foul_rate_per36 > 3.5 and opp_ft_rate > 0.30:
            return True, f"Foul-prone ({foul_rate_per36:.1f}/36) vs high FT-rate team"
        
        return False, "Low foul trouble risk"
    
    def _script_hot_teammate(self, player_ctx: Dict, match_ctx: Dict,
                             line: float, market: str) -> tuple:
        """
        Script 4: Hot teammate
        Kills: usage/points for secondary options
        
        If this isn't the #1 option, a hot teammate steals possessions.
        """
        stats = player_ctx.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_ctx.get('ppg_L5') or
                  player_ctx.get('points_L5') or 0)
        
        # Is this player a secondary or lower option?
        role = player_ctx.get('player_role',
                             player_ctx.get('role', 'role_player'))
        
        secondary_roles = {
            'secondary_star', 'third_option', 'microwave_scorer',
            'catch_and_shoot', 'rim_runner', 'role_player',
            'bench_scorer', 'six_man', 'starter', 'rotation'
        }
        
        if role in secondary_roles and market in ('points', 'assists'):
            # Check if there's a high-usage teammate
            team_injuries = player_ctx.get('team_injuries', {})
            
            # If star teammate is healthy, hot teammate script is alive
            if not team_injuries:
                # All teammates available → usage can be stolen
                return True, f"Secondary role ({role}) → hot teammate steals usage"
            
            # If star teammate is OUT, this script flips (player gets MORE usage)
            key_teammate_out = False
            for tid, p_play in team_injuries.items():
                if p_play < 0.5:
                    key_teammate_out = True
                    break
            
            if not key_teammate_out:
                return True, f"All stars healthy → usage can shift away from {role}"
        
        return False, "Primary option or no teammate usage threat"
    
    def _script_defensive_adj(self, player_ctx: Dict, match_ctx: Dict,
                              line: float, market: str) -> tuple:
        """
        Script 5: Defensive adjustment
        Kills: usage for primary options (double teams, zone, switching)
        
        Smart defenses take away the #1 option.
        """
        stats = player_ctx.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_ctx.get('ppg_L5') or
                  player_ctx.get('points_L5') or 0)
        
        opp_def_rating = match_ctx.get('opp_def_rating', 110) or 110
        
        # Strong defense + high-usage player = defensive attention
        is_primary = ppg_L5 > 22
        is_strong_def = opp_def_rating < 108
        
        if is_primary and is_strong_def:
            if market == 'points':
                return True, f"Primary scorer vs elite D (DEF RTG {opp_def_rating:.0f}) → doubles/traps"
            if market == 'assists':
                return True, f"Elite D traps ball handler → tough passing windows"
        
        # Zone defense kills assists for certain players
        if market == 'assists' and is_strong_def:
            return True, f"Strong D ({opp_def_rating:.0f}) → zone/switching kills assist flow"
        
        # Switching kills rebounds for perimeter players
        if market == 'rebounds' and ppg_L5 > 15:
            # If the player is primarily a perimeter player getting rebounds
            reb_L5 = player_ctx.get('reb_L5', player_ctx.get('rebounds_L5', 5))
            if reb_L5 < 6 and is_strong_def:
                return True, f"Perimeter player + switching D → contested boards"
        
        return False, "No significant defensive scheme threat"
    
    # ==========================================
    # ROLE ANALYSIS
    # ==========================================
    
    def _assess_replaceability(self, player_ctx: Dict,
                               market_type: str) -> str:
        """
        Can this stat disappear without anyone noticing?
        
        High replaceability = stat is optional for the team.
        Low replaceability = team NEEDS this player to produce.
        """
        role = player_ctx.get('player_role',
                             player_ctx.get('role', 'role_player'))
        
        base_fragility = self.ROLE_FRAGILITY.get(role, 0.60)
        
        # Adjust for market type
        if market_type == 'assists' and role in ('floor_general', 'primary_ball_handler'):
            base_fragility *= 0.5  # Point guards' assists are less fragile
        
        if market_type == 'rebounds' and role in ('rim_runner', 'role_player'):
            base_fragility *= 0.7  # Bigs' rebounds are somewhat protected
        
        # Classify
        if base_fragility >= 0.75:
            return 'very_high'
        elif base_fragility >= 0.55:
            return 'high'
        elif base_fragility >= 0.35:
            return 'medium'
        else:
            return 'low'
    
    def _is_ceiling_needed(self, player_ctx: Dict,
                           match_ctx: Dict) -> bool:
        """
        Edge Question Q4: Is the ceiling NEEDED or OPTIONAL?
        
        Needed → team loses without this player going off
        Optional → team can win quietly
        
        Optional ceiling = elite UNDER signal
        """
        stats = player_ctx.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_ctx.get('ppg_L5') or
                  player_ctx.get('points_L5') or 0)
        
        spread = match_ctx.get('team_spread', match_ctx.get('spread', 0)) or 0
        
        # If team is favored, ceiling is optional
        # (team can win without star going nuclear)
        if spread < -5:
            return False  # Heavy favorite → ceiling optional
        
        # If team is underdog, they NEED the ceiling
        if spread > 3:
            return True  # Underdog → needs star to go off
        
        # Close game → ceiling matters but isn't essential
        # Use usage to determine: high-usage players are more "needed"
        if ppg_L5 > 25:
            return True  # Star in close game → ceiling likely needed
        
        return False  # Most scenarios → ceiling is optional
