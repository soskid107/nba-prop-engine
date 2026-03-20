"""
Parlay Builder: Ready-Made Ticket Generator
=============================================
Builds optimal parlay tickets from edge-scored picks.

Parlay Tiers:
  1. PARLAY OF THE DAY (2-3 legs, all parlay_core, max edge)
  2. VALUE PARLAY (3-4 legs, mix of parlay_core + playable)
  3. LONGSHOT PARLAY (4-6 legs, high-kill-count picks)

Anti-Correlation Rules:
  - Never double up on same-game UNDERs (blowout correlation)
  - Never parlay a player OVER with their teammate's UNDER
  - Prefer cross-game legs for independence
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations

logger = logging.getLogger("PARLAY_BUILDER")


class ParlayBuilder:
    """
    Builds ready-made parlay tickets from edge-scored predictions.
    
    Core Principle:
    Each leg must independently have edge. Parlays multiply
    edge, not replace it. No filler legs — ever.
    """
    
    # Parlay ticket templates
    TICKET_CONFIGS = {
        'parlay_of_day': {
            'name': '🔥 PARLAY OF THE DAY',
            'min_legs': 2,
            'max_legs': 3,
            'min_tier': 'parlay_core',  # Only parlay_core picks
            'min_edge_score': 75,  # Match EdgeScorer THRESHOLD_PARLAY (75)
            'description': 'Highest conviction — all legs are parlay core picks'
        },
        'value_parlay': {
            'name': '💰 VALUE PARLAY',
            'min_legs': 2,
            'max_legs': 4,
            'min_tier': 'playable',
            'min_edge_score': 65,  # Match EdgeScorer THRESHOLD_PLAYABLE (65)
            'description': 'Strong value — mix of core + playable picks'
        },
        'longshot': {
            'name': '🎯 LONGSHOT',
            'min_legs': 4,
            'max_legs': 6,
            'min_tier': 'playable',
            'min_edge_score': 60,
            'description': 'High kill-count picks for big payouts'
        }
    }
    
    def __init__(self):
        pass
    
    def build_tickets(self, picks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build parlay tickets from edge-scored picks.
        
        Args:
            picks: List of dicts with keys:
                - player_name, team, opponent, market, line
                - direction, edge_score, edge_tier
                - edge_explanation, kill_count
                
        Returns:
            List of parlay ticket dicts
        """
        if not picks:
            return []
        
        # Sort picks by edge score descending
        sorted_picks = sorted(
            picks, key=lambda p: p.get('edge_score', 0), reverse=True
        )
        
        tickets = []
        
        # 1. PARLAY OF THE DAY
        potd = self._build_ticket(
            sorted_picks, self.TICKET_CONFIGS['parlay_of_day']
        )
        if potd:
            tickets.append(potd)
        
        # 2. VALUE PARLAY (exclude legs already used in POTD)
        potd_players = set()
        if potd:
            potd_players = {leg['player_name'] for leg in potd['legs']}
        
        remaining = [p for p in sorted_picks if p['player_name'] not in potd_players]
        value = self._build_ticket(
            remaining, self.TICKET_CONFIGS['value_parlay']
        )
        if value:
            tickets.append(value)
        
        # 3. LONGSHOT (can reuse legs from value but not POTD)
        longshot_pool = [p for p in sorted_picks if p['player_name'] not in potd_players]
        longshot = self._build_ticket(
            longshot_pool, self.TICKET_CONFIGS['longshot']
        )
        if longshot and len(longshot['legs']) >= 4:
            tickets.append(longshot)
        
        return tickets
    
    def _build_ticket(self, picks: List[Dict],
                      config: Dict) -> Optional[Dict[str, Any]]:
        """Build a single parlay ticket from available picks."""
        
        min_tier = config['min_tier']
        min_score = config['min_edge_score']
        
        # Filter eligible picks
        tier_order = {'parlay_core': 2, 'playable': 1, 'reject': 0}
        min_tier_val = tier_order.get(min_tier, 0)
        
        eligible = [
            p for p in picks
            if tier_order.get(p.get('edge_tier', 'reject'), 0) >= min_tier_val
            and p.get('edge_score', 0) >= min_score
        ]
        
        if len(eligible) < config['min_legs']:
            return None
        
        # Find the best combination that passes anti-correlation checks
        max_legs = min(config['max_legs'], len(eligible))
        min_legs = config['min_legs']
        
        best_combo = None
        best_score = 0
        
        for size in range(max_legs, min_legs - 1, -1):
            for combo in combinations(eligible, size):
                if self._check_correlation(list(combo)):
                    combo_score = sum(p.get('edge_score', 0) for p in combo) / len(combo)
                    if combo_score > best_score:
                        best_score = combo_score
                        best_combo = list(combo)
            if best_combo:
                break  # Found a valid combo at this size
        
        if not best_combo:
            return None
        
        # Calculate combined odds estimate
        legs = []
        combined_implied_prob = 1.0
        
        for pick in best_combo:
            leg = {
                'player_name': pick.get('player_name', 'Unknown'),
                'team': pick.get('team', ''),
                'opponent': pick.get('opponent', ''),
                'market': pick.get('market', 'PTS').upper(),
                'line': pick.get('line', 0),
                'direction': pick.get('direction', ''),
                'edge_score': pick.get('edge_score', 0),
                'edge_tier': pick.get('edge_tier', ''),
                'explanation': pick.get('edge_explanation', ''),
                'kill_count': pick.get('kill_count', 0),
            }
            legs.append(leg)
            
            # Rough implied probability (standard -110 lines)
            combined_implied_prob *= 0.52  # Each leg ~52% breakeven
        
        # Calculate payout multiplier (approximate)
        if combined_implied_prob > 0:
            payout_multiplier = 1.0 / combined_implied_prob
        else:
            payout_multiplier = 1.0
        
        return {
            'ticket_name': config['name'],
            'description': config['description'],
            'legs': legs,
            'num_legs': len(legs),
            'avg_edge_score': best_score,
            'estimated_payout': f'{payout_multiplier:.1f}x',
            'combined_win_prob': combined_implied_prob,
        }
    
    def _check_correlation(self, picks: List[Dict]) -> bool:
        """
        Anti-correlation checks. Returns True if the combination is valid.
        
        Rules:
        1. No two UNDER picks from the same game
        2. No player OVER + teammate UNDER in same parlay
        3. Max 2 picks from any single game
        """
        # Group by game
        game_picks = {}
        for p in picks:
            game_key = frozenset([p.get('team', ''), p.get('opponent', '')])
            if game_key not in game_picks:
                game_picks[game_key] = []
            game_picks[game_key].append(p)
        
        for game_key, game_legs in game_picks.items():
            # Rule 3: Max 2 picks from same game
            if len(game_legs) > 2:
                return False
            
            if len(game_legs) == 2:
                # Rule 1: No two UNDERs from same game
                dirs = [p.get('direction', '') for p in game_legs]
                if dirs[0] == 'UNDER' and dirs[1] == 'UNDER':
                    return False
                
                # Rule 2: No OVER + teammate UNDER (conflicting game script)
                teams = [p.get('team', '') for p in game_legs]
                if teams[0] == teams[1]:
                    # Same team
                    if 'OVER' in dirs and 'UNDER' in dirs:
                        return False
                        
                # Rule 3: No defensive stats (Blk/Stl) combined with Unders on same team
                # (Defensive events require opponent possessions, Unders want fewer)
                # This is a heuristic - keeping it simple for now.
        
        return True
    
    def format_tickets_markdown(self, tickets: List[Dict]) -> str:
        """Format parlay tickets as markdown for the report."""
        if not tickets:
            return "\n### 🎫 Parlay Tickets\n_No qualifying parlays today (need ≥2 picks with edge score ≥70)_\n"
        
        lines = []
        lines.append("\n### 🎫 Ready-Made Parlay Tickets")
        lines.append("")
        
        for ticket in tickets:
            lines.append(f"#### {ticket['ticket_name']}")
            lines.append(f"_{ticket['description']}_")
            lines.append(f"**Legs:** {ticket['num_legs']} | **Avg Edge Score:** {ticket['avg_edge_score']:.0f}/100 | **Est. Payout:** {ticket['estimated_payout']}")
            lines.append("")
            
            lines.append("| # | Player | Market | Pick | Line | Edge Score |")
            lines.append("|---|--------|--------|------|------|------------|")
            
            for i, leg in enumerate(ticket['legs'], 1):
                tier_icon = "🔥" if leg['edge_tier'] == 'parlay_core' else "✅"
                market_str = leg['market']
                pick_str = leg['direction']
                lines.append(
                    f"| {i} | {tier_icon} {leg['player_name']} ({leg['team']}) "
                    f"| {market_str} | {pick_str} | {leg['line']:.1f} "
                    f"| {leg['edge_score']:.0f} |"
                )
            
            lines.append("")
            
            # Add brief reasoning for each leg
            lines.append("**Why each leg:**")
            for i, leg in enumerate(ticket['legs'], 1):
                explanation = leg.get('explanation', 'Edge detected')
                # Truncate very long explanations
                if len(explanation) > 150:
                    explanation = explanation[:147] + "..."
                lines.append(f"{i}. {leg['player_name']}: {explanation}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
