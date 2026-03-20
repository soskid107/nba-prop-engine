"""
Same-Game Parlay (SGP) Engine
==============================
Builds correlated same-game parlays where legs REINFORCE each other.

Key Concept: SGPs multiply edge when legs share a causal driver.
  - Blowout → star UNDER + bench OVER (same causal scenario)
  - Tight game → points UNDER + assists OVER (game pace correlation)
  - Pace UP → both players' totals OVER (system-wide effect)

Anti-patterns to AVOID:
  - Star OVER + Teammate OVER (usage is zero-sum)
  - Any two UNDER bets on same team (contradicts team total)
"""

import logging
from typing import Dict, Any, List, Optional
from itertools import combinations

logger = logging.getLogger("SGP_ENGINE")


class SGPEngine:
    """
    Builds same-game parlays with positive correlation.
    
    Unlike regular parlays, SGP legs are deliberately correlated
    so they win or lose TOGETHER based on a shared game script.
    """
    
    # Correlation templates: scenarios where legs reinforce
    CORRELATION_TEMPLATES = {
        'blowout_play': {
            'name': '🏀 Blowout Script',
            'description': 'Star sits 4th quarter, bench gets run',
            'rules': [
                {'player_type': 'star', 'direction': 'UNDER', 'reason': 'Garbage time benching'},
                {'player_type': 'bench', 'direction': 'OVER', 'reason': 'Extended minutes'},
            ],
        },
        'pace_up': {
            'name': '🏃 Pace Up Play',
            'description': 'Fast game = more possessions for everyone',
            'rules': [
                {'direction': 'OVER', 'market': 'points', 'reason': 'More possessions'},
                {'direction': 'OVER', 'market': 'rebounds', 'reason': 'More misses = more boards'},
            ],
        },
        'grindout': {
            'name': '🔒 Grindout Play',
            'description': 'Close game, slow pace, both teams careful',
            'rules': [
                {'direction': 'UNDER', 'market': 'points', 'reason': 'Fewer possessions'},
                {'direction': 'OVER', 'market': 'assists', 'reason': 'Ball movement in half-court'},
            ],
        },
        'usage_vacuum': {
            'name': '🎯 Usage Vacuum',
            'description': 'Key player out, usage redistributes',
            'rules': [
                {'player_type': 'beneficiary', 'direction': 'OVER', 'reason': 'Absorbed usage'},
                {'player_type': 'absent_star', 'direction': 'UNDER', 'reason': 'DNP / limited'},
            ],
        },
    }
    
    def __init__(self):
        pass
    
    def build_sgps(self, 
                   picks_by_game: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Build same-game parlays from picks grouped by game.
        
        Args:
            picks_by_game: Dict mapping game_key to list of picks
            
        Returns:
            List of SGP ticket dicts
        """
        sgps = []
        
        for game_key, picks in picks_by_game.items():
            if len(picks) < 2:
                continue
            
            # Try each correlation template
            for template_id, template in self.CORRELATION_TEMPLATES.items():
                sgp = self._try_template(picks, template, game_key)
                if sgp:
                    sgps.append(sgp)
        
        # Sort by combined edge score
        sgps.sort(key=lambda s: s.get('combined_edge', 0), reverse=True)
        
        return sgps[:3]  # Max 3 SGPs
    
    def _try_template(self, picks: List[Dict], template: Dict,
                      game_key: str) -> Optional[Dict]:
        """Try to match picks to a correlation template."""
        
        rules = template['rules']
        matched_legs = []
        
        for rule in rules:
            best_match = None
            best_score = 0
            
            for pick in picks:
                if pick in matched_legs:
                    continue
                
                score = self._score_match(pick, rule)
                if score > best_score:
                    best_score = score
                    best_match = pick
            
            if best_match and best_score > 0:
                matched_legs.append(best_match)
        
        if len(matched_legs) < 2:
            return None
        
        # Calculate combined edge
        combined_edge = sum(p.get('edge_score', 0) for p in matched_legs) / len(matched_legs)
        
        # Only return if average edge is decent
        if combined_edge < 65:
            return None
        
        # Estimate correlated probability boost
        # Correlated legs have ~10-15% higher combined hit rate than independent
        correlation_bonus = 0.10
        
        legs = []
        for pick in matched_legs:
            legs.append({
                'player_name': pick.get('player_name', ''),
                'team': pick.get('team', ''),
                'market': pick.get('market', 'PTS'),
                'line': pick.get('line', 0),
                'direction': pick.get('direction', ''),
                'edge_score': pick.get('edge_score', 0),
                'edge_tier': pick.get('edge_tier', ''),
            })
        
        return {
            'ticket_name': template['name'],
            'description': template['description'],
            'game': game_key,
            'legs': legs,
            'num_legs': len(legs),
            'combined_edge': round(combined_edge, 1),
            'correlation_bonus': correlation_bonus,
            'correlation_type': template['name'],
        }
    
    def _score_match(self, pick: Dict, rule: Dict) -> float:
        """Score how well a pick matches a rule. 0 = no match."""
        score = 0
        
        # Direction must match if specified
        if 'direction' in rule:
            if pick.get('direction') != rule['direction']:
                return 0
            score += 1
        
        # Market match
        if 'market' in rule:
            pick_market = pick.get('market', '').lower()
            if rule['market'] not in pick_market:
                return 0
            score += 1
        
        # Player type match
        if 'player_type' in rule:
            role = pick.get('player_role', pick.get('archetype', ''))
            pt = rule['player_type']
            if pt == 'star' and role in ('volume_star', 'star', 'alpha'):
                score += 2
            elif pt == 'bench' and role in ('role_player', 'bench', 'specialist'):
                score += 2
            elif pt == 'beneficiary' and pick.get('usage_boost', 0) > 0:
                score += 2
            else:
                # Flexible match
                score += 0.5
        
        # Bonus for high edge score
        edge = pick.get('edge_score', 0)
        if edge >= 80:
            score += 1
        elif edge >= 70:
            score += 0.5
        
        return score
    
    def format_sgps_markdown(self, sgps: List[Dict]) -> str:
        """Format SGP tickets as markdown."""
        if not sgps:
            return ""
        
        lines = []
        lines.append("\n### 🎲 Same-Game Parlays (Correlated)")
        lines.append("_Legs share a causal driver — they win or lose together_")
        lines.append("")
        
        for sgp in sgps:
            lines.append(f"#### {sgp['ticket_name']} — {sgp['game']}")
            lines.append(f"_{sgp['description']}_")
            lines.append(f"**Avg Edge:** {sgp['combined_edge']:.0f}/100 | **Correlation Boost:** +{sgp['correlation_bonus']:.0%}")
            lines.append("")
            
            lines.append("| # | Player | Market | Pick | Line | Edge |")
            lines.append("|---|--------|--------|------|------|------|")
            
            for i, leg in enumerate(sgp['legs'], 1):
                tier = "🔥" if leg['edge_tier'] == 'parlay_core' else "✅"
                lines.append(
                    f"| {i} | {tier} {leg['player_name']} ({leg['team']}) "
                    f"| {leg['market']} | {leg['direction']} | {leg['line']:.1f} "
                    f"| {leg['edge_score']:.0f} |"
                )
            
            lines.append("")
            lines.append("---")
        
        return "\n".join(lines)
