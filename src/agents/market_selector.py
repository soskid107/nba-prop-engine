"""
Agent 9: Market Selector (The "Brain")
======================================
Role: Analyze Player & Match Context to select the SINGLE BEST market to predict.
Input: Player Stats, Opponent DVP, Form, Injuries.
Output: Selected Market (e.g. 'assists') + Contextual Reasoning.

Logic:
1. DVP Check: Does opponent allow excessive production in a specific stat?
2. Form Check: Is player trending up in a specific stat (L5 > Season)?
3. Role/Injury Check: Does missing teammate create a vacuum for a specific stat?
4. Score & Select: Rank markets (Pts, Ast, Reb) and pick the winner.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class MarketSelection:
    market_type: str  # 'points', 'assists', 'rebounds'
    confidence: float # 0-100 score indicating how strong this "angle" is
    reasoning: List[str] # Why this market? (e.g. "Opponent Rank #28 vs PG")
    context_flags: List[str] # e.g. ["DVP_PLUS", "FORM_HOT"]
    ranked_candidates: List[Dict[str, Any]]
    score_gap_to_next: float

class MarketSelectorAgent:
    def __init__(self):
        # DVP Thresholds (Opponent Rank 1-30, where 30 is worst defense/best for over)
        self.DVP_GOOD_MATCHUP = 20 # Rank 20+ (Bottom 10 defense)
        self.DVP_ELITE_MATCHUP = 25 # Rank 25+ (Bottom 5 defense)
        self.DVP_BAD_MATCHUP = 10  # Rank 1-10 (Top 10 defense)

    def select_best_market(self, 
                          player_name: str,
                          player_context: Dict[str, Any],
                          match_context: Dict[str, Any],
                          available_lines: Dict[str, float]) -> Optional[MarketSelection]:
        """
        Analyze context and return the best market to target.
        If no market is compelling, returns None (skip player).
        """
        scores = {}
        reasoning_map = {}

        # 1. Analyze Core Markets
        # [PHASE 13] Expanded to include Threes, Blocks, Steals, FGM
        markets_to_analyze = [
            'points', 'assists', 'rebounds', 
            'threes', 'blocks', 'steals', 
            'field_goals'
        ]
        
        for market in markets_to_analyze:
            # Map 'field_goals' to 'field_goals_made' if needed, or rely on key matching
            # Odds API key is 'player_field_goals', so we likely get 'field_goals' as key in available_lines
            if market not in available_lines:
                continue
                
            score, reasons = self._score_market_potential(
                market, player_context, match_context, available_lines[market]
            )
            scores[market] = score
            reasoning_map[market] = reasons

        # 2. Select Winner
        if not scores:
            return None

        # Sort by score descending
        sorted_markets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_market, best_score = sorted_markets[0]
        
        # Threshold Check (Don't force a bet if everything looks bad)
        # MIN_SCORE = 50 
        # if best_score < MIN_SCORE:
        #    return None 

        ranked_candidates = []
        for rank, (market, score) in enumerate(sorted_markets, start=1):
            ranked_candidates.append({
                'rank': rank,
                'market_type': market,
                'confidence': score,
                'line': available_lines.get(market),
                'reasoning': reasoning_map.get(market, []),
            })

        score_gap = 999.0
        if len(sorted_markets) > 1:
            score_gap = best_score - sorted_markets[1][1]

        return MarketSelection(
            market_type=best_market,
            confidence=best_score,
            reasoning=reasoning_map[best_market],
            context_flags=[], # Todo: Populate flags
            ranked_candidates=ranked_candidates,
            score_gap_to_next=score_gap,
        )

    def _score_market_potential(self, 
                              market: str, 
                              p_ctx: Dict, 
                              m_ctx: Dict,
                              line: float) -> Tuple[float, List[str]]:
        """
        Score a specific market based on logic.
        Uses: Specific DVP, Teammate Boosts, Hit Rates.
        """
        score = 50.0 # Base score
        reasons = []
        
        # --- A. Specific Matchup (DVP) ---
        # match_context['dvp_stats'] is {stat: {pos: mult}}
        dvp_stats = m_ctx.get('dvp_stats', {})
        
        market_key = market
        if market == 'threes': market_key = 'threes'
        # [PHASE 13] Map other keys if needed (blocks, steals usually match)
        
        # Get Position Group
        position = p_ctx.get('position') or 'G'
        pos_group = 'G'
        if 'C' in position: pos_group = 'C'
        elif 'F' in position: pos_group = 'F'
        
        # Access: dvp_stats[market][pos]
        # Handle if market or pos missing
        stat_data = dvp_stats.get(market_key, {})
        if isinstance(stat_data, dict):
            multiplier = stat_data.get(pos_group, 1.0)
        else:
            multiplier = 1.0
            
        # Refined DVP Scoring
        if multiplier >= 1.15: 
            score += 25
            reasons.append(f"Elite Matchup: Opponent allows +{(multiplier-1)*100:.0f}% {market} to {pos_group}s")
        elif multiplier >= 1.05:
            score += 15
            reasons.append(f"Good Matchup: Opponent allows +{(multiplier-1)*100:.0f}% {market} to {pos_group}s")
        elif multiplier <= 0.85:
            score -= 20
            reasons.append(f"Bad Matchup: Opponent allows {(multiplier-1)*100:.0f}% {market} to {pos_group}s")

        # --- B. Teammate Correlations (Phase 11) ---
        teammate_impact = m_ctx.get('teammate_impact', {})
        if teammate_impact and teammate_impact.get('has_significant_impact'):
            # Check specific stat boost
            boost_key = f"total_{market}_boost" # e.g. total_assists_boost
            
            # Special case for Points (usage boost proxy)
            if market == 'points':
                boost = teammate_impact.get('expected_points_boost', 0)
                if boost > 2.0:
                    score += 20
                    reasons.append(f"Usage Bump: Expecting +{boost} pts due to missing teammates")
            
            # Other markets
            elif market in ['assists', 'rebounds', 'threes', 'blocks', 'steals', 'field_goals']:
                boost = teammate_impact.get(boost_key, 0)
                if boost >= 1.5: # Significant correlation
                    score += 25
                    reasons.append(f"Role Change: Expecting +{boost} {market} due to injuries")
                elif boost >= 0.8:
                    score += 10
                    reasons.append(f"Role Change: Expecting +{boost} {market}")

        # --- C. Recent Form & Hit Rate (Phase 11) ---
        stats = p_ctx
        key_map = {
            'points': 'points', 
            'assists': 'assists', 
            'rebounds': 'rebounds',
            'threes': 'fg3m' # fallback
        }
        stat_key = key_map.get(market, market)
        
        # 1. L5 vs L15 surge
        l5 = stats.get(f'{stat_key}_L5') # using alias from gatherer
        l15 = stats.get(f'{stat_key}_L15')
        
        if l5 is None and market == 'points': l5 = stats.get('points_L5') # fallback
        if l15 is None and market == 'points': l15 = stats.get('points_L15')

        if l5 is not None and l15 is not None and l15 > 0:
            diff_pct = (l5 - l15) / l15
            if diff_pct > 0.25:
                score += 15
                reasons.append(f"Hot Streak: L5 run is +{diff_pct*100:.0f}% vs baseline")
            elif diff_pct < -0.20:
                score -= 10
        
        # 2. Hit Rate (Consistency)
        recent_logs = p_ctx.get('recent_logs', [])
        if recent_logs and line > 0:
            hits = 0
            valid_games = 0
            for g in recent_logs:
                val = g.get(stat_key, 0)
                # For threes, key might be different in logs
                if market == 'threes': val = g.get('fg3m', 0)
                
                if val > line: hits += 1
                valid_games += 1
            
            if valid_games > 0:
                hit_rate = hits / valid_games
                if hit_rate >= 0.70: # 7/10
                    score += 15
                    reasons.append(f"Consistent: Covered line ({line}) in {hits}/{valid_games} L10")
                elif hit_rate <= 0.30: # 3/10
                    score -= 20
                    reasons.append(f"Inconsistent: Covered line in only {hits}/{valid_games} L10")

        # --- D. Line Context ---
        if l15 and l15 > 0:
            line_diff = (line - l15) / l15
            if line_diff < -0.15: # Line is discounted
                score += 10
                reasons.append(f"Value Line: {line} is lower than L15 avg {l15:.1f}")

        return score, reasons
