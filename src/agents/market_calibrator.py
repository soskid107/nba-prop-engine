"""
Agent 4: Market Calibrator & Decision Engine
=============================================
Role: Convert model output to betting-usable format, anchor to market lines

Core Principles (Hinton - Representation Learning):
- Market lines are informed priors, not noise
- Model vs Market = Edge Detection
- Never ignore what the market is telling you

Decision Framework:
1. Compare model distribution to sportsbook line
2. Calculate true probability of over/under
3. Apply Kelly criterion for position sizing
4. Generate structured reasoning for transparency
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BettingDecision:
    """Structured betting recommendation"""
    player_name: str
    line: float  # Sportsbook line
    model_mean: float
    model_std: float
    
    # Probabilities
    prob_over: float
    prob_under: float
    
    # Edge
    edge_over: float  # vs implied -110 odds (52.4%)
    edge_under: float
    
    # Recommendation
    direction: str  # 'OVER', 'UNDER', 'NO_BET'
    confidence: str  # 'high', 'medium', 'low', 'no_edge'
    kelly_fraction: float  # Suggested bet size
    
    # Context (Phase 7)
    edge_source: str  # e.g., 'MINUTES_MISPRICING', 'USAGE_SPIKE'
    
    # Reasoning
    reasoning: List[str]
    blocker_reason: str = ""


class MarketCalibratorAgent:
    """
    Agent 4: Market Calibrator
    
    Converts model predictions into actionable betting decisions.
    Anchors to sportsbook lines as informed priors.
    
    Key Insight: The market is smart. If our model disagrees significantly,
    we need strong conviction in WHY we're right.
    """
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        
        # Standard -110 odds implies 52.38% breakeven
        self.BREAKEVEN_PROB = 0.5238
        
        # Minimum edge thresholds for betting
        self.MIN_EDGE_HIGH_CONF = 0.06  # 6% edge for high confidence
        self.MIN_EDGE_MED_CONF = 0.08   # 8% edge for medium (wider uncertainty)
        self.MIN_EDGE_LOW_CONF = 0.10   # 10% edge for low confidence
        
        # Maximum Kelly fraction (risk management)
        self.MAX_KELLY = 0.05  # Never bet more than 5% of bankroll
        
        # Market calibration factors
        self.MARKET_ANCHOR_WEIGHT = 0.15  # Pull model toward market slightly
    
    
    def calibrate(self,
                  audited_prediction: Dict[str, Any],
                  market_line: float,
                  player_name: str,
                  player_context: Dict[str, Any],
                  match_context: Dict[str, Any],
                  market_odds: float = -110) -> BettingDecision:
        """
        Main calibration function
        
        Args:
            audited_prediction: From Agent 3
            market_line: Sportsbook points line
            player_name: For display
            player_context: From Agent 1
            match_context: From Agent 1
            market_odds: American odds (e.g. -110, +105)
            
        Returns:
            BettingDecision with full recommendation
        """
        # Calculate Breakeven Probability from Odds
        breakeven_prob = self._calculate_breakeven_prob(market_odds)
        
        # Get model distribution parameters
        model_mean = audited_prediction.get('mean', 15)
        model_std = audited_prediction.get('std', 5)
        confidence = audited_prediction.get('confidence', 'medium')
        market_type = (
            audited_prediction.get('market_type')
            or match_context.get('selection_reasoning', {}).get('market')
            or 'points'
        )
        
        # [R6] Bayesian market anchoring (conjugate normal-normal posterior)
        regime_flags = match_context.get('regime_flags', [])

        calibrated_mean = self._apply_market_anchor(
            model_mean,
            market_line,
            confidence,
            model_std=model_std,
            regime_flags=regime_flags,
            prediction_health=audited_prediction.get('prediction_health', {}),
        )
        
        # [R9] Graduated std inflation based on defense level
        calibrated_std = model_std
        if 'SYSTEM_SEVERELY_OVERCONFIDENT' in regime_flags:
            calibrated_std = model_std * 1.40  # Widen intervals by 40%
        elif 'SYSTEM_OVERCONFIDENT' in regime_flags:
            calibrated_std = model_std * 1.30  # Widen intervals by 30%
        elif 'SYSTEM_RECOVERING' in regime_flags:
            calibrated_std = model_std * 1.15  # Mild defense — graduating
        elif 'SYSTEM_SLIGHTLY_OVERCONFIDENT' in regime_flags:
            calibrated_std = model_std * 1.10  # Slight widening
        elif 'SYSTEM_UNDERCONFIDENT' in regime_flags:
            calibrated_std = model_std * 0.90  # Tighten slightly

        # Market-specific calibration: recent audits show points is still the
        # least trustworthy market, while assists/rebounds are materially tighter.
        if market_type == 'points':
            calibrated_std *= 1.10
        elif market_type == 'assists':
            calibrated_std *= 1.02
        elif market_type == 'rebounds':
            calibrated_std *= 1.04

        if 'SYSTEM_STRICT_INJURY_CONTEXT' in regime_flags:
            injury_context_present = (
                'team_injuries' in player_context or
                'team_injuries' in match_context
            )
            if not injury_context_present:
                return BettingDecision(
                    player_name=player_name,
                    line=market_line,
                    model_mean=calibrated_mean,
                    model_std=calibrated_std,
                    prob_over=0.5,
                    prob_under=0.5,
                    edge_over=0.0,
                    edge_under=0.0,
                    direction='NO_BET',
                    confidence='no_edge',
                    kelly_fraction=0.0,
                    edge_source='MISSING_INJURY_CONTEXT',
                    reasoning=['Policy block: missing injury context after recent miss-pattern review.'],
                    blocker_reason='missing injury context after recent miss-pattern review'
                )
        
        # Calculate probabilities using normal distribution (with Shrinkage)
        prob_over, prob_under = self._calculate_probabilities(
            calibrated_mean, calibrated_std, market_line, confidence
        )
        
        # Calculate edges (TRUE EDGE vs dynamic breakeven)
        edge_over = prob_over - breakeven_prob
        edge_under = prob_under - breakeven_prob
        clv_notes: List[str] = []
        edge_over, edge_under, clv_notes = self._apply_clv_pressure(
            edge_over, edge_under, match_context
        )
        
        # Determine direction and bet sizing (with Regime Defense)
        direction, bet_confidence, blocker_reason = self._determine_direction(
            edge_over,
            edge_under,
            confidence,
            calibrated_std,
            regime_flags,
            market_type=market_type,
            edge_analysis=match_context.get('edge_analysis', {}),
            prediction_health=audited_prediction.get('prediction_health', {}),
            market_consensus=match_context.get('market_consensus', {}),
            player_consensus=match_context.get('player_consensus', {}),
        )
        
        # Calculate Kelly with specific odds
        kelly = self._calculate_kelly(edge_over, edge_under, direction, bet_confidence, market_odds)
        
        # Determine Edge Source (Phase 7)
        edge_match_context = dict(match_context or {})
        edge_match_context['prediction_health'] = audited_prediction.get('prediction_health', {})
        edge_source = self._determine_edge_source(
            direction, model_mean, market_line, prob_over, prob_under,
            calibrated_std, player_context, edge_match_context
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            model_mean, calibrated_mean, market_line,
            prob_over, prob_under, edge_over, edge_under,
            player_context, match_context, confidence
        )
        reasoning.extend(clv_notes)
        if blocker_reason:
            reasoning.append(f"Final gate: {blocker_reason}")
        
        return BettingDecision(
            player_name=player_name,
            line=market_line,
            model_mean=calibrated_mean,
            model_std=calibrated_std,
            prob_over=prob_over,
            prob_under=prob_under,
            edge_over=edge_over,
            edge_under=edge_under,
            direction=direction,
            confidence=bet_confidence,
            kelly_fraction=kelly,
            edge_source=edge_source,
            reasoning=reasoning,
            blocker_reason=blocker_reason
        )
        
    def _calculate_breakeven_prob(self, american_odds: float) -> float:
        """
        Calculate implied probability from American odds.
        Negative: -110 -> 110/(110+100) = 0.5238
        Positive: +120 -> 100/(120+100) = 0.4545
        """
        # Handle None or invalid odds - default to standard -110
        if american_odds is None:
            return self.BREAKEVEN_PROB  # 0.5238
            
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    def _calculate_decimal_odds(self, american_odds: float) -> float:
        """Convert American to Decimal odds for Kelly formula."""
        # Handle None - default to -110 (decimal 1.909)
        if american_odds is None:
            american_odds = -110
            
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    
    def _determine_edge_source(self, direction: str,
                             model_mean: float,
                             market_line: float,
                             prob_over: float,
                             prob_under: float,
                             model_std: float,
                             player_context: Dict,
                             match_context: Dict) -> str:
        """
        Identify the PRIMARY driver of the edge (Phase 7).
        Why do we think we win?
        """
        if direction == 'NO_BET':
            return 'NONE'
        
        signals = player_context.get('inferred_signals', {})
        stats = player_context.get('stats', {})
        lineup_context = match_context.get('lineup_context') or player_context.get('lineup_context') or {}
        usage_impact = match_context.get('usage_impact') or {}
        movement = match_context.get('line_movement', {}) or {}
        prediction_health = match_context.get('prediction_health') or {}
        disagreement = prediction_health.get('reference_disagreement')
        disagreement = float(disagreement) if disagreement is not None else None
        usage_delta = float(
            lineup_context.get('usage_proxy_delta')
            or usage_impact.get('usage_delta')
            or 0.0
        )
        minutes_delta = float(
            lineup_context.get('role_minutes_delta')
            or lineup_context.get('minutes_delta')
            or usage_impact.get('minutes_delta')
            or 0.0
        )
        lineup_volatility = float(lineup_context.get('volatility_score') or 0.0)
        market_gap = abs(model_mean - market_line)
        
        # 1. Variance Mispricing (High Volatility + Under)
        # If we take an under on a high variance player
        if direction == 'UNDER' and model_std > 8.0:
            return 'VOLATILITY_UNDER_EDGE'
             
        # 2. Lineup- and role-driven edges
        if abs(minutes_delta) >= 2.5 or lineup_volatility >= 0.60:
            return 'LINEUP_SHIFT_EDGE'
        if stats.get('minutes_trend', 'stable') != 'stable':
             return 'MINUTES_ROLE_EDGE'
              
        # 3. Usage Redistribution
        if signals.get('usage_spike') or abs(usage_delta) >= 0.08:
             return 'USAGE_REDISTRIBUTION_EDGE'
        
        # 4. Role Misclassification
        if signals.get('role_change'):
             return 'ROLE_CHANGE_EDGE'

        # 5. Model disagreement is informative when references diverge materially.
        if disagreement is not None and disagreement >= 4.5:
            return 'MODEL_DISAGREEMENT_EDGE'
              
        # 6. Defensive Mismatch
        opp_def = match_context.get('opp_def_rating', 110)
        if direction == 'OVER' and opp_def > 115:
            return 'DEFENSIVE_MISMATCH'
        if direction == 'UNDER' and opp_def < 105:
             return 'DEFENSIVE_MISMATCH'

        # 7. Market movement disagreement
        sharp_direction = movement.get('sharp_direction')
        movement_score = float(movement.get('movement_score', 0) or 0)
        if sharp_direction and movement_score >= 25:
            if (direction == 'OVER' and sharp_direction == 'UNDER') or (
                direction == 'UNDER' and sharp_direction == 'OVER'
            ):
                return 'MARKET_ANCHOR_DISAGREEMENT'
              
        # 8. Blowout Misestimation
        blowout_prob = match_context.get('blowout_probability', 0)
        if direction == 'UNDER' and blowout_prob > 0.25:
             return 'BLOWOUT_MISESTIMATION'

        # 9. Large model-vs-market gap without a cleaner structural explanation.
        if market_gap >= max(2.5, 0.12 * max(abs(market_line), 1.0)):
            return 'UNATTRIBUTED_MODEL_EDGE'

        return 'LOW_SIGNAL_EDGE'

    def _apply_clv_pressure(self,
                            edge_over: float,
                            edge_under: float,
                            match_context: Dict[str, Any]) -> Tuple[float, float, List[str]]:
        """Penalize edges that are fighting strong market movement signals."""
        movement = match_context.get('line_movement', {}) or {}
        sharp_direction = movement.get('sharp_direction')
        movement_score = float(movement.get('movement_score', 0) or 0)
        total_move = float(movement.get('total_move', 0) or 0)
        notes: List[str] = []

        if movement_score < 20 or not sharp_direction:
            return edge_over, edge_under, notes

        pressure = min(0.03, 0.005 * (movement_score / 10.0))
        if sharp_direction == 'UNDER':
            edge_over -= pressure
            edge_under += pressure * 0.25
            notes.append(f"CLV pressure: market movement leans UNDER ({total_move:+.1f}).")
        elif sharp_direction == 'OVER':
            edge_under -= pressure
            edge_over += pressure * 0.25
            notes.append(f"CLV pressure: market movement leans OVER ({total_move:+.1f}).")

        return edge_over, edge_under, notes
    
    def _apply_market_anchor(self, model_mean: float, market_line: float,
                             confidence: str, model_std: float = None,
                             regime_flags: List[str] = None,
                             prediction_health: Dict[str, Any] = None) -> float:
        """
        [R6] Bayesian Calibration: Conjugate Normal-Normal Update.
        
        Prior: Market line (μ_prior = market_line, σ_prior from market width/confidence)
        Likelihood: Model prediction (μ_model, σ_model)  
        Posterior: Precision-weighted combination
        
        This replaces manual anchor weights with principled Bayesian math.
        The market is treated as an informative prior whose precision depends on
        how tight the market is (narrow market = high precision = hard to beat).
        """
        # Market precision: how confident is the market?
        # Standard vig line: market knows within ~3-4 points (σ ≈ 3.5)
        market_sigma = 3.5  # Conservative estimate of market uncertainty
        regime_flags = regime_flags or []
        prediction_health = prediction_health or {}
        health_score = float(prediction_health.get('health_score', 1.0) or 1.0)
        used_fallback = bool(prediction_health.get('used_fallback_model'))
        if 'SYSTEM_REDUCE_MARKET_ANCHOR' in regime_flags:
            market_sigma *= 1.75
        if used_fallback:
            market_sigma *= 0.80
        elif health_score >= 0.85:
            market_sigma *= 1.35
        elif health_score >= 0.70:
            market_sigma *= 1.15
        elif health_score < 0.55:
            market_sigma *= 0.90
        
        # Model precision: how confident are we?
        # Use model std directly if available, else estimate from confidence tier
        if model_std and model_std > 0:
            model_sigma = model_std
        else:
            sigma_by_confidence = {
                'high': 3.0,     # We're very sure → comparable to market
                'good': 4.0,
                'medium': 5.5,
                'low': 7.0,
                'very_low': 10.0, # Very uncertain → market dominates
            }
            model_sigma = sigma_by_confidence.get(confidence, 5.5)

        disagreement = abs(model_mean - market_line)
        if disagreement >= 6 and health_score >= 0.80 and not used_fallback:
            market_sigma *= 1.25
        elif disagreement <= 1.5:
            market_sigma *= 0.95
        
        # Conjugate Normal-Normal posterior
        # τ = 1/σ² (precision)
        tau_prior = 1.0 / (market_sigma ** 2)
        tau_likelihood = 1.0 / (model_sigma ** 2)
        tau_posterior = tau_prior + tau_likelihood
        
        # Posterior mean = precision-weighted average
        posterior_mean = (
            tau_prior * market_line + tau_likelihood * model_mean
        ) / tau_posterior
        
        return posterior_mean
    
    def _shrink_probability(self, raw_prob: float, confidence: str, 
                          sample_size: int = 0) -> float:
        """
        Phase 6: Probability Shrinkage
        "Systems that don't calibrate hallucinate certainty."
        
        Formula: 0.5 + (raw - 0.5) * weight
        """
        # Base weights by confidence tier
        weights = {
            'high': 0.90,     # Trust high confidence 90%
            'medium': 0.75,   # Trust medium 75%
            'low': 0.60,      # Trust low 60%
            'very_low': 0.40, # Trust very low 40% (heavy dampening)
            'no_edge': 0.0    # Should not be here, but 0 trust
        }
        
        weight = weights.get(confidence, 0.60)
        
        # Further penalize small samples (dashboard rule 1.2)
        if sample_size > 0 and sample_size < 10:
            weight *= 0.8  # 20% penalty for small sample
            
        # Apply shrinkage
        shrinked_prob = 0.5 + (raw_prob - 0.5) * weight
        
        return shrinked_prob

    def _calculate_probabilities(self, mean: float, std: float,
                                  line: float, confidence: str = 'medium') -> Tuple[float, float]:
        """
        Calculate P(over) and P(under) from normal distribution
        NOW WITH SHRINKAGE applied.
        """
        if std is None or std <= 0.01:
            if mean is None or line is None:
                return 0.5, 0.5
            raw_prob_over = 0.5 if abs(mean - line) < 0.1 else (0.60 if mean > line else 0.40)
            raw_prob_under = 1 - raw_prob_over
            prob_under = self._shrink_probability(raw_prob_under, confidence)
            prob_over = self._shrink_probability(raw_prob_over, confidence)
            total = prob_under + prob_over
            prob_under /= total
            prob_over /= total
            return prob_over, prob_under

        # Standard normal CDF
        z_score = (line - mean) / std
        
        # P(under) = P(X < line)
        raw_prob_under = stats.norm.cdf(z_score)
        raw_prob_over = 1 - raw_prob_under
        
        # Apply Shrinkage (Center toward 50%)
        prob_under = self._shrink_probability(raw_prob_under, confidence)
        prob_over = self._shrink_probability(raw_prob_over, confidence)
        
        # Renormalize to sum to 1 (optional, but good for display)
        total = prob_under + prob_over
        prob_under /= total
        prob_over /= total
        
        return prob_over, prob_under
    
    def _determine_direction(self, edge_over: float, edge_under: float,
                             model_confidence: str,
                             model_std: float,
                             regime_flags: List[str] = None,
                             market_type: str = 'points',
                             edge_analysis: Dict[str, Any] = None,
                             prediction_health: Dict[str, Any] = None,
                             market_consensus: Dict[str, Any] = None,
                             player_consensus: Dict[str, Any] = None) -> Tuple[str, str, str]:
        """
        Determine betting direction with STRICT GATING & REGIME DEFENSE (Phase 7)
        
        Rules:
        - If edge < threshold (after shrinkage) -> NO BET
        - If uncertainty is inflated (std > 8) -> Require massive edge
        - If REGIME SHIFT DETECTED -> Tighten gates automatically
        """
        regime_flags = regime_flags or []
        edge_analysis = edge_analysis or {}
        prediction_health = prediction_health or {}
        market_consensus = market_consensus or {}
        player_consensus = player_consensus or {}
        
        # Strict thresholds for Phase 6 "Dominant" mode
        edge_thresholds = {
            'high': 0.04,     # 4% edge allowed for high confidence
            'medium': 0.07,   # 7% edge for medium
            'low': 0.10,      # 10% edge required for low confidence
            'very_low': 0.15, # 15% edge required (rarely happens)
        }
        
        min_edge = edge_thresholds.get(model_confidence, 0.10)

        if market_type == 'points':
            min_edge += 0.015
        elif market_type in ('assists', 'rebounds'):
            min_edge = max(0.03, min_edge - 0.005)
        
        # 1. Volatility Gate
        if model_std > 8.0:
            min_edge += 0.04
            
        # 2. Regime Defense (Phase 7)
        # If system is decaying, tighten everything
        if 'ACCURACY_DECAY' in regime_flags:
            min_edge += 0.03  # Add 3% edge requirement
            
        # [R9] Graduated edge penalties based on defense level
        if 'SYSTEM_SEVERELY_OVERCONFIDENT' in regime_flags:
            min_edge += 0.06  # Maximum penalty — barely any picks will pass
        elif 'SYSTEM_OVERCONFIDENT' in regime_flags:
            min_edge += 0.03  # Strong penalty, but allow elite candidates to surface
        elif 'SYSTEM_RECOVERING' in regime_flags:
            min_edge += 0.02  # Mild penalty — system graduating back
        elif 'SYSTEM_SLIGHTLY_OVERCONFIDENT' in regime_flags:
            min_edge += 0.02  # Mild penalty
        
        if 'SYSTEM_UNDERCONFIDENT' in regime_flags:
            min_edge = max(0.01, min_edge - 0.02)  # Lower threshold (but keep > 1%)
        if 'SYSTEM_STRICT_INJURY_CONTEXT' in regime_flags:
            min_edge += 0.03
        if 'SYSTEM_SUPPRESS_FALLBACK_MODELS' in regime_flags:
            min_edge += 0.02

        edge_tier = edge_analysis.get('candidate_tier') or edge_analysis.get('tier')
        edge_score = float(edge_analysis.get('score', 0) or 0)
        edge_direction = edge_analysis.get('candidate_direction') or edge_analysis.get('direction')
        health_score = float(prediction_health.get('health_score', 1.0) or 1.0)
        used_fallback = bool(prediction_health.get('used_fallback_model'))
        market_trust_score = float(market_consensus.get('trust_score', 50.0) or 50.0)
        player_trust_score = float(player_consensus.get('trust_score', 50.0) or 50.0)
        disagreement = prediction_health.get('reference_disagreement')
        disagreement = float(disagreement) if disagreement is not None else None

        if market_trust_score < 40:
            min_edge += 0.03
        elif market_trust_score < 55:
            min_edge += 0.015
        elif market_trust_score >= 80 and player_trust_score >= 80:
            min_edge = max(0.02, min_edge - 0.01)

        if model_std >= 7.5:
            min_edge += 0.01
        if model_std >= 9.5:
            min_edge += 0.02
            
        best_edge = max(edge_over, edge_under)
        
        direction = 'OVER' if edge_over > edge_under else 'UNDER'

        # Synchronize the candidate and final gate: if the edge layer has a strong,
        # healthy candidate in the same direction, allow a modest threshold discount.
        same_direction_candidate = (
            edge_direction in ('OVER', 'UNDER') and
            edge_direction == direction
        )
        strong_health = health_score >= 0.90 and not used_fallback
        acceptable_disagreement = disagreement is None or disagreement <= 5.0
        if same_direction_candidate and strong_health and acceptable_disagreement:
            if edge_tier == 'parlay_core' and edge_score >= 82:
                min_edge = max(0.03, min_edge - 0.015)
            elif edge_tier == 'playable' and edge_score >= 72:
                min_edge = max(0.035, min_edge - 0.008)
        
        # 3. Bias Defense
        # If we know we are over-betting overs and losing, STOP doing it.
        if direction == 'OVER' and 'OVER_BIAS_DETECTED' in regime_flags:
            # Only take OVER if edge is massive
            min_edge += 0.05
            
        if direction == 'UNDER' and 'UNDER_BIAS_DETECTED' in regime_flags:
            min_edge += 0.05
            
        # 4. Weak Edge Penalty (Instruction 3.2)
        # Manually penalize known weak edges (Hardcoded for now, ideally dynamic)
        # Future: Read from edge_performance table
        if model_std > 9.0 and direction == 'OVER':
            # High variance Overs are historically dangerous "Fool's Gold"
            min_edge += 0.04
        
        close_call_candidate = (
            same_direction_candidate and
            strong_health and
            acceptable_disagreement and
            edge_tier == 'parlay_core' and
            edge_score >= 85 and
            model_std <= 6.5 and
            best_edge >= max(0.02, min_edge - 0.01)
        )

        if best_edge >= min_edge or close_call_candidate:
            # [FIX #6] Calibrated confidence: edge normalized by player variance
            # A 15% edge on a volatile player (std=10) is LESS reliable
            # than a 10% edge on a stable player (std=3)
            normalized_edge = best_edge / max(model_std, 1.0) * 10  # Scale to ~0-1
            
            if normalized_edge >= 0.60 and best_edge >= min_edge + 0.06:
                bet_conf = 'high'
            elif normalized_edge >= 0.38 and best_edge >= min_edge + 0.03:
                bet_conf = 'medium'
            else:
                bet_conf = 'low'
            if 'SYSTEM_SEVERELY_OVERCONFIDENT' in regime_flags and bet_conf != 'low':
                bet_conf = 'low'
            elif 'SYSTEM_OVERCONFIDENT' in regime_flags:
                if bet_conf == 'high':
                    bet_conf = 'medium'
                elif bet_conf == 'medium' and model_std >= 7.0:
                    bet_conf = 'low'

            if market_type == 'points':
                if bet_conf == 'high':
                    bet_conf = 'medium'
                elif bet_conf == 'medium' and model_std >= 6.5:
                    bet_conf = 'low'
            blocker = ''
            if close_call_candidate and best_edge < min_edge:
                blocker = "close-call promotion: elite candidate within 1% of threshold"
            return direction, bet_conf, blocker
        else:
            blocker_parts: List[str] = [f"edge {best_edge:.1%} below threshold {min_edge:.1%}"]
            if edge_tier in ('parlay_core', 'playable'):
                blocker_parts.append(f"candidate={edge_tier}")
            if model_std > 8.0:
                blocker_parts.append("high volatility")
            if used_fallback:
                blocker_parts.append("fallback model")
            if disagreement is not None and disagreement > 5.0:
                blocker_parts.append(f"disagreement {disagreement:.1f}")
            if market_trust_score < 55:
                blocker_parts.append(f"market trust {market_trust_score:.0f}")
            if 'SYSTEM_OVERCONFIDENT' in regime_flags or 'SYSTEM_SEVERELY_OVERCONFIDENT' in regime_flags:
                blocker_parts.append("system overconfident regime")
            return 'NO_BET', 'no_edge', "; ".join(blocker_parts)
    
    def _calculate_kelly(self, edge_over: float, edge_under: float,
                         direction: str, confidence: str, market_odds: float = -110) -> float:
        """
        Calculate Kelly criterion bet size using TRUE ODDS.
        
        Formula: f* = (bp - q) / b
        Where: b = decimal odds - 1, p = win prob, q = 1-p
        """
        if direction == 'NO_BET':
            return 0.0
        
        # Use the edge for the chosen direction
        edge = edge_over if direction == 'OVER' else edge_under
        if edge <= 0: return 0.0
        
        # Calculate 'p' (Win Probability)
        # edge = p - breakeven
        # so p = edge + breakeven
        breakeven = self._calculate_breakeven_prob(market_odds)
        win_prob = edge + breakeven
        
        # Calculate 'b' (Net Decimal Odds)
        decimal_odds = self._calculate_decimal_odds(market_odds)
        b = decimal_odds - 1
        
        # Kelly Formula
        # (b * p - q) / b  ==  p - (1-p)/b
        kelly = (b * win_prob - (1 - win_prob)) / b
        
        # Apply fractional Kelly based on confidence
        fraction_map = {
            'high': 0.5,    # Half Kelly
            'medium': 0.33, # Third Kelly
            'low': 0.25,    # Quarter Kelly
        }
        
        kelly *= fraction_map.get(confidence, 0.25)
        
        # Cap at max Kelly
        kelly = max(0, min(kelly, self.MAX_KELLY))
        
        return kelly
    
    def _generate_reasoning(self,
                            raw_mean: float,
                            calibrated_mean: float,
                            market_line: float,
                            prob_over: float,
                            prob_under: float,
                            edge_over: float,
                            edge_under: float,
                            player_context: Dict,
                            match_context: Dict,
                            model_confidence: str) -> List[str]:
        """
        Generate human-readable reasoning for the bet.
        
        Priority: Edge-first explanation > Model comparison > Context
        """
        reasons = []
        
        # ===== EDGE-FIRST EXPLANATION =====
        edge = match_context.get('edge_analysis', {})
        if edge and edge.get('tier') in ('parlay_core', 'playable'):
            # Lead with the edge explanation
            explanation = edge.get('explanation', '')
            if explanation:
                reasons.append(explanation)
            
            # Add tier badge
            tier = edge.get('tier', '')
            score = edge.get('score', 0)
            if tier == 'parlay_core':
                reasons.append(f"🔥 Edge Score: {score:.0f}/100 (Parlay Core)")
            elif tier == 'playable':
                reasons.append(f"✅ Edge Score: {score:.0f}/100 (Playable)")
            
            # Key Edge Question answers
            eq = edge.get('edge_questions', {})
            
            q3 = eq.get('Q3_script_kills', {})
            kill_count = q3.get('kill_count', 0)
            if kill_count >= 2:
                kills = q3.get('answer', [])
                reasons.append(f"Kill Scripts ({kill_count}/5): {'; '.join(kills[:3])}")
            
            q4 = eq.get('Q4_ceiling_needed', {})
            if q4.get('ceiling_needed') == False:
                reasons.append("Ceiling is OPTIONAL → team can win quietly")
            elif q4.get('ceiling_needed') == True:
                reasons.append("Ceiling is NEEDED → team depends on this production")
            
            # Narrative inflation
            narrative = edge.get('narrative_analysis', {})
            if narrative.get('narrative_inflation') in ('high', 'medium'):
                sources = narrative.get('inflation_sources', [])
                reasons.append(f"Narrative inflation: {', '.join(sources)}")
        
        # ===== FALLBACK: Model vs Line =====
        diff = calibrated_mean - market_line
        direction = 'above' if diff > 0 else 'below'
        reasons.append(f"Model: {calibrated_mean:.1f} pts, {abs(diff):.1f} {direction} line of {market_line:.1f}")
        
        # Probability insight
        best_dir = 'OVER' if prob_over > prob_under else 'UNDER'
        best_prob = max(prob_over, prob_under)
        reasons.append(f"{best_dir} probability: {best_prob*100:.1f}%")
        
        # Edge calculation
        best_edge = max(edge_over, edge_under)
        if best_edge > 0:
            reasons.append(f"Calculated edge: {best_edge*100:.1f}% vs breakeven")
        
        # Context factors
        stats = player_context.get('stats', {})
        l5_ppg = stats.get('l5_ppg')
        l15_ppg = stats.get('l15_ppg')
        
        if l5_ppg and l15_ppg:
            trend = l5_ppg - l15_ppg
            if abs(trend) > 2:
                trend_dir = 'up' if trend > 0 else 'down'
                reasons.append(f"Recent form: {trend_dir} {abs(trend):.1f} pts L5 vs L15")
        
        # Opponent context
        opp_def = match_context.get('opp_def_rating')
        if opp_def:
            avg_def = 112
            if opp_def > avg_def + 3:
                reasons.append(f"Favorable matchup: Opponent DEF RTG {opp_def:.1f} (weak)")
            elif opp_def < avg_def - 3:
                reasons.append(f"Tough matchup: Opponent DEF RTG {opp_def:.1f} (strong)")
        
        # Confidence caveat
        if model_confidence in ['low', 'very_low']:
            reasons.append(f"⚠️ Model confidence is {model_confidence} - exercise caution")
        
        return reasons
    
    def format_decision(self, decision: BettingDecision) -> str:
        """Format decision for display/output"""
        lines = [
            f"=== {decision.player_name} ===",
            f"Line: {decision.line:.1f} | Model: {decision.model_mean:.1f} ± {decision.model_std:.1f}",
            f"P(Over): {decision.prob_over*100:.1f}% | P(Under): {decision.prob_under*100:.1f}%",
            f"Edge Over: {decision.edge_over*100:+.1f}% | Edge Under: {decision.edge_under*100:+.1f}%",
            "",
            f">>> {decision.direction} ({decision.confidence} confidence)",
            f">>> Kelly: {decision.kelly_fraction*100:.2f}% of bankroll",
            "",
            "Reasoning:"
        ]
        
        for reason in decision.reasoning:
            lines.append(f"  • {reason}")
        
        return "\n".join(lines)
    
    def batch_calibrate(self,
                        predictions: List[Dict],
                        market_lines: List[float],
                        player_names: List[str],
                        player_contexts: List[Dict],
                        match_contexts: List[Dict]) -> List[BettingDecision]:
        """Calibrate multiple predictions at once"""
        decisions = []
        for pred, line, name, p_ctx, m_ctx in zip(
            predictions, market_lines, player_names, player_contexts, match_contexts
        ):
            decision = self.calibrate(pred, line, name, p_ctx, m_ctx)
            decisions.append(decision)
        
        return decisions
    
    def filter_actionable(self, decisions: List[BettingDecision],
                          min_confidence: str = 'low') -> List[BettingDecision]:
        """Filter to only actionable bets"""
        confidence_order = ['no_edge', 'low', 'medium', 'high']
        min_idx = confidence_order.index(min_confidence)
        
        return [d for d in decisions 
                if d.direction != 'NO_BET' 
                and confidence_order.index(d.confidence) >= min_idx]
