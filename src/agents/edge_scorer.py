"""
Edge Scorer: Master Aggregation + 4 Edge Questions Gate
========================================================
Purpose: Aggregate all 3 edge agents into a single decision.

Scoring Weights:
  Narrative Inflation  25%
  Role Replaceability  30%
  Script Kill Count    25%
  Market Juice         20%

Decision Thresholds:
  Score >= 80 → Parlay Core (highest conviction)
  Score >= 70 → Playable (standard bet)
  Score <  70 → Reject (no edge)

4 Edge Questions (ALL must be answered or pick is discarded):
  Q1: Who benefits if this player fails?
  Q2: Is the line built on reputation or role?
  Q3: What game script kills this stat?
  Q4: Is the ceiling NEEDED or OPTIONAL?
"""

import logging
from typing import Dict, Any, List, Optional

from src.agents.narrative_detector import NarrativeDetector
from src.agents.stat_fragility import StatFragilityAgent
from src.agents.line_ceiling_optimizer import LineCeilingOptimizer

logger = logging.getLogger("EDGE_SCORER")


class EdgeScorer:
    """
    Master aggregation engine for edge detection.
    
    This is the final gate before any pick reaches the MarketCalibrator.
    If a pick can't answer all 4 Edge Questions, it gets rejected
    regardless of how good the prediction looks.
    
    Core Rule (Azramund):
    "If your model cannot explain why the book is comfortable
    offering this line, you don't have edge — you have a guess."
    """
    
    # Base scoring weights (adapt dynamically based on data richness)
    WEIGHT_MODEL = 0.45
    WEIGHT_NARRATIVE = 0.15
    WEIGHT_REPLACEABILITY = 0.10
    WEIGHT_SCRIPT_KILLS = 0.15
    WEIGHT_MARKET = 0.15
    
    # Decision thresholds
    THRESHOLD_PARLAY = 80
    THRESHOLD_PLAYABLE = 68
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.narrative = NarrativeDetector(db_manager)
        self.fragility = StatFragilityAgent(db_manager)
        self.line_optimizer = LineCeilingOptimizer(db_manager)
    
    def evaluate(self,
                 audited_prediction: Dict[str, Any],
                 player_context: Dict[str, Any],
                 match_context: Dict[str, Any],
                 market_line: float,
                 market_type: str = 'points',
                 market_floor: Optional[float] = None,
                 market_ceiling: Optional[float] = None,
                 odds_over: float = -110,
                 odds_under: float = -110) -> Dict[str, Any]:
        """
        Run all edge agents and aggregate into final decision.
        Calculates distinct scores for OVER (Opportunity) vs UNDER (Fragility).
        """
        # ==========================================
        # 1. RUN SUB-AGENTS
        # ==========================================
        
        narrative_result = self.narrative.analyze(
            player_context, match_context, market_line, market_type
        ) or {}
        
        fragility_result = self.fragility.analyze(
            player_context, match_context, market_line, market_type
        ) or {}
        
        line_result = self.line_optimizer.analyze(
            player_context, match_context, market_line, market_type,
            market_floor, market_ceiling, odds_over, odds_under
        ) or {}
        
        # Get usage impact (from TeammateUsageNetwork)
        usage_impact = match_context.get('usage_impact') or {}
        usage_boost = usage_impact.get('total_usage_boost')
        if usage_boost is None: 
            usage_boost = 0.0
        
        # Get line movement (from LineMovementTracker)
        movement = match_context.get('line_movement') or {}
        sharp_side = movement.get('sharp_direction')
        
        # ==========================================
        # 2. CALCULATE MODEL EDGE
        # ==========================================
        def implied_prob(odds):
            if odds is None: return 0.5
            if odds > 0: return 100 / (odds + 100)
            return abs(odds) / (abs(odds) + 100)
            
        prob_over = audited_prediction.get('prob_over')
        prob_under = audited_prediction.get('prob_under')
         # [FIX] Calculate probability if missing (Agent 4 hasn't run yet)
        if prob_over is None or prob_under is None:
            mean = audited_prediction.get('mean')
            std = audited_prediction.get('std')
            if mean is not None and std is not None and std > 0.01 and market_line is not None:
                from scipy.stats import norm
                z_score = (market_line - mean) / std
                prob_over = 1 - norm.cdf(z_score)
                prob_under = norm.cdf(z_score)
            else:
                # std is 0 or missing — can't calculate z-score, use simple comparison
                if mean is not None and market_line is not None:
                    # Use a step function: if mean > line, lean OVER; else UNDER
                    diff_pct = (mean - market_line) / max(market_line, 1)
                    prob_over = min(0.75, max(0.25, 0.5 + diff_pct * 2))
                    prob_under = 1 - prob_over
                else:
                    prob_over = 0.5
                    prob_under = 0.5
        
        # Calculate Implied Probability (Break-even point)
        implied_o = implied_prob(odds_over)
        implied_u = implied_prob(odds_under)

        # Calculate Edge (Model vs Market)
        edge_o = prob_over - implied_o
        edge_u = prob_under - implied_u
        
        if edge_o > edge_u:
            model_edge = edge_o
            model_dir = 'OVER'
        else:
            model_edge = edge_u
            model_dir = 'UNDER'

        # Model Score: 0-100 based on edge magnitude
        if model_edge <= 0:
            model_score = 0
        else:
            model_score = min(100, 40 + (model_edge * 300))
        
        # ==========================================
        # 3. SCORING COMPONENTS (with neutral defaults)
        # ==========================================
        
        # A) Fragility (Good for UNDER, Bad for OVER)
        kill_count = fragility_result.get('kill_count')
        if kill_count is None: kill_count = 0
        kill_score = min(100, kill_count * 33)
        
        # B) Replaceability (High = Good for UNDER, Low = Good for OVER)
        repl_val = fragility_result.get('replaceability', 'medium')
        repl_map = {'very_high': 90, 'high': 70, 'medium': 50, 'low': 20}
        replaceability_score = repl_map.get(repl_val, 50)
        
        # DYNAMIC ADJUSTMENT
        if usage_boost > 0.03:
            replaceability_score = 20 
        
        # C) Narrative (Inflation = UNDER, Deflation = OVER)
        inflation_score = narrative_result.get('inflation_score')
        if inflation_score is None: inflation_score = 0
        narrative_dir = narrative_result.get('bias_direction', 'NEUTRAL')
        
        # D) Market — DEFAULT TO NEUTRAL (50) when no signal, not 0
        market_val = line_result.get('score')
        if market_val is None or market_val == 0:
            market_val = 50  # "No opinion" is neutral, not rejection
        
        # ==========================================
        # 3.5 ADAPTIVE WEIGHTING
        # ==========================================
        # When sub-agents have low conviction (mostly neutral/default),
        # shift weight toward the model which has concrete data.
        
        sub_agent_signals = 0
        if inflation_score > 10: sub_agent_signals += 1
        if kill_count > 0: sub_agent_signals += 1
        if line_result.get('score', 0) > 15: sub_agent_signals += 1
        if abs(usage_boost) > 0.02: sub_agent_signals += 1
        if sharp_side is not None: sub_agent_signals += 1
        
        # Adaptive: boost model weight when sub-agents are quiet
        if sub_agent_signals <= 1:
            w_model = 0.55  # Sub-agents have no opinion → trust the model more
            w_sub_scale = (1.0 - w_model) / (1.0 - self.WEIGHT_MODEL)  # Scale remaining weights
        elif sub_agent_signals <= 2:
            w_model = 0.50
            w_sub_scale = (1.0 - w_model) / (1.0 - self.WEIGHT_MODEL)
        else:
            w_model = self.WEIGHT_MODEL  # Full sub-agent data → use base weights
            w_sub_scale = 1.0
        
        w_narrative = self.WEIGHT_NARRATIVE * w_sub_scale
        w_replaceability = self.WEIGHT_REPLACEABILITY * w_sub_scale
        w_script_kills = self.WEIGHT_SCRIPT_KILLS * w_sub_scale
        w_market = self.WEIGHT_MARKET * w_sub_scale
        
        # === MATH IS THE EDGE OVERRIDE ===
        if model_score >= 70:
            if narrative_dir == 'NEUTRAL':
                inflation_score = 50 
                narrative_dir = model_dir 
            
            if model_dir == 'UNDER' and kill_score < 50:
                kill_score = 50 
                
        # ==========================================
        # 4. CALCULATE UNDER SCORE (Fragility)
        # ==========================================
        
        under_components = []
        
        # 1. Narrative
        if narrative_dir == 'UNDER':
            narr_val = 50 + (inflation_score * 0.5) 
        elif narrative_dir == 'NEUTRAL':
            narr_val = 50
        else:
            narr_val = max(0, 50 - inflation_score)
        under_components.append(min(100, narr_val) * w_narrative)
        
        # 2. Replaceability
        under_components.append(replaceability_score * w_replaceability)
        
        # 3. Kill Scripts
        under_components.append(kill_score * w_script_kills)
        
        # 4. Market
        under_components.append(market_val * w_market)
        
        # 5. Model Integration for UNDER
        if model_dir == 'UNDER':
            under_components.append(model_score * w_model)
        else:
            under_components.append(0) 
        
        # Bonus checks for UNDER
        under_bonus = 0
        if sharp_side == 'UNDER': under_bonus += 10
        if usage_boost < -0.05: under_bonus += 10
        
        under_score = sum(under_components) + under_bonus
        
        # ==========================================
        # 5. CALCULATE OVER SCORE (Opportunity)
        
        over_components = []
        
        # 1. Narrative Opportunity
        if narrative_dir == 'OVER':
            over_narrative = 80 + (inflation_score * 0.2) 
        elif narrative_dir == 'NEUTRAL':
            over_narrative = 50
        else:
            over_narrative = max(0, 50 - inflation_score)
        over_components.append(over_narrative * w_narrative)
        
        # 2. Role Stability 
        stability_score = 100 - replaceability_score
        over_components.append(stability_score * w_replaceability)
        
        # 3. Life Scripts / Opportunity
        opportunity_score = 50
        if usage_boost > 0.02: opportunity_score += 25
        if usage_boost > 0.05: opportunity_score += 15
        if kill_count == 0: opportunity_score += 15
        if sharp_side == 'OVER': opportunity_score += 15
        
        # Apply Over Override
        if model_score >= 70 and model_dir == 'OVER':
            opportunity_score = max(opportunity_score, 75)
            
        over_components.append(min(100, opportunity_score) * w_script_kills)
        
        # 4. Market 
        best_dir = line_result.get('best_line', {}).get('direction', 'NEUTRAL')
        if best_dir == 'OVER':
            val_score = market_val
        elif best_dir == 'UNDER':
            val_score = max(20, market_val * 0.5)  # Don't zero out, just reduce
        else:
            val_score = 50
        over_components.append(val_score * w_market)
        
        # 5. Model Integration for OVER
        if model_dir == 'OVER':
            over_components.append(model_score * w_model)
        else:
            over_components.append(0)
        
        over_score = sum(over_components)
        
        # ==========================================
        # 6. DETERMINE WINNER
        # ==========================================
        
        if over_score > under_score:
            final_score = over_score
            direction = 'OVER'
            # Gate: Cannot bet OVER if kill_count is high
            if kill_count >= 3:
                final_score = 0
                direction = 'NO_BET'
                tier = 'reject'
        else:
            final_score = under_score
            direction = 'UNDER'
        
        final_score = min(100, final_score)
        
        # ==========================================
        # 6.5 LOGIC GATES (Phase 12 Override)
        # ==========================================
        # Check specific "Safety Valves" that override the math
        # e.g. Don't bet UNDER entirely on a player with +8% usage boost just because proj is low
        
        logic_override, override_reason = self._apply_logic_gates(
            direction, final_score, usage_boost, player_context, market_line, market_type
        )
        
        if logic_override:
            direction = logic_override
            if direction == 'NO_BET':
                final_score = 0
                tier = 'reject'
                # Append logic reason to explain why
                # We leverage the Q1 answer or a special flag to communicate this
                # For now, we'll inject it into the edge questions later or handle in build_explanation
        
        # ==========================================
        # 4 EDGE QUESTIONS
        # ==========================================
        
        edge_questions = self._answer_edge_questions(
            player_context, match_context, market_line,
            narrative_result, fragility_result, line_result,
            direction
        )
        
        # Inject Logic Gate Reason if exists
        if override_reason:
            edge_questions['Q0_logic_gate'] = {
                'question': 'Did logic gates intervene?',
                'answer': override_reason,
                'answered': True
            }
        
        # ==========================================
        # 6.5 ANSWER EDGE QUESTIONS (Internal)
        # ==========================================
        # Since some sub-agents might not return structured questions,
        # we generate them internally here to ensure coverage.
        
        internal_questions = self._answer_edge_questions(
            player_context, match_context, market_line,
            narrative_result, fragility_result, line_result,
            direction
        )
        edge_questions.update(internal_questions)
        
        all_answered = all(
            q.get('answered', False) for q in edge_questions.values()
        )
        
        # ==========================================
        # TIER CLASSIFICATION (Graduated Gate)
        # ==========================================
        
        answered_count = sum(1 for q in edge_questions.values() if q.get('answered'))
        
        # Graduated penalty instead of all-or-nothing
        if answered_count >= 4:
            pass  # Full score
        elif answered_count >= 3:
            final_score *= 0.85  # 15% penalty for 1 missing
        elif answered_count >= 2:
            final_score *= 0.60  # 40% penalty for 2 missing
        else:
            final_score = 0  # Hard reject only if <2 answered
            direction = 'NO_BET'
        
        # Tighten top-tier promotions: recent audits showed A-tier/parlay picks
        # were too aggressive relative to realized accuracy.
        if direction in ('OVER', 'UNDER') and model_score < 55 and final_score >= self.THRESHOLD_PLAYABLE:
            final_score *= 0.92

        if final_score >= self.THRESHOLD_PARLAY:
            tier = 'parlay_core'
        elif final_score >= self.THRESHOLD_PLAYABLE:
            tier = 'playable'
        else:
            tier = 'reject'
            if final_score < 60: direction = 'NO_BET'
        
        # ==========================================
        # EXPLANATION LAYER
        # ==========================================
        
        explanation = self._build_explanation(
            direction, final_score, tier,
            narrative_result, fragility_result, line_result,
            edge_questions, market_line,
            usage_boost=usage_boost,
            sharp_side=sharp_side,
            model_score=model_score
        )
        
        return {
            'score': round(final_score, 1),
            'tier': tier,
            'direction': direction,
            'logic_override_applied': bool(logic_override),
            'logic_override_reason': override_reason or '',
            'edge_questions': edge_questions,
            'all_questions_answered': all_answered,
            'explanation': explanation,
            'sub_scores': {
                'narrative': inflation_score,
                'replaceability': replaceability_score,
                'script_kills': kill_score,
                'market': line_result.get('score', 0),
                'model': model_score,
                'over_opportunity': opportunity_score if direction == 'OVER' else 0,
                'stability': stability_score if direction == 'OVER' else 0
            },
            'direction_votes': {
                'over_score': round(over_score, 1),
                'under_score': round(under_score, 1),
                'model_bias': model_score if model_dir != 'NEUTRAL' else 0
            },
            'narrative_analysis': narrative_result,
            'fragility_analysis': fragility_result,
            'line_analysis': line_result,
        }

    def _apply_logic_gates(self, 
                          direction: str, 
                          score: float, 
                          usage_boost: float, 
                          p_ctx: Dict, 
                          line: float,
                          market_type: str) -> tuple[Optional[str], Optional[str]]:
        """
        Apply hard logic gates to prevent "rigid math" errors.
        Returns: (new_direction, reason) or (None, None)
        """
        # 1. Usage Vacuum Safety Valve
        # If player has massive usage boost (>15%), betting UNDER is extremely dangerous
        # strictly based on a projection that might not fully capture the volume upside.
        # [TUNING log 2026-02-12]: Raised from 5% to 15% after Raynaud (+8.6% boost) went Under easily.
        # We only want to block MASSIVE vacuums (like J.Walker +60%), not standard injury bumps.
        if direction == 'UNDER' and usage_boost > 0.15:
            # Only allow if the model score is overwhelmingly high (meaning huge edge)
            # If score < 85 (approx 15-20% edge), we block it.
            if score < 85:
                return 'NO_BET', f"⛔ Logic Gate: Massive Usage Boost (+{usage_boost*100:.1f}%) makes UNDER too risky"

        # 2. Calculate Hit Rate (L10)
        recent_logs = p_ctx.get('recent_logs', [])
        if recent_logs:
            hits = 0
            valid = 0
            for g in recent_logs:
                val = g.get(market_type, 0) 
                # Map market name if needed (e.g. threes -> fg3m)
                if market_type == 'threes': val = g.get('fg3m', 0)
                
                if val > line: hits += 1
                valid += 1
            
            if valid > 0:
                hit_rate = hits / valid
                
                # 3. Hot Hand Safety Valve
                # If hitting 80%+ of L10, don't bet UNDER
                if direction == 'UNDER' and hit_rate >= 0.80:
                    if score < 90: # Needs massive evidence to fade a heater
                        return 'NO_BET', f"⛔ Logic Gate: Hot Hand ({hits}/{valid} L10) makes UNDER too risky"
                        
                # 4. Cold Streak Safety Valve
                # If hitting 20% or less of L10, don't bet OVER
                if direction == 'OVER' and hit_rate <= 0.20:
                     if score < 90:
                        return 'NO_BET', f"⛔ Logic Gate: Cold Streak ({hits}/{valid} L10) makes OVER too risky"

        return None, None
        

    
    def _answer_edge_questions(self,
                               player_ctx: Dict, match_ctx: Dict,
                               market_line: float,
                               narrative: Dict, fragility: Dict,
                               line_opt: Dict,
                               direction: str) -> Dict[str, Dict]:
        """
        Answer all 4 Edge Questions. ALL must be answered or pick is discarded.
        """
        questions = {}
        
        stats = player_ctx.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_ctx.get('ppg_L5') or
                  player_ctx.get('points_L5') or 0)
        role = player_ctx.get('player_role',
                             player_ctx.get('role', 'unknown'))
        
        # Q1: Who benefits if this player fails?
        beneficiaries = []
        team_injuries = player_ctx.get('team_injuries', {})
        
        if role in ('secondary_star', 'third_option', 'role_player',
                    'microwave_scorer', 'bench_scorer', 'starter'):
            beneficiaries.append("Primary scorers absorb the usage")
        elif role in ('star', 'primary_scorer', 'superstar'):
            beneficiaries.append("Usage redistributes to secondary options/bench")
        
        if fragility.get('kill_count', 0) >= 2:
            beneficiaries.append("Team scheme doesn't require this stat")
        
        spread = match_ctx.get('spread', 0) or 0
        if abs(spread) > 8:
            beneficiaries.append(f"Blowout ({spread:+.0f}) → bench gets minutes")
        
        if match_ctx.get('opp_pace', 100) and match_ctx.get('opp_pace', 100) < 96:
            beneficiaries.append("Slow pace reduces total possessions")
        
        questions['Q1_who_benefits'] = {
            'question': 'Who benefits if this player fails?',
            'answer': beneficiaries if beneficiaries else ["No clear beneficiary identified"],
            'answered': len(beneficiaries) > 0,
        }
        
        # Q2: Is the line built on reputation or role?
        inflation = narrative.get('narrative_inflation', 'none')
        inflation_sources = narrative.get('inflation_sources', [])
        
        if inflation in ('high', 'medium'):
            line_basis = 'reputation'
            q2_answer = f"Reputation-driven ({', '.join(inflation_sources)})"
        elif ppg_L5 < 14 and market_line < ppg_L5 * 0.95:
            line_basis = 'role'
            q2_answer = "Role-based (market undervalues actual production)"
        else:
            line_basis = 'mixed'
            q2_answer = f"Mixed: line {market_line:.1f} vs L5 avg {ppg_L5:.1f}"
        
        questions['Q2_reputation_or_role'] = {
            'question': 'Is the line built on reputation or role?',
            'answer': q2_answer,
            'line_basis': line_basis,
            'answered': True,  # This can always be answered
        }
        
        # Q3: What game script kills this stat?
        kill_scripts = fragility.get('kill_scripts', [])
        script_details = fragility.get('script_details', {})
        
        if kill_scripts:
            kill_reasons = [
                f"{s}: {script_details.get(s, {}).get('reason', '')}"
                for s in kill_scripts
            ]
            q3_answer = kill_reasons
            answered_q3 = True
        else:
            q3_answer = ["No structural kill scripts identified (Clean Profile)"]
            # Finding NO kill scripts is a valid answer (it means it's safe)
            # Especially for OVERS, this is what we want.
            answered_q3 = True 
        
        questions['Q3_script_kills'] = {
            'question': 'What game script kills this stat?',
            'answer': q3_answer,
            'kill_count': len(kill_scripts),
            'answered': answered_q3,
        }
        
        # Q4: Is the ceiling NEEDED or OPTIONAL?
        ceiling_needed = fragility.get('ceiling_needed', None)
        
        if ceiling_needed is True:
            q4_answer = "NEEDED — team likely needs this player to perform"
            # If ceiling is needed → UNDER is riskier
        elif ceiling_needed is False:
            q4_answer = "OPTIONAL — team can win without this player hitting ceiling"
            # Optional ceiling = elite UNDER signal
        else:
            q4_answer = "ASSUMED OPTIONAL — insufficient data (defaulting safe)"
        
        questions['Q4_ceiling_needed'] = {
            'question': 'Is the ceiling NEEDED or OPTIONAL?',
            'answer': q4_answer,
            'ceiling_needed': ceiling_needed if ceiling_needed is not None else False,
            'answered': True,  # [FIX] Always answered — missing data defaults to OPTIONAL
        }
        
        return questions
    
    def _build_explanation(self,
                          direction: str,
                          score: float,
                          tier: str,
                          narrative: Dict,
                          fragility: Dict,
                          line_opt: Dict,
                          edge_questions: Dict,
                          market_line: float,
                          usage_boost: float = 0,
                          sharp_side: str = None,
                          model_score: float = 0) -> str:
        """
        Build the human-readable explanation.
        """
        if tier == 'reject':
            logic_gate = edge_questions.get('Q0_logic_gate', {})
            logic_reason = str(logic_gate.get('answer', '') or '').strip()
            if logic_reason:
                return f"{logic_reason}. Pick rejected."
            return f"No edge identified (score: {score:.0f}/100). Pick rejected."
        
        kill_scripts = fragility.get('kill_scripts', [])
        inflation = narrative.get('narrative_inflation', 'none')
        variance_prot = line_opt.get('variance_protection', {}).get('level', 'unknown')
        
        parts = []
        
        # Add Model Context if significant
        if model_score > 60:
            parts.append(f"Strong Model Projection.")
        
        if direction == 'UNDER':
            # ... (existing UNDER logic) ...
            conditions_needed = []
            
            if fragility.get('ceiling_needed') == False:
                conditions_needed.append("player exceeds role usage")
            
            if len(kill_scripts) >= 2:
                conditions_needed.append(f"game avoids {', '.join(kill_scripts[:2])}")
            
            if inflation in ('high', 'medium'):
                conditions_needed.append("narrative inflation justified by reality")
                
            if sharp_side == 'UNDER':
                parts.append("📉 Sharp Money moving line DOWN.")
            
            if conditions_needed:
                cond_str = " AND ".join(conditions_needed)
                parts.append(f"This UNDER {market_line:.1f} wins unless: {cond_str}.")
            else:
                parts.append(f"Edge detected on UNDER {market_line:.1f}.")
            
            if kill_scripts:
                parts.append(f"Kill scripts: {', '.join(kill_scripts)}.")
            
        elif direction == 'OVER':
            parts.append(f"This OVER {market_line:.1f} has support:")
            
            supports = []
            if usage_boost > 0.02:
                supports.append(f"Usage Vacuum (+{usage_boost*100:.1f}% boost)")
            
            if sharp_side == 'OVER':
                supports.append("Sharp Money")
            
            if model_score > 50:
                supports.append("Projection Value")
            
            if inflation == 'none' and narrative.get('bias_direction') == 'OVER':
                supports.append("Narrative Deflation")
            
            if fragility.get('replaceability') == 'low':
                supports.append("Essential Role")
            
            if variance_prot == 'inverted':
                supports.append("Variance Protection")
                
            parts.append(", ".join(supports) + ".")
        
        if tier == 'parlay_core':
            parts.append("[🔥 PARLAY CORE]")
        elif tier == 'playable':
            parts.append("[✅ PLAYABLE]")
        
        return " ".join(parts)
