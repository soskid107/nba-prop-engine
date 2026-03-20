"""
Orchestrator: Multi-Agent Prediction Pipeline
==============================================
Coordinates the 5-agent system for NBA player points predictions.

Pipeline Flow:
1. Agent 1 (Data Gatherer) → Collect player & match context
2. Agent 2 (Mechanistic Modeler) → Generate prediction distributions
3. Agent 3 (Auditor) → Apply skeptical filters
4. Agent 4 (Market Calibrator) → Convert to betting decisions
5. Agent 5 (Learning Loop) → Log for post-game learning

Principles:
- Each agent has a single responsibility
- Information flows forward, never backward
- No agent sees sportsbook lines except Agent 4
- Agent 5 only runs post-game, never affects live predictions
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from src.utils.database import DatabaseManager
from src.agents.data_gatherer import DataGathererAgent
from src.agents.mechanistic_modeler import MechanisticModelerAgent
from src.agents.auditor import AuditorAgent
from src.agents.market_calibrator import MarketCalibratorAgent, BettingDecision
from src.agents.learning_loop import LearningLoopAgent
from src.models.ml_model import ResidualAdjuster
from src.learning.feature_store import FeatureStore
from src.simulation.monte_carlo import SimulationEngine
from src.agents.edge_scorer import EdgeScorer
from src.agents.minutes_model import MinutesProjector
from src.agents.line_movement import LineMovementTracker
from src.agents.edge_tracker import EdgeTracker
from src.agents.teammate_network import TeammateUsageNetwork
from src.agents.defensive_schemes import DefensiveSchemeAnalyzer
from src.agents.sgp_engine import SGPEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Complete prediction with all agent outputs"""
    player_id: int
    player_name: str
    team: str
    opponent: str
    
    # Agent 1: Context
    player_context: Dict[str, Any]
    match_context: Dict[str, Any]
    
    # Agent 2: Raw prediction
    raw_prediction: Dict[str, Any]
    
    # Agent 3: Audited prediction
    audited_prediction: Dict[str, Any]
    audit_flags: List[Any]
    
    # Agent 4: Betting decision
    betting_decision: Optional[BettingDecision]
    
    # Metadata
    timestamp: str
    pipeline_success: bool


class PredictionOrchestrator:
    """
    The conductor of the multi-agent prediction system.
    
    Coordinates data flow between agents and ensures
    each agent receives only the information it needs.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db = DatabaseManager(db_path)
        
        # Initialize agents
        self.agent1_legacy = DataGathererAgent(self.db)
        self.agent1 = self.agent1_legacy # Alias for compatibility
        
        # [new] Consensus Engine (The "GenLayer" Trust Layer)
        from src.agents.consensus.engine import ConsensusEngine
        self.consensus_engine = ConsensusEngine(self.db)
        
        self.agent2 = MechanisticModelerAgent(self.db)  # Kept as fallback
        self.agent3 = AuditorAgent(self.db)
        self.agent4 = MarketCalibratorAgent(self.db)
        self.agent5 = LearningLoopAgent(self.db)
        
        # [NEW] Market Selector Brain
        from src.agents.market_selector import MarketSelectorAgent
        self.market_selector = MarketSelectorAgent()

        # [NEW] Multi-Market Production Models
        from src.models.simple_predictor import MarketPredictor
        self.predictors = {
            'points': MarketPredictor('points'),
            'assists': MarketPredictor('assists'),
            'rebounds': MarketPredictor('rebounds'),
            # [PHASE 13] New Markets
            'threes': MarketPredictor('threes'),
            'blocks': MarketPredictor('blocks'),
            'steals': MarketPredictor('steals'),
            'field_goals': MarketPredictor('field_goals')
        }
        # self.predictors = {}
        
        # [NEW] Simulation Engine for Points (Monte Carlo)
        # We use the trusted "normal" logic for points
        self.sim_engine = SimulationEngine(self.db)
        if not self.sim_engine.load_models():
            logger.warning("Failed to load simulation models - using fallback logic")
        
        # [NEW] Edge Scoring Layer
        self.edge_scorer = EdgeScorer(self.db)
        
        # [NEW] Enhancement Agents
        self.minutes_projector = MinutesProjector()
        self.line_tracker = LineMovementTracker(self.db)
        self.edge_tracker = EdgeTracker(self.db)
        self.teammate_network = TeammateUsageNetwork(self.db)
        self.defensive_analyzer = DefensiveSchemeAnalyzer(self.db)
        self.sgp_engine = SGPEngine()
        
        # Legacy ML Layer (kept for compatibility)
        self.ml_model = ResidualAdjuster()
        self.feature_store = FeatureStore()
        
        logger.info("Orchestrator initialized with full edge pipeline + Market Selector")
    
    def predict_player(self,
                       player_id: int,
                       opponent: str,
                       reference_date: Optional[str] = None,
                       market_line: Optional[float] = None,
                       team_injuries: Optional[Dict] = None,
                       market_odds: float = -110,
                       calibration_flags: List[str] = None,
                       market_floor: Optional[float] = None,
                       market_ceiling: Optional[float] = None,
                       preloaded_player_context: Optional[Dict] = None,
                       active_props: List[Dict] = None) -> List[PredictionResult]:
        """
        Run full prediction pipeline for a single player
        
        Args:
            player_id: NBA player ID
            opponent: 3-letter opponent abbreviation
            market_line: Optional sportsbook points line
            team_injuries: Optional dict of team injuries
            market_odds: American odds for the bet
            calibration_flags: List of system-wide calibration flags
            market_floor: Lowest available market line (context)
            market_ceiling: Highest available market line (context)
            preloaded_player_context: Optional pre-fetched context to skip DB call
            active_props: List of prop dictionaries from DB (optional)

            
        Returns:
            List of PredictionResult objects (one per market)
        """
        timestamp = datetime.now().isoformat()
        game_date = reference_date or datetime.now().strftime('%Y-%m-%d')  # Use slate date for rest/B2B and line tracking
        results = []
        
        try:
            # Get player info
            player_name, team = self._get_player_info(player_id)
            
            # ===== GENLAYER CONSENSUS CHECK =====
            # Instead of blindly predicting, we first "Propose" the player to the Validation Jury.
            # We check the primary market (e.g., Points > Line)
            # For efficiency, we use a rough line estimate or just check general status
            
            # 1. Run Consensus on "General Playability" (Line = 0 for general check)
            # Or better, we check if they are "Playable" at all.
            # NarrativeValidator handles Injury/Rest VETO.
            # DataValidator handles "Do we have enough data?" ABSTAIN/NO.
            
            verdict = self.consensus_engine.evaluate_player_viability(player_id, reference_date=game_date)
            
            if verdict.decision == "REJECTED":
                logger.info(f"  [CONSENSUS REJECT] {player_name}: {verdict.reasoning}")
                self.agent5.log_rejection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    rejection_stage='player_consensus',
                    reason=verdict.reasoning,
                    consensus_level=verdict.consensus_level,
                    player_trust_score=verdict.trust_score,
                    validator_vote_summary=str(verdict.trust_components),
                )
                return [] # Skip this player
                
            # If Approved, we proceed to gather full context for the Model
            # The Consensus Engine already gathered some, but we need the full "Model Ready" context
            # for the ML pipeline. 
            # ideally ConsensusEngine returns the context it used, but for now we re-gather or
            # ask DataGatherer (which is consistent since it hits DB/Cache).
            
            player_context = preloaded_player_context or self.agent1_legacy.gather_player_context(player_id, date_limit=game_date)
            match_context = self.agent1_legacy.gather_match_context(team, opponent, game_date=game_date)
            match_context['regime_flags'] = calibration_flags or []
            team_injury_map = {pid: prob for pid, prob in (team_injuries or {}).items() if pid != player_id}
            player_context['team_injuries'] = team_injury_map
            match_context['team_injuries'] = team_injury_map
            match_context['injury_context_present'] = team_injuries is not None
            match_context['injury_context_size'] = len(team_injury_map)
            
            # Inject Narrative Info from Verdict into Context
            # (So the model 'knows' why they were approved, e.g. "Well Rested")
            player_context['consensus_verdict'] = verdict.decision
            player_context['consensus_level'] = verdict.consensus_level # [NEW] Needed for sizing
            player_context['consensus_reason'] = verdict.reasoning
            match_context['player_consensus'] = {
                'status': verdict.decision,
                'level': verdict.consensus_level,
                'reason': verdict.reasoning,
                'trust_score': verdict.trust_score,
                'trust_components': verdict.trust_components,
            }
            
            # Fallback for team name
            if team == 'UNK' and player_context.get('team'):
                team = player_context['team']
            
            # Determine Markets to Run
            markets_to_run = []
            if active_props:
                markets_to_run = active_props
            elif market_line:
                # Legacy single-mode
                markets_to_run = [{
                    'market_key': 'player_points', 
                    'line': market_line, 
                    'odds_over': market_odds, # Approximate
                    'odds_under': market_odds
                }]
            else:
                # No lines available? Skip.
                return []
            
            # [NEW] Enhanced Context (Head-to-Head)
            # Fetch last 3 games vs this opponent for reasoning context
            h2h_stats = self.agent1.get_player_vs_opponent_history(player_id, opponent)
            match_context['h2h_history'] = h2h_stats

            # [NEW] Teammate Correlation Analysis (Phase 11)
            # Identify missing teammates and calculate impact
            if team_injuries:
                missing_players = []
                # team_injuries is {player_id: probability_of_playing}
                for pid, prob in team_injuries.items():
                    if prob < 0.3: # OUT or Doubtful
                        # Determine name for reporting (simple DB lookup or placeholder)
                        p_name = "Teammate" # Default
                        # optimization: could fetch name, but ID is what matters for calc
                        missing_players.append({'player_id': pid, 'player_name': p_name})
                
                if missing_players:
                    # Analyze impact
                    impact = self.teammate_network.analyze_usage_impact(
                        player_id, player_name, team, missing_players, game_date
                    )
                    match_context['teammate_impact'] = impact

            lineup_context = self._build_lineup_context(
                player_context,
                team_injury_map,
                match_context.get('teammate_impact')
            )
            player_context['lineup_context'] = lineup_context
            match_context['lineup_context'] = lineup_context

            # ==========================================
            # MARKET SELECTOR (Brain)
            # ==========================================
            aggregated_markets = {}
            for prop in markets_to_run:
                mkey = prop.get('market_key', 'player_points')
                line = prop.get('line')
                if line is None: continue
                
                if mkey not in aggregated_markets:
                    aggregated_markets[mkey] = []
                aggregated_markets[mkey].append(prop)

            # Build available lines dict for Selector
            available_lines = {}
            for mkey, props in aggregated_markets.items():
                m_type = mkey.replace('player_', '')
                # Use median line as representative
                lines = sorted([p['line'] for p in props])
                if lines:
                    available_lines[m_type] = lines[len(lines)//2]
            
            # CALL SELECTOR
            selection = self.market_selector.select_best_market(
                player_name, player_context, match_context, available_lines
            )
            
            if not selection:
                self.agent5.log_rejection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    rejection_stage='market_selector',
                    reason='Selector found no actionable market candidate for this player.',
                    player_trust_score=match_context.get('player_consensus', {}).get('trust_score'),
                )
                return []
            
            candidate_pool = selection.ranked_candidates[:1]
            if selection.score_gap_to_next <= 8 and len(selection.ranked_candidates) > 1:
                candidate_pool = selection.ranked_candidates[:2]

            approved_candidates = []
            rejected_candidates = []
            for candidate in candidate_pool:
                market_verdict = self.consensus_engine.evaluate_market_candidate(
                    player_id,
                    candidate['market_type'],
                    float(candidate.get('line') or 0.0),
                    reference_date=game_date,
                )
                candidate_record = dict(candidate)
                candidate_record['consensus_status'] = market_verdict.decision
                candidate_record['consensus_level'] = market_verdict.consensus_level
                candidate_record['consensus_reason'] = market_verdict.reasoning
                candidate_record['consensus_trust_score'] = market_verdict.trust_score
                candidate_record['consensus_trust_components'] = market_verdict.trust_components
                if market_verdict.decision == 'APPROVED':
                    approved_candidates.append(candidate_record)
                else:
                    rejected_candidates.append(candidate_record)

            selected_candidate = approved_candidates[0] if approved_candidates else None
            if not selected_candidate:
                top_reject = rejected_candidates[0] if rejected_candidates else {
                    'market_type': selection.market_type,
                    'consensus_reason': 'No market candidate reached approval.'
                }
                logger.info(
                    f"  [MARKET CONSENSUS REJECT] {player_name}: "
                    f"{top_reject['market_type']} -> {top_reject['consensus_reason']}"
                )
                self.agent5.log_rejection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    rejection_stage='market_consensus',
                    reason=top_reject['consensus_reason'],
                    proposed_market=top_reject.get('market_type'),
                    proposed_line=top_reject.get('line'),
                    consensus_level=top_reject.get('consensus_level'),
                    player_trust_score=match_context.get('player_consensus', {}).get('trust_score'),
                    market_trust_score=top_reject.get('consensus_trust_score'),
                    candidate_rank=top_reject.get('rank'),
                    validator_vote_summary=str(top_reject.get('consensus_trust_components', {})),
                )
                return []

            target_key = f"player_{selected_candidate['market_type']}"
            if target_key not in aggregated_markets:
                self.agent5.log_rejection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    rejection_stage='market_data_missing',
                    reason=f"Selected market {selected_candidate['market_type']} had no aggregated prop rows available.",
                    proposed_market=selected_candidate.get('market_type'),
                    proposed_line=selected_candidate.get('line'),
                    consensus_level=selected_candidate.get('consensus_level'),
                    player_trust_score=match_context.get('player_consensus', {}).get('trust_score'),
                    market_trust_score=selected_candidate.get('consensus_trust_score'),
                    candidate_rank=selected_candidate.get('rank'),
                    validator_vote_summary=str(selected_candidate.get('consensus_trust_components', {})),
                )
                return []

            aggregated_markets = {target_key: aggregated_markets[target_key]}

            match_context['market_consensus'] = {
                'status': selected_candidate['consensus_status'],
                'level': selected_candidate['consensus_level'],
                'reason': selected_candidate['consensus_reason'],
                'trust_score': selected_candidate.get('consensus_trust_score'),
                'trust_components': selected_candidate.get('consensus_trust_components', {}),
            }
            match_context['market_candidates'] = candidate_pool
            match_context['selection_reasoning'] = {
                'market': selected_candidate['market_type'],
                'score': selected_candidate['confidence'],
                'reasons': selected_candidate['reasoning'],
                'candidate_rank': selected_candidate['rank'],
                'score_gap_to_next': selection.score_gap_to_next,
                'candidate_count_considered': len(candidate_pool),
            }

            # ==========================================
            # LOOP THROUGH MARKETS (Now Single)
            # ==========================================
            
            for market_key, props in aggregated_markets.items():
                # if 'points' not in market_key:
                #     continue
                    
                # Find Median Line
                lines = sorted([p['line'] for p in props])
                if not lines: continue
                
                median_line = lines[len(lines)//2]
                
                # Use the prop that matches the median line (or first one)
                # We prioritize "finding" the one that matches median to get correct odds
                representative_prop = next((p for p in props if p['line'] == median_line), props[0])
                
                # Set loop variables
                market_type = market_key.replace('player_', '')
                line = median_line
                
                # Pass explicit odds from representative prop
                prop_odds = representative_prop.get('odds_over', -110)

                
                # Calculate H2H Avg using correct stat for this market
                h2h_hist = match_context.get('h2h_history', [])
                h2h_stat_map = {
                    'points': 'points', 'assists': 'assists', 'rebounds': 'rebounds',
                    'threes': 'fg3m', 'blocks': 'blk', 'steals': 'stl', 'field_goals': 'fgm'
                }
                h2h_key = h2h_stat_map.get(market_type, 'points')
                h2h_vals = [g.get(h2h_key, 0) for g in h2h_hist]
                h2h_avg = sum(h2h_vals)/len(h2h_vals) if h2h_vals else None

                if market_type in self.predictors:
                     # Use MarketPredictor (Direct ML) for ALL props (Points, Assists, Rebounds)
                     # SimulationEngine logic replaced by MarketPredictor for consistency and fix application
                     predictor = self.predictors[market_type]
                     
                     # Update Context
                     match_context['market_context'] = {
                        'line': line,
                        'floor': market_floor,
                        'ceiling': market_ceiling,
                        'odds': prop_odds
                     }
                     
                     raw_prediction = predictor.predict(
                        player_context, 
                        match_context,
                        market_line=line
                     )
                     
                     # Enrich with metadata
                     raw_prediction['market_type'] = market_type
                     raw_prediction['h2h_avg'] = h2h_avg
                     raw_prediction['raw_model_pred'] = raw_prediction.get('mean') # Alias
                     raw_prediction['model_provenance'] = self._build_model_provenance(
                        market_type=market_type,
                        player_context=player_context,
                        match_context=match_context,
                        raw_prediction=raw_prediction,
                        team_injuries=team_injury_map,
                        game_date=game_date,
                        opponent=opponent,
                     )
                     
                elif False: # market_type == 'points_legacy_sim':
                    # Use SimulationEngine (Monte Carlo) - DISABLED
                    pass



                else:
                    continue
                
                if raw_prediction is None:
                    logger.warning(f"  [SKIP] {player_name} {market_type}: Model returned None")
                    continue

                if market_type == 'points':
                    raw_prediction = self._apply_points_ensemble(raw_prediction)

                prediction_health = raw_prediction.get('prediction_health', {})
                fallback_only_markets = {'threes', 'blocks', 'steals', 'field_goals'}
                if (
                    prediction_health.get('used_fallback_model') and
                    market_type in fallback_only_markets
                ):
                    logger.info(
                        f"  [SKIP] {player_name} {market_type}: unsupported fallback-only market "
                        "until a trained model is available"
                    )
                    continue

                if (
                    'SYSTEM_SUPPRESS_FALLBACK_MODELS' in (calibration_flags or []) and
                    prediction_health.get('used_fallback_model')
                ):
                    logger.info(f"  [SKIP] {player_name} {market_type}: suppressed due to fallback-model policy")
                    continue
                
                # ===== AGENT 3: Auditing =====
                audited = self.agent3.audit(raw_prediction, player_context, match_context, market_type=market_type)
                
                # ===== CONTEXT ENRICHMENT (Pre-Edge) =====
                match_context['context_enrichment_failed'] = False
                match_context['context_enrichment_error'] = ''
                try:
                    # Minutes Projection
                    minutes_proj = self.minutes_projector.project_minutes(
                        player_context, match_context
                    )
                    match_context['minutes_projection'] = minutes_proj
                    
                    # Defensive Scheme Analysis
                    def_scheme = self.defensive_analyzer.analyze_defense(
                        opponent, game_date
                    )
                    match_context['defensive_scheme'] = def_scheme
                    
                    # Line Movement
                    market_key = f'player_{market_type}'
                    movement = self.line_tracker.analyze_movement(
                        player_name, market_key, game_date,
                        current_line=line or 0,
                        current_odds_over=prop_odds,
                        current_odds_under=representative_prop.get('odds_under', -110),
                        player_id=player_id,
                    )
                    match_context['line_movement'] = movement
                    
                    # Teammate Usage Network
                    missing = []
                    for inj_id, inj_prob in (team_injuries or {}).items():
                        if inj_prob < 0.5:  # Less than 50% chance to play
                            missing.append({'player_id': inj_id, 'player_name': f'ID_{inj_id}', 'status': 'OUT'})
                    if missing:
                        usage_impact = self.teammate_network.analyze_usage_impact(
                            player_id, player_name, team, missing, game_date
                        )
                        match_context['usage_impact'] = usage_impact
                        if 'minutes_projection' in match_context:
                            match_context['minutes_projection'] = self.minutes_projector.project_minutes(
                                player_context, match_context
                            )
                    
                    # Head-to-Head History (for NarrativeDetector)
                    h2h = self.agent1.get_player_vs_opponent_history(
                        player_id, opponent, limit=3
                    )
                    if h2h:
                        match_context['h2h_history'] = h2h
                    
                except Exception as e:
                    match_context['context_enrichment_failed'] = True
                    match_context['context_enrichment_error'] = str(e)
                    logger.warning(f"Context enrichment partial for {player_name}: {e}")
                
                # ===== EDGE LAYER: Score & Gate =====
                try:
                    edge_result = self.edge_scorer.evaluate(
                        audited, player_context, match_context,
                        market_line=line,
                        market_type=market_type,
                        market_floor=market_floor,
                        market_ceiling=market_ceiling,
                        odds_over=prop_odds,
                        odds_under=representative_prop.get('odds_under', -110)
                    )
                    
                    # Inject edge analysis into match_context for downstream agents
                    match_context['edge_analysis'] = edge_result
                    
                    # Log edge result
                    edge_tier = edge_result.get('tier', 'reject')
                    edge_score = edge_result.get('score', 0)
                    edge_dir = edge_result.get('direction', 'NO_BET')
                    logger.info(
                        f"    Edge: {player_name} {market_type} → "
                        f"score={edge_score:.0f} tier={edge_tier} dir={edge_dir}"
                    )
                    
                except Exception as e:
                    logger.warning(f"Edge scoring failed for {player_name}: {e}")
                    match_context['edge_analysis'] = {
                        'score': 0, 'tier': 'reject', 'direction': 'NO_BET',
                        'explanation': f'Edge scoring error: {e}',
                        'all_questions_answered': False
                    }
                
                # ===== AGENT 4: Calibration =====
                betting_decision = None
                if line is not None:
                    betting_decision = self.agent4.calibrate(
                        audited, line, player_name, player_context, match_context,
                        market_odds=prop_odds
                    )

                edge_snapshot = match_context.get('edge_analysis', {})
                candidate_tier = edge_snapshot.get('tier', 'reject')
                candidate_direction = edge_snapshot.get('direction', 'NO_BET')
                final_direction = betting_decision.direction if betting_decision else 'NO_BET'
                approved_bet = final_direction in ('OVER', 'UNDER')
                match_context['decision_alignment'] = {
                    'candidate_tier': candidate_tier,
                    'candidate_direction': candidate_direction,
                    'final_direction': final_direction,
                    'approved_bet': approved_bet,
                    'alignment_status': (
                        'approved_bet'
                        if approved_bet else
                        'candidate_only' if candidate_tier in ('parlay_core', 'playable') else
                        'rejected'
                    ),
                }
                match_context['decision_trace'] = {
                    'player_consensus_status': match_context.get('player_consensus', {}).get('status', 'UNKNOWN'),
                    'player_consensus_level': match_context.get('player_consensus', {}).get('level', ''),
                    'player_consensus_trust_score': match_context.get('player_consensus', {}).get('trust_score'),
                    'market_consensus_status': match_context.get('market_consensus', {}).get('status', 'UNKNOWN'),
                    'market_consensus_level': match_context.get('market_consensus', {}).get('level', ''),
                    'market_consensus_trust_score': match_context.get('market_consensus', {}).get('trust_score'),
                    'candidate_rank': match_context.get('selection_reasoning', {}).get('candidate_rank', 1),
                    'candidate_score_gap': match_context.get('selection_reasoning', {}).get('score_gap_to_next'),
                    'final_status': (
                        'BET'
                        if approved_bet else
                        'LEAN'
                        if candidate_tier in ('parlay_core', 'playable') else
                        'WATCH'
                    ),
                    'rejection_stage': (
                        ''
                        if approved_bet else
                        'calibrator'
                        if candidate_tier in ('parlay_core', 'playable') else
                        'edge_scorer'
                    ),
                    'final_decision_reason': (
                        ''
                        if approved_bet else
                        (betting_decision.blocker_reason if betting_decision else 'No betting decision returned')
                    ),
                    'context_enrichment_failed': match_context.get('context_enrichment_failed', False),
                    'context_enrichment_error': match_context.get('context_enrichment_error', ''),
                }

                if not approved_bet and candidate_tier in ('parlay_core', 'playable'):
                    edge_snapshot['candidate_tier'] = candidate_tier
                    edge_snapshot['candidate_direction'] = candidate_direction
                    edge_snapshot['tier'] = 'lean'
                    edge_snapshot['direction'] = 'NO_BET'
                    match_context['edge_analysis'] = edge_snapshot

                if (
                    betting_decision and
                    betting_decision.direction != 'NO_BET' and
                    match_context.get('edge_analysis', {}).get('tier') in ('parlay_core', 'playable')
                ):
                    edge_result = match_context.get('edge_analysis', {})
                    self.edge_tracker.log_edge_pick(
                        game_date=game_date,
                        player_name=player_name,
                        player_id=player_id,
                        market_type=market_type,
                        line=line or 0,
                        opening_line=match_context.get('line_movement', {}).get('opening_line'),
                        current_line=match_context.get('line_movement', {}).get('current_line', line),
                        direction=betting_decision.direction,
                        edge_score=edge_result.get('score', 0),
                        edge_tier=edge_result.get('tier', 'reject'),
                        kill_count=edge_result.get('sub_scores', {}).get('script_kills', 0)
                    )
                
                # ===== AGENT 5: Log =====
                self.agent5.log_prediction(
                    player_id, 
                    player_name, 
                    audited, 
                    match_context,
                    betting_decision=betting_decision
                )
                
                results.append(PredictionResult(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    player_context=player_context,
                    match_context=match_context,
                    raw_prediction=raw_prediction,
                    audited_prediction=audited,
                    audit_flags=audited.get('flags', []),
                    betting_decision=betting_decision,
                    timestamp=timestamp,
                    pipeline_success=True
                ))
            
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"Pipeline failed for player {player_id}: {e}")
            logger.error(traceback.format_exc())
            try:
                self.agent5.log_rejection(
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    rejection_stage='pipeline_error',
                    reason=str(e),
                )
            except Exception:
                pass
            return []
    
    def predict_game(self,
                     home_team: str,
                     away_team: str,
                     market_lines: Optional[Dict[int, Any]] = None,
                     market_odds: Optional[Dict[int, float]] = None,
                     calibration_flags: List[str] = None,
                     game_date: str = None) -> List[PredictionResult]:
        """
        Run predictions for all players in a game
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            market_lines: Optional dict mapping player_id to market line
            market_odds: Optional dict mapping player_id to american odds
            calibration_flags: List of system-wide calibration flags
            game_date: Date of the game (YYYY-MM-DD), defaults to today
            
        Returns:
            List of PredictionResults for all rotation players
        """
        results = []
        market_lines = market_lines or {}
        market_odds = market_odds or {}
        calibration_flags = calibration_flags or []
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        
        # Get Team IDs
        home_id = self._get_team_id(home_team)
        away_id = self._get_team_id(away_team)
        
        if not home_id or not away_id:
            logger.error(f"Could not find IDs for {home_team} or {away_team}")
            return []
            
        # [FIX] Fetch and map injuries for filtering
        from src.ingestion.injury_ingestion import InjuryIngestion
        injury_ingestion = InjuryIngestion(self.db)
        
        # Get all injuries for the target date (or today if not tracked historically by date correctly yet)
        # Assuming injury snapshots are daily, we use 'today' for latest status, or game_date if available
        # But usually we want the LATEST known injury status regardless of game date being in future
        # using 'today' is safer for "current status"
        # using 'today' is safer for "current status"
        all_injuries = injury_ingestion.get_todays_injuries() or []
        
        # Build map: {player_id: p_play}
        injury_map = {}
        # [FIX] Re-enabled injury filtering with News Agent support
        count_out = 0
        for inj in all_injuries:
            if inj['player_id']:
                injury_map[inj['player_id']] = inj['p_play']
                if inj['p_play'] < 0.5:
                    count_out += 1
        
        logger.info(f"  Loaded injury map with {len(injury_map)} players ({count_out} marked OUT/DOUBTFUL)")

        # Fetch ALL props for rotation players (Batch)
        all_props = self.db.get_player_props_for_date(game_date)
        
        # Group props by player
        props_by_player = {}
        for prop in all_props:
            pid = prop['player_id']
            if pid not in props_by_player: props_by_player[pid] = []
            props_by_player[pid].append(prop)
        
        # Determine rotation players
        rotation_players = []
        for team_abbr, team_id in [(home_team, home_id), (away_team, away_id)]:
            players = self.agent1.get_eligible_players(team_id, team_abbr, injuries=injury_map)
            rotation_players.extend(players)
        
        # [FALLBACK] If roster fetch failed (stats.nba.com timeout), derive players from odds props
        if not rotation_players and props_by_player:
            logger.info(f"  [FALLBACK] Roster fetch failed — using {len(props_by_player)} players from odds props")
            # Build team lookup: player_id -> team_abbr
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    player_ids_from_props = [pid for pid in props_by_player.keys() if pid is not None]
                    if player_ids_from_props:
                        placeholders = ','.join('?' * len(player_ids_from_props))
                        cursor.execute(f"""
                            SELECT p.player_id, p.full_name, t.abbreviation, p.position
                            FROM players p
                            LEFT JOIN teams t ON p.team_id = t.team_id
                            WHERE p.player_id IN ({placeholders})
                        """, player_ids_from_props)
                        
                        home_team_ids = set()
                        away_team_ids = set()
                        for row in cursor.fetchall():
                            pid, name, team_abbr_db, position = row
                            if team_abbr_db in (home_team, away_team):
                                rotation_players.append({
                                    'player_id': pid,
                                    'full_name': name,
                                    'team': team_abbr_db,
                                    'position': position or 'G'
                                })
                        logger.info(f"  [FALLBACK] Matched {len(rotation_players)} players to {home_team}/{away_team}")
            except Exception as e:
                logger.warning(f"  [FALLBACK] Failed to build player list from odds: {e}")
            
        player_ids = [p['player_id'] for p in rotation_players]
        
        # BATCH CALL to Agent 1
        logger.info(f"  Batch gathering context for {len(player_ids)} players in {home_team} vs {away_team}")
        if hasattr(self.agent1, 'gather_batch_player_contexts'):
             batch_contexts = self.agent1.gather_batch_player_contexts(player_ids, date_limit=game_date)
        else:
             # Fallback
             batch_contexts = {pid: self.agent1.gather_player_context(pid, date_limit=game_date) for pid in player_ids}

        # Predict for each
        team_rosters = {}
        for team_abbr in (home_team, away_team):
            team_rosters[team_abbr] = {
                p['player_id'] for p in rotation_players if p.get('team') == team_abbr
            }

        for player in rotation_players:
            player_id = player['player_id']
            player_team = player['team']
            opponent = away_team if player_team == home_team else home_team
            team_specific_injuries = {
                pid: prob for pid, prob in injury_map.items()
                if pid in team_rosters.get(player_team, set())
            }
            
            # Combine passed market_lines (legacy) with DB props
            active_props = props_by_player.get(player_id, [])
            
            # If explicit line passed in arg, prioritize it (add as Points prop)
            raw_line_data = market_lines.get(player_id)
            if raw_line_data:
                 # Logic to force specific line if testing
                 pass 
            
            # Skip if no props at all
            if not active_props and not raw_line_data:
                 continue

            odds = market_odds.get(player_id, -110)
            p_ctx = batch_contexts.get(player_id)
            
            result_list = self.predict_player(
                player_id, opponent, 
                reference_date=game_date,
                market_line=None, # Use active_props
                team_injuries=team_specific_injuries,
                market_odds=odds,
                calibration_flags=calibration_flags,
                preloaded_player_context=p_ctx,
                active_props=active_props
            )
            results.extend(result_list)

        logger.info(f"Generated {len(results)} predictions for {home_team} vs {away_team}")
        return results
        
    def _get_team_id(self, abbr: str) -> Optional[int]:
        """Look up team ID by abbreviation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT team_id FROM teams WHERE abbreviation = ?", (abbr,))
            row = cursor.fetchone()
            return row['team_id'] if row else None

    def _build_lineup_context(self,
                              player_context: Dict[str, Any],
                              team_injuries: Dict[int, float],
                              teammate_impact: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build unified lineup and role-shift context for downstream models."""
        inferred = player_context.get('inferred_signals', {}) or {}
        missing_probs = [prob for prob in (team_injuries or {}).values() if prob is not None]
        unavailable_count = sum(1 for prob in missing_probs if prob < 0.30)
        doubtful_count = sum(1 for prob in missing_probs if 0.30 <= prob < 0.75)
        usage_delta = float((teammate_impact or {}).get('total_usage_boost') or 0.0)
        minutes_delta = float((teammate_impact or {}).get('total_minutes_boost') or 0.0)
        assists_delta = float((teammate_impact or {}).get('total_assists_boost') or 0.0)
        rebounds_delta = float((teammate_impact or {}).get('total_rebounds_boost') or 0.0)
        role_change = bool(inferred.get('role_change', False))
        usage_spike = bool(inferred.get('usage_spike', False))
        rotation_tightening = bool(inferred.get('rotation_tightening', False))
        volatility_score = (
            (0.20 * unavailable_count) +
            (0.08 * doubtful_count) +
            min(0.35, abs(usage_delta) * 4.0) +
            min(0.25, abs(minutes_delta) / 8.0) +
            (0.12 if role_change else 0.0) +
            (0.08 if usage_spike else 0.0) +
            (0.08 if rotation_tightening else 0.0)
        )
        volatility_score = max(0.0, min(1.0, volatility_score))

        return {
            'player_role': str(player_context.get('player_role') or 'rotation'),
            'position': str(player_context.get('position') or 'G'),
            'missing_rotation_count': unavailable_count,
            'doubtful_rotation_count': doubtful_count,
            'usage_delta': round(usage_delta, 4),
            'minutes_delta': round(minutes_delta, 2),
            'assists_delta': round(assists_delta, 2),
            'rebounds_delta': round(rebounds_delta, 2),
            'role_change': role_change,
            'usage_spike': usage_spike,
            'rotation_tightening': rotation_tightening,
            'volatility_score': round(volatility_score, 2),
            'significant_absence_cluster': unavailable_count >= 2 or abs(minutes_delta) >= 2.5,
        }

    def _build_model_provenance(self,
                                market_type: str,
                                player_context: Dict[str, Any],
                                match_context: Dict[str, Any],
                                raw_prediction: Dict[str, Any],
                                team_injuries: Dict[int, float],
                                game_date: str,
                                opponent: str) -> Dict[str, Any]:
        """Track which models and enrichers contributed to a live pick."""
        provenance = {
            'core_predictor': 'market_predictor',
            'market_type': market_type,
            'mechanistic_reference_used': False,
            'mechanistic_reference_mean': None,
            'mechanistic_reference_std': None,
            'residual_model_ready': bool(getattr(self.ml_model, 'is_ready', False)),
            'residual_reference_adjustment': 0.0,
            'simulation_engine_loaded': bool(getattr(self.sim_engine, 'trainer', None)),
            'monte_carlo_reference_used': False,
            'monte_carlo_reference_mean': None,
            'monte_carlo_reference_p10': None,
            'monte_carlo_reference_p90': None,
            'injury_context_present': bool(match_context.get('injury_context_present', False)),
            'injury_context_size': int(match_context.get('injury_context_size', 0) or 0),
            'edge_scorer_active': True,
            'minutes_projector_active': True,
            'line_tracker_active': True,
            'teammate_network_active': bool(team_injuries),
            'defensive_scheme_active': True,
        }

        if market_type == 'points':
            try:
                mech = self.agent2.predict(player_context, match_context, team_injuries)
                provenance['mechanistic_reference_used'] = True
                provenance['mechanistic_reference_mean'] = mech.get('mean')
                provenance['mechanistic_reference_std'] = mech.get('std')
            except Exception as exc:
                provenance['mechanistic_reference_error'] = str(exc)
            try:
                team_abbr = player_context.get('team') or match_context.get('team')
                if team_abbr and opponent:
                    sim = self.sim_engine.simulate_player_points(
                        player_id=player_context['player_id'],
                        team_abbr=team_abbr,
                        opponent_abbr=opponent,
                        game_date=game_date,
                        p_play=float(player_context.get('p_play', 1.0) or 1.0),
                        team_injuries=team_injuries,
                        market_line=raw_prediction.get('market_line'),
                    )
                    provenance['monte_carlo_reference_used'] = True
                    provenance['monte_carlo_reference_mean'] = sim.get('mean')
                    provenance['monte_carlo_reference_p10'] = sim.get('p10')
                    provenance['monte_carlo_reference_p90'] = sim.get('p90')
            except Exception as exc:
                provenance['monte_carlo_reference_error'] = str(exc)

        if provenance['residual_model_ready']:
            try:
                features_dict = raw_prediction.get('features_used', {})
                provenance['residual_reference_adjustment'] = self.ml_model.predict_residual(features_dict)
            except Exception as exc:
                provenance['residual_reference_error'] = str(exc)

        return provenance

    def _apply_points_ensemble(self, raw_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Blend points references conservatively and log disagreement explicitly."""
        provenance = raw_prediction.get('model_provenance', {})
        core_mean = float(raw_prediction.get('mean') or 0.0)
        core_std = float(raw_prediction.get('std') or 1.0)
        health = raw_prediction.get('prediction_health', {})
        health_score = float(health.get('health_score', 1.0) or 1.0)
        used_fallback = bool(health.get('used_fallback_model'))

        mech_mean = provenance.get('mechanistic_reference_mean')
        mc_mean = provenance.get('monte_carlo_reference_mean')
        residual_adj = float(provenance.get('residual_reference_adjustment') or 0.0)
        ensemble_mean = core_mean
        ensemble_applied = False
        ensemble_notes: List[str] = []
        disagreement = None
        reference_means: List[float] = []

        if mech_mean is not None:
            mech_mean = float(mech_mean)
            reference_means.append(mech_mean)
            disagreement = abs(core_mean - mech_mean)
            provenance['model_disagreement'] = disagreement

            if not used_fallback and health_score >= 0.80 and disagreement <= 4.0:
                ensemble_mean = (0.70 * core_mean) + (0.30 * mech_mean)
                ensemble_applied = True
                ensemble_notes.append('blended_with_mechanistic_reference')
            elif not used_fallback and health_score >= 0.72 and disagreement <= 7.0:
                ensemble_mean = (0.82 * core_mean) + (0.18 * mech_mean)
                ensemble_applied = True
                ensemble_notes.append('light_blend_with_mechanistic_reference')
            else:
                ensemble_notes.append('mechanistic_reference_logged_only')
                if disagreement > 7.0:
                    ensemble_notes.append('high_model_disagreement')
        else:
            ensemble_notes.append('mechanistic_reference_unavailable')

        if mc_mean is not None:
            mc_mean = float(mc_mean)
            reference_means.append(mc_mean)
            mc_disagreement = abs(core_mean - mc_mean)
            if disagreement is None:
                disagreement = mc_disagreement
            else:
                disagreement = max(disagreement, mc_disagreement)

            if not used_fallback and health_score >= 0.78 and mc_disagreement <= 4.5:
                blend_base = ensemble_mean if ensemble_applied else core_mean
                ensemble_mean = (0.80 * blend_base) + (0.20 * mc_mean)
                ensemble_applied = True
                ensemble_notes.append('blended_with_monte_carlo_reference')
            elif not used_fallback and health_score >= 0.70 and mc_disagreement <= 7.0:
                blend_base = ensemble_mean if ensemble_applied else core_mean
                ensemble_mean = (0.88 * blend_base) + (0.12 * mc_mean)
                ensemble_applied = True
                ensemble_notes.append('light_blend_with_monte_carlo_reference')
            else:
                ensemble_notes.append('monte_carlo_reference_logged_only')

        if provenance.get('residual_model_ready'):
            residual_weight = 0.0
            if not used_fallback and health_score >= 0.75 and (disagreement is None or disagreement <= 5.0):
                residual_weight = 0.25
            elif not used_fallback and health_score >= 0.65 and (disagreement is None or disagreement <= 3.0):
                residual_weight = 0.15

            if residual_weight > 0:
                capped_adjustment = max(-2.0, min(2.0, residual_adj))
                ensemble_mean += capped_adjustment * residual_weight
                ensemble_applied = True
                ensemble_notes.append('residual_adjustment_applied')
            else:
                ensemble_notes.append('residual_logged_only')

        delta = ensemble_mean - core_mean
        if abs(delta) > 0.001:
            raw_prediction['mean'] = ensemble_mean
            raw_prediction['p10'] = float(raw_prediction.get('p10', core_mean - 1.28 * core_std) + delta)
            raw_prediction['p25'] = float(raw_prediction.get('p25', core_mean - 0.67 * core_std) + delta)
            raw_prediction['p50'] = float(raw_prediction.get('p50', core_mean) + delta)
            raw_prediction['p75'] = float(raw_prediction.get('p75', core_mean + 0.67 * core_std) + delta)
            raw_prediction['p90'] = float(raw_prediction.get('p90', core_mean + 1.28 * core_std) + delta)
            if 'samples' in raw_prediction:
                samples = raw_prediction['samples']
                raw_prediction['samples'] = np.clip(samples + delta, 0, None)

        mc_p10 = provenance.get('monte_carlo_reference_p10')
        mc_p90 = provenance.get('monte_carlo_reference_p90')
        if mc_p10 is not None and mc_p90 is not None and health_score >= 0.72:
            raw_prediction['p10'] = float((0.85 * raw_prediction.get('p10', ensemble_mean - 1.28 * core_std)) + (0.15 * float(mc_p10)))
            raw_prediction['p90'] = float((0.85 * raw_prediction.get('p90', ensemble_mean + 1.28 * core_std)) + (0.15 * float(mc_p90)))
            raw_prediction['p50'] = float(ensemble_mean)
            ensemble_notes.append('quantiles_informed_by_monte_carlo')

        provenance['ensemble_applied'] = ensemble_applied
        provenance['ensemble_mean'] = ensemble_mean
        provenance['core_predictor_mean'] = core_mean
        if reference_means:
            provenance['reference_consensus_mean'] = round(float(np.mean(reference_means)), 2)
        provenance['model_disagreement'] = round(disagreement, 2) if disagreement is not None else None
        provenance['ensemble_notes'] = ensemble_notes
        health['reference_disagreement'] = round(disagreement, 2) if disagreement is not None else None
        health['ensemble_applied'] = ensemble_applied
        health['ensemble_notes'] = ensemble_notes
        raw_prediction['prediction_health'] = health
        raw_prediction['model_provenance'] = provenance
        raw_prediction['ensemble_mean'] = ensemble_mean
        raw_prediction['ensemble_applied'] = int(ensemble_applied)
        return raw_prediction
    
    def predict_todays_slate(self,
                             market_lines: Optional[Dict[int, float]] = None,
                             game_date: str = None) -> List[PredictionResult]:
        """
        Run predictions for all games on a specific date (default: today)
        
        Returns:
            List of all PredictionResults for the slate
        """
        all_results = []
        
        # Get games for date
        games = self._get_games_for_date(game_date)
        
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            results = self.predict_game(home_team, away_team, market_lines)
            all_results.extend(results)
        
        logger.info(f"Generated {len(all_results)} total predictions for today's slate")
        return all_results
    
    def run_learning_cycle(self, game_date: str) -> Dict[str, Any]:
        """
        Run post-game learning for a completed game date
        
        Args:
            game_date: Date in YYYY-MM-DD format
            
        Returns:
            Learning report
        """
        logger.info(f"Running learning cycle for {game_date}")
        report = self.agent5.generate_daily_report(game_date)
        
        # Log summary
        for line in report.get('summary', []):
            logger.info(f"  {line}")
        
        return report
    
    def _get_player_info(self, player_id: int) -> tuple:
        """Get player name and team (try logs first, then players table)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Try logs first (most recent team)
            cursor.execute('''
                SELECT p.full_name, pl.team_abbreviation
                FROM player_logs pl
                JOIN players p ON pl.player_id = p.player_id
                WHERE pl.player_id = ?
                ORDER BY pl.game_date DESC
                LIMIT 1
            ''', (player_id,))
            row = cursor.fetchone()
            
            if row and row['team_abbreviation']:
                return row['full_name'] or 'Unknown', row['team_abbreviation']
                
            # Fallback to players table
            cursor.execute('''
                SELECT full_name, team_abbreviation 
                FROM players 
                WHERE player_id = ?
            ''', (player_id,))
            row = cursor.fetchone()
            
            if row and row['team_abbreviation']:
                return row['full_name'] or 'Unknown', row['team_abbreviation']
                
        return 'Unknown', 'UNK'
    
    def _get_games_for_date(self, game_date: str = None) -> List[Dict]:
        """Get list of games scheduled for a specific date"""
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to get from schedule or odds tables
            cursor.execute('''
                SELECT DISTINCT 
                    home_team,
                    away_team
                FROM odds_snapshots
                WHERE game_date = ?
            ''', (game_date,))
            
            rows = cursor.fetchall()
        
        return [{'home_team': r['home_team'], 'away_team': r['away_team']} for r in rows]
    
    def format_predictions_table(self, results: List[PredictionResult]) -> str:
        """Format predictions as a readable table"""
        lines = [
            f"{'Player':<25} {'Mkt':<4} {'Tm':<3} {'Vs':<3} {'Mean':>5} {'Std':>4} {'P10':>4} {'P90':>4} {'Conf':<6}"
        ]
        lines.append("-" * 75)
        
        for r in results:
            if not r.pipeline_success:
                continue
                
            ap = r.audited_prediction
            conf = ap.get('confidence', 'unk')
            market = r.raw_prediction.get('market_type', 'pts')[:3].upper()
            
            lines.append(
                f"{r.player_name[:24]:<25} {market:<4} {r.team:<3} {r.opponent:<3} "
                f"{ap.get('mean', 0):>5.1f} {ap.get('std', 0):>4.1f} "
                f"{ap.get('p10', 0):>4.1f} {ap.get('p90', 0):>4.1f} {conf:<6}"
            )
        
        return "\n".join(lines)
    
    def format_betting_table(self, results: List[PredictionResult]) -> str:
        """
        Format betting decisions as a detailed table (Step 4).
        Includes: Player, Market, Team, Matchup, Model, Line, Edge, Conf, Status, Reason.
        """
        # Sort by edge magnitude (descending)
        sorted_results = sorted(
            [r for r in results if r.betting_decision], 
            key=lambda x: max(x.betting_decision.edge_over, x.betting_decision.edge_under), 
            reverse=True
        )
        
        if not sorted_results:
            return "No betting decisions generated."
        
        lines = [
            f"{'Player':<20} {'Mkt':<4} {'Tm':<3} {'Vs':<3} {'Model ± σ':<10} {'Line':<5} {'Edge':<6} {'Conf':<6} {'Status':<8} {'Reason'}"
        ]
        lines.append("-" * 115)
        
        count = 0
        for r in sorted_results:
            bd = r.betting_decision
            edge = max(bd.edge_over, bd.edge_under)
            market = r.raw_prediction.get('market_type', 'pts')[:3].upper()
            
            # Determine Status
            if bd.direction != 'NO_BET':
                status = f"BET {bd.direction[:1]}" # BET O / BET U
            elif edge > 0.02:
                status = "WATCH"
            else:
                status = "NO PLAY"
                if edge <= 0.0: continue

            # Formatting
            model_str = f"{bd.model_mean:.1f}±{bd.model_std:.1f}"
            edge_str = f"{edge*100:.1f}%"
            
            # Get primary reason
            reason = bd.edge_source
            if len(reason) > 25: reason = reason[:22] + "..."
            
            lines.append(
                f"{r.player_name[:19]:<20} {market:<4} {r.team:<3} {r.opponent:<3} "
                f"{model_str:<10} {bd.line:<5} {edge_str:<6} {bd.confidence:<6} {status:<8} {reason}"
            )
            count += 1
            
        if count == 0:
            return "No actionable bets or near-misses found."
        
        return "\n".join(lines)


# Entry point for daily predictions
if __name__ == '__main__':
    orchestrator = PredictionOrchestrator()
    
    # Example: Predict for a specific player
    # result = orchestrator.predict_player(player_id=203999, opponent='BOS', market_line=27.5)
    
    # Example: Run today's slate
    # results = orchestrator.predict_todays_slate()
    # print(orchestrator.format_predictions_table(results))
    
    print("Orchestrator ready. Use orchestrator.predict_player() or orchestrator.predict_todays_slate()")
