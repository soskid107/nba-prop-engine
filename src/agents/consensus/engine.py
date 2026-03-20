
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .validator import ValidatorAgent, Vote
from .narrative_validator import NarrativeValidator
from .data_validator import DataValidator
from ...utils.database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class ConsensusVerdict:
    player_id: int
    prop_type: str
    decision: str     # 'APPROVED' or 'REJECTED'
    consensus_level: str # 'UNANIMOUS', 'MAJORITY', 'SPLIT', 'VETO'
    affirmative_votes: int
    total_votes: int
    votes: List[Vote]
    reasoning: str
    trust_score: float
    trust_components: Dict[str, Any]

class ConsensusEngine:
    """
    The Trust Layer.
    Aggregates votes from Validator Agents and issues a final verdict using Optimistic Democracy.
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.validators: List[ValidatorAgent] = [
            DataValidator(db),      # The Math
            NarrativeValidator(db), # The Context
            # MarketValidator(db)   # The Price (Future Phase)
        ]

    def _collect_votes(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> Tuple[List[Vote], int, int, int, Optional[str]]:
        votes: List[Vote] = []
        yes_votes = 0
        no_votes = 0
        abstains = 0
        veto_reason = None

        for validator in self.validators:
            try:
                vote = validator.validate(player_id, prop_type, line, reference_date=reference_date)
                votes.append(vote)

                if vote.verdict == "YES":
                    yes_votes += 1
                elif vote.verdict == "NO":
                    no_votes += 1
                    if vote.confidence >= 0.9:
                        veto_reason = f"VETO by {vote.agent_name}: {vote.reason}"
                else:
                    abstains += 1
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")

        return votes, yes_votes, no_votes, abstains, veto_reason

    def _calculate_trust_score(self,
                               votes: List[Vote],
                               yes_votes: int,
                               no_votes: int,
                               abstains: int,
                               veto_reason: Optional[str]) -> Tuple[float, Dict[str, Any]]:
        """Convert validator votes into a weighted 0-100 trust score."""
        score = 50.0
        components: Dict[str, Any] = {
            'data_vote': None,
            'narrative_vote': None,
            'yes_votes': yes_votes,
            'no_votes': no_votes,
            'abstains': abstains,
            'veto': bool(veto_reason),
        }

        if veto_reason:
            return 0.0, components

        for vote in votes:
            entry = {
                'verdict': vote.verdict,
                'confidence': vote.confidence,
                'reason': vote.reason,
            }
            if vote.agent_name == "DataValidator":
                components['data_vote'] = entry
                if vote.verdict == "YES":
                    score += 30.0 * vote.confidence
                elif vote.verdict == "NO":
                    score -= 35.0 * vote.confidence
                else:
                    score -= 10.0
            elif vote.agent_name == "NarrativeValidator":
                components['narrative_vote'] = entry
                if vote.verdict == "YES":
                    score += 20.0 * vote.confidence
                elif vote.verdict == "NO":
                    score -= 25.0 * vote.confidence
                else:
                    score -= 5.0

        if yes_votes > 0 and no_votes > 0:
            score -= 10.0
        if abstains > 1:
            score -= 5.0 * (abstains - 1)

        score = max(0.0, min(100.0, score))
        components['score_band'] = (
            'elite' if score >= 80 else
            'strong' if score >= 65 else
            'fragile' if score >= 45 else
            'weak'
        )
        return score, components

    def _build_verdict(self,
                       player_id: int,
                       prop_type: str,
                       decision: str,
                       consensus_level: str,
                       yes_votes: int,
                       votes: List[Vote],
                       reasoning: str,
                       no_votes: int,
                       abstains: int,
                       veto_reason: Optional[str]) -> ConsensusVerdict:
        trust_score, trust_components = self._calculate_trust_score(
            votes, yes_votes, no_votes, abstains, veto_reason
        )
        return ConsensusVerdict(
            player_id=player_id,
            prop_type=prop_type,
            decision=decision,
            consensus_level=consensus_level,
            affirmative_votes=yes_votes,
            total_votes=len(votes),
            votes=votes,
            reasoning=reasoning,
            trust_score=trust_score,
            trust_components=trust_components,
        )

    def evaluate_player_viability(self, player_id: int, reference_date: str = None) -> ConsensusVerdict:
        """Evaluate whether the player is viable to model at all."""
        return self.evaluate_proposal(player_id, 'status_check', 0.0, reference_date=reference_date)

    def evaluate_market_candidate(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> ConsensusVerdict:
        """Evaluate whether a proposed market/line combination is trustworthy enough to model."""
        votes, yes_votes, no_votes, abstains, veto_reason = self._collect_votes(player_id, prop_type, line, reference_date=reference_date)
        trust_score, _ = self._calculate_trust_score(votes, yes_votes, no_votes, abstains, veto_reason)

        if veto_reason:
            return self._build_verdict(player_id, prop_type, "REJECTED", "VETO", yes_votes, votes, veto_reason, no_votes, abstains, veto_reason)

        data_vote = next((vote for vote in votes if vote.agent_name == "DataValidator"), None)
        if data_vote is None:
            return self._build_verdict(player_id, prop_type, "REJECTED", "NO_DATA_VALIDATOR", yes_votes, votes, "Data validator unavailable.", no_votes, abstains, veto_reason)

        # For market viability, a directional "NO" from the data validator still means
        # the line is modelable; it often just indicates an UNDER lean rather than an
        # invalid market. Only true abstains/no-data should block the market outright.
        if data_vote.verdict == "ABSTAIN":
            if yes_votes > no_votes and trust_score >= 52:
                return self._build_verdict(
                    player_id,
                    prop_type,
                    "APPROVED",
                    "CAUTIOUS_MAJORITY",
                    yes_votes,
                    votes,
                    f"Data support is thin, but the market remains actionable ({yes_votes} vs {no_votes}).",
                    no_votes,
                    abstains,
                    veto_reason,
                )
            reason = data_vote.reason or "Data validator did not affirm the market."
            return self._build_verdict(player_id, prop_type, "REJECTED", "NO_DATA_SUPPORT", yes_votes, votes, reason, no_votes, abstains, veto_reason)

        if yes_votes > 0 and no_votes > 0:
            return self._build_verdict(
                player_id,
                prop_type,
                "APPROVED",
                "MIXED_DIRECTIONAL",
                yes_votes,
                votes,
                f"Validators disagree on direction, but the market remains actionable ({yes_votes} vs {no_votes}).",
                no_votes,
                abstains,
                veto_reason,
            )

        if yes_votes >= 2 and no_votes == 0:
            return self._build_verdict(player_id, prop_type, "APPROVED", "UNANIMOUS", yes_votes, votes, "Data and narrative validators agree.", no_votes, abstains, veto_reason)

        if yes_votes > no_votes:
            return self._build_verdict(player_id, prop_type, "APPROVED", "MAJORITY", yes_votes, votes, f"Data validator supports market ({yes_votes} vs {no_votes}).", no_votes, abstains, veto_reason)

        if no_votes > 0 and yes_votes == 0:
            return self._build_verdict(
                player_id,
                prop_type,
                "APPROVED",
                "DIRECTIONAL_ONLY",
                yes_votes,
                votes,
                "Data validator found a directional edge signal, even without narrative support.",
                no_votes,
                abstains,
                veto_reason,
            )

        return self._build_verdict(player_id, prop_type, "REJECTED", "SPLIT_OR_NEGATIVE", yes_votes, votes, f"Insufficient support ({yes_votes} vs {no_votes})", no_votes, abstains, veto_reason)
        
    def evaluate_proposal(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> ConsensusVerdict:
        """
        Run the democratic process on a betting proposal.
        """
        votes, yes_votes, no_votes, abstains, veto_reason = self._collect_votes(player_id, prop_type, line, reference_date=reference_date)
                
        # Consensus Logic
        total_active_votes = yes_votes + no_votes
        
        # 1. Immediate Veto Check
        if veto_reason:
            return self._build_verdict(player_id, prop_type, "REJECTED", "VETO", yes_votes, votes, veto_reason, no_votes, abstains, veto_reason)
            
        # 2. Majority Rule (Optimistic Democracy)
        # If no active votes (all abstain), we default to REJECT (Safety)
        if total_active_votes == 0:
             return self._build_verdict(player_id, prop_type, "REJECTED", "NO_QUORUM", 0, votes, "No active votes cast.", no_votes, abstains, veto_reason)
             
        # If Unanimous YES
        if yes_votes == total_active_votes and yes_votes > 0:
             return self._build_verdict(player_id, prop_type, "APPROVED", "UNANIMOUS", yes_votes, votes, "All validators in agreement.", no_votes, abstains, veto_reason)
             
        # If Majority YES (e.g. 2 vs 1)
        if yes_votes > no_votes:
             return self._build_verdict(player_id, prop_type, "APPROVED", "MAJORITY", yes_votes, votes, f"Majority Support ({yes_votes} vs {no_votes})", no_votes, abstains, veto_reason)
             
        # Default Reject
        return self._build_verdict(player_id, prop_type, "REJECTED", "SPLIT_OR_NEGATIVE", yes_votes, votes, f"Insufficient support ({yes_votes} vs {no_votes})", no_votes, abstains, veto_reason)
