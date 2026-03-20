
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Bet:
    player_id: int
    player_name: str
    team: str
    opponent: str
    prop_type: str
    line: float
    prediction: float
    edge: float
    consensus_level: str # 'UNANIMOUS' or 'MAJORITY'
    units: float = 0.0
    rationale: str = ""

class PortfolioManagerAgent:
    """
    The 'Treasury' Agent.
    Responsibilities:
    1. Sizing: Assign unit sizes based on Edge + Consensus.
    2. Risk: Cap exposure per game/team.
    3. Correlation: Detect and reduce correlated bets (e.g. 5 OVERs on one team).
    """
    
    def __init__(self, bankroll: float = 1000.0, max_game_exposure: float = 0.15):
        self.bankroll = bankroll
        self.max_game_exposure = max_game_exposure # Max 15% of bankroll on one game

    def optimize_portfolio(self, raw_bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Take raw favorable predictons and turn them into a sized portfolio.
        Returns the original dicts with added 'units' and 'risk_rationale'.
        """
        approved_bets = []
        
        # 1. Initial Sizing based on Conviction
        for rb in raw_bets:
            bet_dict = self._size_individual_bet(rb)
            if bet_dict.get('units', 0) > 0:
                approved_bets.append(bet_dict)
                
        # 2. Correlation / Exposure Check
        final_bets = self._apply_risk_caps(approved_bets)
        
        return final_bets

    def _size_individual_bet(self, rb: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Base Unit Size."""
        # Extract Consensus Info
        consensus = rb.get('consensus_level', 'MAJORITY') 
        edge = rb.get('edge_score', 0)
        
        # Base Sizing Logic
        units = 1.0 # Standard Unit
        
        # Boost for Unanimous Consensus
        if consensus == 'UNANIMOUS':
            units *= 1.5
            
        # Boost for High Edge (using Kelly-lite logic)
        if edge > 10.0:
            units *= 1.2
            
        # Cap
        units = min(units, 2.0) # Max 2 units
        
        # Create output dict (copy of input + new fields)
        out = rb.copy()
        out['units'] = round(units, 2)
        out['risk_rationale'] = f"Edge {edge:.1f} | {consensus}"
        return out

    def _apply_risk_caps(self, bets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reduce size if too heavily exposed to one game."""
        game_exposure = defaultdict(float)
        
        # Group by Game (Team vs Opp)
        for bet in bets:
            # Safe access
            t = str(bet.get('team', 'UNK'))
            o = str(bet.get('opponent', 'UNK'))
            game_id = sorted([t, o])
            game_key = f"{game_id[0]}_{game_id[1]}"
            game_exposure[game_key] += bet['units']
            
        # Check against Max Exposure (e.g., 5 units max per game)
        MAX_UNITS_PER_GAME = 5.0
        
        accepted_bets = []
        for bet in bets:
            t = str(bet.get('team', 'UNK'))
            o = str(bet.get('opponent', 'UNK'))
            game_id = sorted([t, o])
            game_key = f"{game_id[0]}_{game_id[1]}"
            
            if game_exposure[game_key] > MAX_UNITS_PER_GAME:
                # Naive scaling: Reduce all bets in this game proportionally
                reduction_factor = MAX_UNITS_PER_GAME / game_exposure[game_key]
                bet['units'] = round(bet['units'] * reduction_factor, 2)
                bet['risk_rationale'] += f" (Risk Capped {reduction_factor:.1f}x)"
            
            accepted_bets.append(bet)
            
        return accepted_bets
