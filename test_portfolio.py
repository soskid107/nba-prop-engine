
import sys
import os
from pprint import pprint

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.portfolio_manager import PortfolioManagerAgent

def test_portfolio_sizing():
    pm = PortfolioManagerAgent()
    
    print("\n--- Test 1: Sizing Logic ---")
    bets = [
        # Case A: Standard
        {'player_id': 1, 'player_name': 'Standard Guy', 'team': 'A', 'opponent': 'B', 'edge_score': 5.0, 'consensus_level': 'MAJORITY'},
        # Case B: Unanimous
        {'player_id': 2, 'player_name': 'Unanimous Guy', 'team': 'C', 'opponent': 'D', 'edge_score': 5.0, 'consensus_level': 'UNANIMOUS'},
        # Case C: High Edge
        {'player_id': 3, 'player_name': 'Edge Guy', 'team': 'E', 'opponent': 'F', 'edge_score': 12.0, 'consensus_level': 'MAJORITY'},
        # Case D: Both
        {'player_id': 4, 'player_name': 'Perfekt Guy', 'team': 'G', 'opponent': 'H', 'edge_score': 12.0, 'consensus_level': 'UNANIMOUS'},
    ]
    
    optimized = pm.optimize_portfolio(bets)
    for b in optimized:
        print(f"Player: {b['player_name']:<15} | Units: {b['units']:<4} | Rationale: {b['risk_rationale']}")

def test_portfolio_risk_cap():
    pm = PortfolioManagerAgent()
    print("\n--- Test 2: Risk Cap (Max 5 Units Per Game) ---")
    
    # 5 bets on same game (Team A vs Team B), each 1.5 units (Total 7.5) -> Should be capped
    bets = []
    for i in range(5):
        bets.append({
            'player_id': 10+i, 
            'player_name': f'Player {i}', 
            'team': 'LAL', 
            'opponent': 'GSW', 
            'edge_score': 5.0, 
            'consensus_level': 'UNANIMOUS' # 1.5 units each
        })
        
    optimized = pm.optimize_portfolio(bets)
    
    total_units = sum(b['units'] for b in optimized)
    print(f"Total Units for LAL vs GSW: {total_units:.2f} (Expected ~5.0)")
    
    for b in optimized:
        print(f"Player: {b['player_name']:<15} | Units: {b['units']:<4} | Rationale: {b['risk_rationale']}")

if __name__ == "__main__":
    test_portfolio_sizing()
    test_portfolio_risk_cap()
