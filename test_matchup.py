
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.getcwd())

from src.utils.database import DatabaseManager
from src.agents.defensive_schemes import DefensiveSchemeAnalyzer
from src.agents.learning_loop import LearningLoopAgent
from src.models.matchup_model import MatchupModel

def test_matchup_pipeline():
    print("Testing Dynamic Matchup Model Pipeline...")
    db = DatabaseManager()
    
    # 1. Test Scheme Analyzer Trends
    print("\n[1] Testing Defensive Scheme Trends...")
    scheme_analyzer = DefensiveSchemeAnalyzer(db)
    # Pick a team that likely has stats, e.g., BOS
    analysis = scheme_analyzer.analyze_defense('BOS')
    print(f"  BOS Schemes: {analysis.get('schemes')}")
    print(f"  Raw Stats: {analysis.get('raw_stats')}")
    
    # 2. Test Bias Learning (Mock some data if needed, or run on existing)
    print("\n[2] Testing Scheme Bias Learning...")
    learner = LearningLoopAgent(db)
    # We'll try to run it. If no data, it returns empty, which is fine for "not crashing"
    biases = learner.analyze_scheme_bias(lookback_days=100)
    print(f"  Found {len(biases)} learned scheme biases.")
    if biases:
        for k, v in list(biases.items())[:3]:
            print(f"    - {k}: {v.mean_error:.2f} err ({v.bias_direction}) -> Adj: {v.recommended_adjustment:.2f}")
            
    # 3. Test Matchup Model Lookup
    print("\n[3] Testing Matchup Model...")
    mm = MatchupModel(db)
    # Mock a player ID and opponent
    # We need a player ID. Let's find one from DB.
    with db.get_connection() as conn:
        pid = conn.execute("SELECT player_id FROM players LIMIT 1").fetchone()
        if pid:
            pid = pid[0]
            print(f"  Testing with Player ID: {pid} vs BOS")
            result = mm.get_matchup_multiplier(pid, 'BOS', 'points')
            print(f"  Result: {result}")
        else:
            print("  [WARN] No players in DB to test.")

if __name__ == "__main__":
    test_matchup_pipeline()
