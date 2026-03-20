
import sys
import os
from pathlib import Path

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.consensus.engine import ConsensusEngine
from src.utils.database import DatabaseManager

def test_consensus():
    db = DatabaseManager()
    engine = ConsensusEngine(db)
    
    # Test Case 1: Random Active Player (Should ideally get "Majority" or "Split")
    # We need a player with recent games for DataValidator to work
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, team_abbreviation FROM player_logs ORDER BY game_date DESC LIMIT 1 OFFSET 5") 
        # Offset 5 to get someone interesting
        row = cursor.fetchone()
        
    if not row:
        print("No players found in DB.")
        return

    player_id = row['player_id']
    
    # Test Prop: Points, Line: 15.5
    prop_type = 'points'
    line = 15.5
    
    print(f"\n[Test] Running Consensus Engine for Player ID {player_id} on {prop_type} > {line}...")
    
    verdict = engine.evaluate_proposal(player_id, prop_type, line)
    
    print(f"\n--- Consensus Verdict ---")
    print(f"Decision: {verdict.decision}")
    print(f"Level: {verdict.consensus_level}")
    print(f"Vote Count: {verdict.affirmative_votes} / {verdict.total_votes}")
    print(f"Reasoning: {verdict.reasoning}")
    
    print(f"\n--- Individual Votes ---")
    for vote in verdict.votes:
        print(f"[{vote.agent_name}] {vote.verdict}: {vote.reason}")

def test_orchestrator():
    from src.agents.orchestrator import PredictionOrchestrator
    
    print("\n[Test] Initializing Orchestrator with Consensus Engine...")
    orch = PredictionOrchestrator()
    
    # Check if consensus engine is loaded
    if hasattr(orch, 'consensus_engine'):
        print("PASS: Consensus Engine loaded in Orchestrator.")
    else:
        print("FAIL: Consensus Engine NOT found.")
        return

    # Mock DB or use real one
    # We will just run a simple check to see if it doesn't crash
    # finding a player
    with orch.db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id FROM player_logs ORDER BY game_date DESC LIMIT 1")
        pid = cursor.fetchone()['player_id']
        
    print(f"[Test] running predict_player for {pid} (Dry Run)...")
    try:
        # We expect it might return [] if rejected, or a list if approved
        # This tests the flow through evaluate_proposal
        res = orch.predict_player(pid, '2026-02-16')
        print(f"Result type: {type(res)}")
        print("PASS: predict_player executed without crash.")
    except Exception as e:
        print(f"FAIL: predict_player crashed: {e}")

if __name__ == "__main__":
    test_consensus()
    test_orchestrator()
