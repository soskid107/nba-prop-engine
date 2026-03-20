
import sys
import os
from pathlib import Path

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.context_synthesizer import ContextSynthesizerAgent
from src.utils.database import DatabaseManager

def test_synthesize():
    db = DatabaseManager()
    agent = ContextSynthesizerAgent(db)
    
    # Get a player who played recently to test
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, team_abbreviation FROM player_logs ORDER BY game_date DESC LIMIT 1")
        row = cursor.fetchone()
        
    if not row:
        print("No players found in DB.")
        return

    player_id = row['player_id']
    team = row['team_abbreviation']
    opponent = 'BOS' # Dummy opponent
    
    print(f"\n[Test] Synthesizing Brief for Player ID {player_id} ({team} vs {opponent})...")
    
    brief = agent.build_player_brief(player_id, opponent)
    
    if brief:
        print(f"\n--- Player Brief: {brief.player_name} ---")
        print(f"Stats (L5 PPG): {brief.stats.get('points_L5'):.1f}")
        print(f"Injury Status: {brief.injury_status}")
        print(f"News: {brief.news_headline}")
        print(f"Narrative Score: {brief.narrative_score}")
        print(f"Narrative Notes: {brief.narrative_notes}")
    else:
        print("Failed to build brief.")

if __name__ == "__main__":
    test_synthesize()
