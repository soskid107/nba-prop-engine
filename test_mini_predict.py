
from src.utils.database import DatabaseManager
from src.agents.orchestrator import PredictionOrchestrator
from datetime import datetime

def mini_predict():
    db = DatabaseManager()
    orchestrator = PredictionOrchestrator()
    
    # Target a game known to exist in our sync (e.g. from 2026-03-07)
    # We'll just run one game
    print("Running mini-prediction to populate logs...")
    
    # Using dummy teams for which we have some data or just real ones if synced
    # From my test_sync, it synced 6 games.
    
    results = orchestrator.predict_game("BOS", "PHI", game_date="2026-03-07")
    print(f"Generated {len(results)} predictions.")
    
    # The orchestrator.predict_game should have called LearningLoopAgent.log_prediction
    # Let's verify prediction_log
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_log WHERE game_date = '2026-03-07'")
        count = cursor.fetchone()[0]
        print(f"Predictions in log: {count}")

if __name__ == "__main__":
    mini_predict()
