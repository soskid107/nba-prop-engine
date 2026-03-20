
import logging
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
from src.utils.database import DatabaseManager
from src.agents.orchestrator import PredictionOrchestrator
from src.learning.feature_store import FeatureStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATA_GEN")

def generate_data(n_samples=500):
    db = DatabaseManager()
    orch = PredictionOrchestrator()
    fs = FeatureStore()
    
    training_rows = []
    
    logger.info(f"Starting Backtest Loop for {n_samples} samples...")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Get pool of game logs (valid games with >0 minutes)
        # We limit to known recent seasons (2024-2026) to ensure data quality
        cursor.execute("""
            SELECT player_id, game_date, points, team_abbreviation, minutes, 
                   opponent_abbreviation, is_home
            FROM player_logs 
            WHERE minutes > 10
            AND game_date > '2023-10-01'
            ORDER BY RANDOM()
            LIMIT ?
        """, (n_samples,))
        
        games = cursor.fetchall()
        
    logger.info(f"Pool selected: {len(games)} games. Processing...")
    
    for game in tqdm(games):
        pid = game['player_id']
        date = game['game_date']
        actual_points = game['points']
        team = game['team_abbreviation']
        
        try:
            # 2. Replay Context (Time Travel)
            # CRITICAL: enforce date_limit to prevent leakage
            player_context = orch.agent1.gather_player_context(pid, date_limit=date)
            
            if not player_context:
                with open("gen_debug.log", "a") as f: f.write(f"Sample {pid}: No Player Context\n")
                continue
                
            # Simulate Match Context
            # [OPTIMIZATION] Use pre-fetched columns from player_logs!
            is_home = bool(game['is_home'])
            opponent = game['opponent_abbreviation']
            
            match_context = {
                'opponent': opponent,
                'is_home': is_home,
                'game_date': date,
                'blowout_probability': 0.1 # Default assumption
            }
            
            # 3. Run Mechanistic Model
            # Note: We assume healthy teammates for this simplified backtest 
            # (Fetching historical teammate injury status is complex and slow)
            result = orch.agent2.predict(player_context, match_context)
            
            mech_pred = result['mean']
            
            # 4. Extract Features
            features = fs.extract_features(player_context, match_context, mech_pred)
            
            # 5. Calculate Target (Residual)
            # Residual = ACTUAL - PREDICTED
            # If Actual=30, Pred=20 -> Residual +10 (Model under-predicted)
            residual = actual_points - mech_pred
            
            row = features.copy()
            row['target_residual'] = residual
            row['actual_points'] = actual_points
            row['player_id'] = pid
            
            training_rows.append(row)
            
        except Exception as e:
            # logger.warning(f"Failed sample {pid} on {date}: {e}")
            with open("gen_error_debug.log", "a") as f:
                f.write(f"Sample {pid} Failed: {str(e)}\n")
            continue
            
    # 6. Save
    df = pd.DataFrame(training_rows)
    if df.empty:
        logger.error("No training data generated!")
        with open("gen_status.txt", "w") as f: f.write("FAILED: No rows")
        return

    filename = 'training_data.csv'
    df.to_csv(filename, index=False)
    with open("gen_status.txt", "w") as f: f.write(f"SUCCESS: {len(df)} rows")
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(df)} samples to {filename}")
    
    # Quick Stats
    mae = (df['actual_points'] - df['mech_pred']).abs().mean()
    logger.info(f"Baseline Mechanistic MAE: {mae:.2f}")

if __name__ == "__main__":
    generate_data(n_samples=2000) # Production batch
