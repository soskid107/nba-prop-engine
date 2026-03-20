
import sqlite3
import pandas as pd
from src.utils.database import DatabaseManager

def audit_player(player_name, db):
    print(f"\n{'='*50}")
    print(f"AUDITING: {player_name}")
    print(f"{'='*50}")
    
    with db.get_connection() as conn:
        # 1. Get stats from last 15 games
        query = """
        SELECT 
            game_date, 
            points, 
            minutes, 
            ppm, 
            is_home 
        FROM player_logs 
        WHERE player_id = (SELECT player_id FROM players WHERE full_name LIKE ? LIMIT 1)
        ORDER BY game_date DESC 
        LIMIT 15
        """
        try:
            df = pd.read_sql_query(query, conn, params=(f"%{player_name}%",))
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return

        if df.empty:
            print("No game logs found!")
            return

        print("\nLast 15 Games:")
        print(df.to_string(index=False))
        
        # 2. Calculate Averages
        print("\n--- Averages ---")
        l5 = df.head(5)
        l10 = df.head(10)
        
        print(f"L5 Pts: {l5['points'].mean():.1f} | Mins: {l5['minutes'].mean():.1f}")
        print(f"L10 Pts: {l10['points'].mean():.1f} | Mins: {l10['minutes'].mean():.1f}")
        print(f"Full ({len(df)}) Pts: {df['points'].mean():.1f} | Mins: {df['minutes'].mean():.1f}")

        # 3. Check for specific anomalies
        zeros = df[df['points'] == 0]
        if not zeros.empty:
            print(f"\n[WARNING] Found {len(zeros)} games with 0 points!")
            
        low_mins = df[df['minutes'] < 10]
        if not low_mins.empty:
            print(f"\n[WARNING] Found {len(low_mins)} games with < 10 minutes!")

def main():
    db = DatabaseManager()
    audit_player("Jalen Johnson", db)
    audit_player("Marcus Smart", db)
    # audit_player("Dyson Daniels", db)

if __name__ == "__main__":
    main()
