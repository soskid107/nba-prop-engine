import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('nba_props.db')
    
    # Correct query using last_name
    query = "SELECT * FROM players WHERE last_name LIKE '%Booker%'"
    df_p = pd.read_sql_query(query, conn)
    print("Players found:\n", df_p[['player_id', 'first_name', 'last_name', 'team_id']])
    
    if not df_p.empty:
        pid = df_p.iloc[0]['player_id']
        print(f"\nChecking logs for Player ID: {pid} (Devin Booker)")
        
        # Check recent logs
        q_logs = f"SELECT * FROM player_logs WHERE player_id = {pid} ORDER BY game_date DESC LIMIT 10"
        df_logs = pd.read_sql_query(q_logs, conn)
        
        if not df_logs.empty:
            cols = df_logs.columns.tolist()
            # print("Columns:", cols)
            pts = 'pts' if 'pts' in cols else 'points'
            mins = 'min' if 'min' in cols else 'minutes'
            
            print(f"\nRecent Games (Last 5):")
            print(df_logs.head(5)[['game_date', pts, mins, 'matchup']])
            
            l5 = df_logs.head(5)[pts].mean()
            l10 = df_logs.head(10)[pts].mean()
            print(f"\nCalculated L5 Points: {l5}")
            print(f"Calculated L10 Points: {l10}")
            
            # Check market line cap logic
            market_line = 23.5
            cap = l5 * 1.25
            print(f"\nL5 Cap (Old Logic): {cap}")
            print(f"L5 Cap (New Relaxed Logic - Max(Cap, Line*1.15)): {max(cap, market_line * 1.15)}")
            
        else:
            print("No logs found for Booker.")

except Exception as e:
    print(f"DB Error: {e}")
finally:
    if 'conn' in locals(): conn.close()
