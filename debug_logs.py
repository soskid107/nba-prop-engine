import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('nba_props.db')
    
    # Check player_logs count
    c = conn.cursor()
    c.execute("SELECT count(*) FROM player_logs")
    print("Player Logs Count:", c.fetchone()[0])
    
    # Sample logs
    query = "SELECT * FROM player_logs LIMIT 5"
    df_logs = pd.read_sql_query(query, conn)
    print("Sample Logs:\n", df_logs)
    
    # Check if we can find Booker in logs by name? 
    # Usually logs have player_id. If players table is empty, we can't map ID to name easily without API.
    # But maybe logs have name column?
    print("Log Columns:", df_logs.columns.tolist())

except Exception as e:
    print(f"DB Error: {e}")
finally:
    if 'conn' in locals(): conn.close()
