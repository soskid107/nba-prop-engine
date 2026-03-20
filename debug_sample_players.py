import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('nba_props.db')
    
    # List first 5 players
    query = "SELECT * FROM players LIMIT 5"
    df_p = pd.read_sql_query(query, conn)
    print("Sample Players:\n", df_p)
    
    # List columns explicitly
    c = conn.cursor()
    c.execute("PRAGMA table_info(players)")
    print("\nColumns:", [row[1] for row in c.fetchall()])

except Exception as e:
    print(f"DB Error: {e}")
finally:
    if 'conn' in locals(): conn.close()
