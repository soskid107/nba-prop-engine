import sqlite3
import pandas as pd

conn = sqlite3.connect('nba_props.db')
df = pd.read_sql("""
    SELECT 
        endpoint as Endpoint, 
        status_code as Status, 
        COUNT(*) as Count, 
        MAX(timestamp) as Last_Seen
    FROM api_calls 
    WHERE api_name='the_odds_api' 
    GROUP BY endpoint, status_code 
    ORDER BY MAX(timestamp) DESC
    LIMIT 20
""", conn)

print("\n--- Odds API Calls Summary ---")
print(df.to_string(index=False))
