import sqlite3
import pandas as pd

db_path = "data/nba_props.db"
conn = sqlite3.connect(db_path)

print("--- RECENT ODDS API USAGE (LAST 50 CALLS) ---")
df = pd.read_sql("""
    SELECT 
        call_date, 
        call_time, 
        endpoint, 
        response_status, 
        cached
    FROM api_usage 
    WHERE api_name = 'the_odds_api'
    ORDER BY call_date DESC, call_time DESC
    LIMIT 50
""", conn)

print(df.to_string(index=False))

print("\n--- STATUS SUMMARY FOR MARCH ---")
df_mar = pd.read_sql("""
    SELECT 
        response_status, 
        COUNT(*) as count
    FROM api_usage 
    WHERE api_name = 'the_odds_api' AND call_date >= '2026-03-01'
    GROUP BY response_status
""", conn)
print(df_mar.to_string(index=False))

conn.close()
