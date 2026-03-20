import sqlite3
import os

dbs = ["nba_props.db", "data/nba_props.db"]

for db_path in dbs:
    if not os.path.exists(db_path):
        print(f"--- {db_path} does not exist ---")
        continue
        
    print(f"--- Analyzing {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        print(f"Tables: {', '.join(tables)}")
        
        if 'api_usage' in tables:
            cursor.execute("SELECT MAX(call_date) as last_date, COUNT(*) as count FROM api_usage WHERE api_name='the_odds_api'")
            res = cursor.fetchone()
            print(f"  the_odds_api: Last call on {res['last_date']}, Total calls: {res['count']}")
            
            cursor.execute("SELECT call_date, response_status, COUNT(*) FROM api_usage WHERE api_name='the_odds_api' AND call_date >= '2026-03-01' GROUP BY call_date, response_status")
            mar_calls = cursor.fetchall()
            if mar_calls:
                print("  March Calls:")
                for c in mar_calls:
                    print(f"    {c[0]} | Status {c[1]} | Count {c[2]}")
            else:
                print("  No March calls found in api_usage.")
        
        if 'player_prop_odds' in tables:
             cursor.execute("SELECT MAX(game_date) as last_prop FROM player_prop_odds")
             res = cursor.fetchone()
             print(f"  player_prop_odds: Last prop date {res['last_prop']}")
             
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")
    print("\n")
