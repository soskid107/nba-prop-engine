import sqlite3
import os

dbs = ["nba_props.db", "data/nba_props.db"]

for db_path in dbs:
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        print(f"PATH: {abs_path} | EXISTS: False")
        continue
        
    stats = os.stat(abs_path)
    print(f"PATH: {abs_path}")
    print(f"  Size: {stats.st_size} bytes")
    print(f"  Modified: {stats.st_mtime}")
    
    try:
        conn = sqlite3.connect(abs_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM api_usage WHERE api_name='the_odds_api'")
        count = c.fetchone()[0]
        print(f"  the_odds_api count: {count}")
        
        c.execute("SELECT MAX(call_date) FROM api_usage WHERE api_name='the_odds_api'")
        last_date = c.fetchone()[0]
        print(f"  Last call date: {last_date}")
        
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")
    print("-" * 30)
