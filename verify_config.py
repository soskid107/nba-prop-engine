import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.config import get_config
    config = get_config()
    print(f"Project Root: {config.project_root}")
    print(f"Database Path: {config.database_path}")
    print(f"Database Exists: {config.database_path.exists()}")
    
    if config.database_path.exists():
        import sqlite3
        conn = sqlite3.connect(config.database_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"Tables in config DB: {[r[0] for r in c.fetchall()]}")
        conn.close()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
