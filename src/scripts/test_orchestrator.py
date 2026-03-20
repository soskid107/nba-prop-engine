
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import PredictionOrchestrator

def run_test():
    print("Initializing Orchestrator...")
    orchestrator = PredictionOrchestrator()
    
    print("Running prediction for today's slate (checking integration)...")
    results = orchestrator.predict_todays_slate()
    
    print("\n[Betting Table]")
    print(orchestrator.format_betting_table(results))

if __name__ == "__main__":
    run_test()
