
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.training import ModelTrainer

def run_training():
    print("Initializing ModelTrainer...")
    trainer = ModelTrainer()
    
    print("Starting Multi-Market Training...")
    # Train using last season + current season (approx) or just let it grab default
    # Using small min_samples for testing if needed
    results = trainer.train_market_models(min_samples=50, seasons=['2024-25', '2025-26'])
    
    print("\nTraining Complete!")
    print(results)

if __name__ == "__main__":
    run_training()
