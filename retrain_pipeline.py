
import logging
from src.models.training import ModelTrainer
from src.utils.database import DatabaseManager

logging.basicConfig(level=logging.INFO)

def run_retraining():
    print("=== STARTING MODEL RETRAINING ===")
    print("Target: Full 5-Year History (2021-2026)")
    
    db = DatabaseManager()
    trainer = ModelTrainer(db)
    
    # Train Minutes and PPM models
    # We pass seasons=['ALL'] to indicate we want to fetch everything
    metrics = trainer.train_models(min_samples=1000, seasons=['ALL'])
    
    print("\n=== TRAINING COMPLETE ===")
    print("Minutes Model Metrics:")
    print(metrics.get('minutes', {}))
    print("\nPPM Model Metrics:")
    print(metrics.get('ppm', {}))
    
    # Save Models
    import os
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    # Use internal save methods from classes
    from pathlib import Path
    model_dir = Path('models')
    
    trainer.minutes_model.save(model_dir / 'minutes_model.pkl')
    trainer.ppm_model.save(model_dir / 'ppm_model.pkl')
    
    print(f"Models saved to {model_dir.absolute()}")
    
if __name__ == "__main__":
    run_retraining()
