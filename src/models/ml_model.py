
import logging
import joblib
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("ML_MODEL")

class ResidualAdjuster:
    """
    Wraps the XGBoost/GradientBoosting residual model.
    """
    
    def __init__(self, model_path: str = 'src/models/residual_model.pkl', 
                 features_path: str = 'src/models/model_features.json'):
        self.model = None
        self.feature_names = []
        self.model_path = None
        self.features_path = None
        resolved_model_path, resolved_features_path = self._resolve_paths(model_path, features_path)
        self._load_model(resolved_model_path, resolved_features_path)

    def _resolve_paths(self, model_path: str, features_path: str) -> Tuple[str, str]:
        """Prefer top-level trained assets, then fall back to legacy paths."""
        model_candidates = [Path("models/residual_model.pkl"), Path(model_path)]
        feature_candidates = [Path("models/model_features.json"), Path(features_path)]
        resolved_model = next((str(path) for path in model_candidates if path.exists()), model_path)
        resolved_features = next((str(path) for path in feature_candidates if path.exists()), features_path)
        return resolved_model, resolved_features
        
    def _load_model(self, model_path, features_path):
        try:
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                self.model_path = model_path
                logger.info(f"Loaded residual model from {model_path}")
            else:
                logger.info("Residual model not found. Residual adjustments disabled.")
                
            if Path(features_path).exists():
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.features_path = features_path
            else:
                logger.info("Residual feature names file not found. Residual adjustments disabled.")
                
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.feature_names)

    def predict_residual(self, features_dict: Dict[str, float]) -> float:
        """
        Predict the residual (adjustment) for a single player.
        """
        if not self.model or not self.feature_names:
            return 0.0
            
        try:
            # Create DataFrame with exact columns
            df = pd.DataFrame([features_dict])
            
            # Ensure columns match training
            # 1. Add missing cols with 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            
            # 2. Select ordered cols
            df = df[self.feature_names]
            
            # Predict
            residual = float(self.model.predict(df)[0])
            
            # Safety Clamp: Don't allow massive ML swings yet
            residual = max(-8.0, min(8.0, residual))
            
            return residual
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return 0.0
