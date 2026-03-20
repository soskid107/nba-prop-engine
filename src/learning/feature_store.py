
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class FeatureStore:
    """
    Central repository for ML features.
    Ensures consistency between Training and Inference.
    """
    
    @staticmethod
    def extract_features(player_context: Dict, match_context: Dict, 
                         mechanistic_prediction: float) -> Dict[str, float]:
        """
        Extract flat feature vector for a single prediction.
        
        Args:
            player_context: Player data
            match_context: Match/Opponent data
            mechanistic_prediction: The base output from Agent 2
            
        Returns:
            Dict of numerical features
        """
        
        features = {}
        
        # 1. Base Mechanistic Signal
        features['mech_pred'] = mechanistic_prediction
        features['minutes_proj'] = player_context.get('minutes_L5', 0) # Fallback if direct proj not passed
        
        # 2. Opponent Context
        features['is_home'] = 1.0 if match_context.get('is_home') else 0.0
        features['rest_days'] = float(player_context.get('rest_days', 1))
        
        # Defense Strength (Lower is harder)
        # We need to parse this from match_context if available, or pass it in.
        # Assuming match_context has 'defense_rating' or similar, or we infer from opponent name later.
        # For now, let's use what we have.
        
        # 3. Recent Form
        features['l5_avg'] = player_context.get('points_L5', 0)
        features['l15_avg'] = player_context.get('points_L15', 0)
        
        # Trend
        features['trend_diff'] = features['l5_avg'] - features['l15_avg']
        
        # 4. Market Context (if available, powerful feature)
        # implied_total = match_context.get('market_context', {}).get('implied_total', 0)
        # features['team_total'] = implied_total
        
        return features

    @staticmethod
    def to_dataframe(features_list: List[Dict]) -> pd.DataFrame:
        """Convert list of feature dicts to DataFrame"""
        return pd.DataFrame(features_list)
