"""

Model Training



Two-stage model training:

1. Minutes Model: Predicts playing time distribution

2. PPM Model: Predicts points per minute efficiency



Uses LightGBM with time-series cross-validation to prevent leakage.

"""



import pickle

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



import numpy as np

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



try:

    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False

    print("[WARN] LightGBM not installed, using dummy model")



from ..utils.config import get_config

from ..utils.database import DatabaseManager

from .feature_engineering import FeatureEngineer





class MinutesModel:

    """Predicts player minutes based on pre-game features."""

    

    FEATURE_COLUMNS = [

        'minutes_L3', 'minutes_L5', 'minutes_L10', 'minutes_season',

        'min_std_L5',

        'is_starter', 'rest_days', 'is_home', 'games_played',

        'spread', 'blowout_risk',

    ]

    

    def __init__(self):

        self.model = None

        self.feature_importance = {}

        self.metrics = {}

    

    def train(self, X: pd.DataFrame, y: pd.Series, 

              n_splits: int = 5) -> Dict[str, float]:

        """Train the minutes model with time-series CV.

        

        Args:

            X: Feature DataFrame

            y: Target (actual minutes)

            n_splits: Number of CV splits

            

        Returns:

            Dict of evaluation metrics

        """

        if not LIGHTGBM_AVAILABLE:

            print("  [WARN] LightGBM not available, using simple mean model")

            self.model = {'type': 'mean', 'value': y.mean()}

            return {'mae': y.std(), 'rmse': y.std()}

        

        # Prepare features

        X_clean = X[self.FEATURE_COLUMNS].copy()

        X_clean = X_clean.fillna(0)

        

        # Time series cross-validation

        tscv = TimeSeriesSplit(n_splits=n_splits)

        

        mae_scores = []

        rmse_scores = []

        

        for train_idx, val_idx in tscv.split(X_clean):

            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            

            # Train LightGBM

            train_data = lgb.Dataset(X_train, label=y_train)

            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            

            params = {

                'objective': 'regression',

                'metric': 'mae',

                'boosting_type': 'gbdt',

                'num_leaves': 31,

                'learning_rate': 0.05,

                'feature_fraction': 0.8,

                'bagging_fraction': 0.8,

                'bagging_freq': 5,

                'verbose': -1,

                'seed': 42

            }

            

            model = lgb.train(

                params,

                train_data,

                num_boost_round=500,

                valid_sets=[val_data],

                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]

            )

            

            # Evaluate

            preds = model.predict(X_val)

            mae_scores.append(mean_absolute_error(y_val, preds))

            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        

        # Train final model on all data

        train_data = lgb.Dataset(X_clean, label=y)

        self.model = lgb.train(params, train_data, num_boost_round=500)

        

        # Store feature importance

        importance = self.model.feature_importance(importance_type='gain')

        self.feature_importance = dict(zip(self.FEATURE_COLUMNS, importance))

        

        self.metrics = {

            'mae': np.mean(mae_scores),

            'mae_std': np.std(mae_scores),

            'rmse': np.mean(rmse_scores),

            'rmse_std': np.std(rmse_scores)

        }

        

        return self.metrics

    

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        """Predict minutes.

        

        Args:

            X: Feature DataFrame

            

        Returns:

            Array of predicted minutes

        """

        if self.model is None:

            raise ValueError("Model not trained yet")

        

        if isinstance(self.model, dict) and self.model['type'] == 'mean':

            return np.full(len(X), self.model['value'])

        

        X_clean = X[self.FEATURE_COLUMNS].copy()

        X_clean = X_clean.fillna(0)

        return self.model.predict(X_clean)

    

    def predict_distribution(self, X: pd.DataFrame, 

                              n_samples: int = 1000) -> np.ndarray:

        """Predict minutes with uncertainty (for simulation).

        

        Args:

            X: Feature DataFrame (single row)

            n_samples: Number of samples to generate

            

        Returns:

            Array of sampled minute values

        """

        mean_pred = self.predict(X)[0]

        

        # Use historical std or estimated uncertainty

        std = X['min_std_L5'].values[0] if 'min_std_L5' in X.columns else 5.0

        std = max(std, 2.0)  # Minimum uncertainty

        

        # Sample from truncated normal (minutes >= 0)

        samples = np.random.normal(mean_pred, std, n_samples)

        samples = np.maximum(samples, 0)

        samples = np.minimum(samples, 48)  # Cap at 48 minutes

        

        return samples

    

    def save(self, path: Path) -> None:

        """Save model to disk."""

        with open(path, 'wb') as f:

            pickle.dump({

                'model': self.model,

                'feature_importance': self.feature_importance,

                'metrics': self.metrics,

                'feature_columns': self.FEATURE_COLUMNS

            }, f)

    

    def load(self, path: Path) -> None:

        """Load model from disk."""

        with open(path, 'rb') as f:

            data = pickle.load(f)

            self.model = data['model']

            self.feature_importance = data['feature_importance']

            self.metrics = data['metrics']





class PPMModel:

    """Predicts points per minute based on pre-game features."""

    

    FEATURE_COLUMNS = [

        'ppm_L3', 'ppm_L5', 'ppm_L10',

        'ppm_std_L5',

        'is_starter', 'rest_days', 'is_home', 'games_played',

        'total', 'pace_proxy', 'opp_def_rating',

        # Style Edge Features (NEW)

        'pace_edge', 'efficiency_edge',

        'three_pt_style_match', 'paint_style_match',

        'opp_pace_L10', 'opp_def_rating_L10',

    ]

    

    def __init__(self):

        self.model = None

        self.feature_importance = {}

        self.metrics = {}

    

    def train(self, X: pd.DataFrame, y: pd.Series,

              n_splits: int = 5) -> Dict[str, float]:

        """Train the PPM model with time-series CV."""

        if not LIGHTGBM_AVAILABLE:

            print("  [WARN] LightGBM not available, using simple mean model")

            self.model = {'type': 'mean', 'value': y.mean()}

            return {'mae': y.std(), 'rmse': y.std()}

        

        X_clean = X[self.FEATURE_COLUMNS].copy()

        X_clean = X_clean.fillna(0)

        

        tscv = TimeSeriesSplit(n_splits=n_splits)

        

        mae_scores = []

        rmse_scores = []

        

        for train_idx, val_idx in tscv.split(X_clean):

            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            

            train_data = lgb.Dataset(X_train, label=y_train)

            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            

            params = {

                'objective': 'regression',

                'metric': 'mae',

                'boosting_type': 'gbdt',

                'num_leaves': 31,

                'learning_rate': 0.05,

                'feature_fraction': 0.8,

                'bagging_fraction': 0.8,

                'bagging_freq': 5,

                'verbose': -1,

                'seed': 42

            }

            

            model = lgb.train(

                params,

                train_data,

                num_boost_round=500,

                valid_sets=[val_data],

                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]

            )

            

            preds = model.predict(X_val)

            mae_scores.append(mean_absolute_error(y_val, preds))

            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        

        # Final model

        train_data = lgb.Dataset(X_clean, label=y)

        self.model = lgb.train(params, train_data, num_boost_round=500)

        

        importance = self.model.feature_importance(importance_type='gain')

        self.feature_importance = dict(zip(self.FEATURE_COLUMNS, importance))

        

        self.metrics = {

            'mae': np.mean(mae_scores),

            'mae_std': np.std(mae_scores),

            'rmse': np.mean(rmse_scores),

            'rmse_std': np.std(rmse_scores)

        }

        

        return self.metrics

    

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        """Predict PPM."""

        if self.model is None:

            raise ValueError("Model not trained yet")

        

        if isinstance(self.model, dict) and self.model['type'] == 'mean':

            return np.full(len(X), self.model['value'])

        

        X_clean = X[self.FEATURE_COLUMNS].copy()

        X_clean = X_clean.fillna(0)

        return self.model.predict(X_clean)

    

    def predict_distribution(self, X: pd.DataFrame,

                              n_samples: int = 1000) -> np.ndarray:

        """Predict PPM with uncertainty."""

        mean_pred = self.predict(X)[0]

        

        std = X['ppm_std_L5'].values[0] if 'ppm_std_L5' in X.columns else 0.15

        std = max(std, 0.05)  # Minimum uncertainty

        

        samples = np.random.normal(mean_pred, std, n_samples)

        samples = np.maximum(samples, 0)

        

        return samples

    

    def save(self, path: Path) -> None:

        """Save model to disk."""

        with open(path, 'wb') as f:

            pickle.dump({

                'model': self.model,

                'feature_importance': self.feature_importance,

                'metrics': self.metrics,

                'feature_columns': self.FEATURE_COLUMNS

            }, f)

    

    def load(self, path: Path) -> None:

        """Load model from disk."""

        with open(path, 'rb') as f:

            data = pickle.load(f)

            self.model = data['model']

            self.feature_importance = data['feature_importance']

            self.metrics = data['metrics']





class MarketModel:
    """Predicts raw stats (Points, Assists, Rebounds) directly using XGBoost/LightGBM."""
    
    def __init__(self, market_type: str):
        self.market_type = market_type
        self.model = None
        self.feature_importance = {}
        self.metrics = {}
        
        # Define features based on MarketPredictor config
        # This duplicates simple_predictor.py slightly but keeps training decoupled
        if market_type == 'points':
            self.FEATURE_COLUMNS = ['ppg_L5', 'minutes_L5', 'minutes_season', 'ppm_L5', 'is_home', 'is_b2b', 'opp_def_rating']
        elif market_type == 'assists':
            # Use only features that exist in training dataset
            self.FEATURE_COLUMNS = [
                'ast_L5',              # Recent assists avg
                'minutes_L5',          # Playing time
                'minutes_season',      # [NEW] Long-term role context
                'is_home', 'is_b2b',   # Game context
                'opp_def_rating',      # Opponent defense
                'rest_days',           # Rest/fatigue
            ]
        elif market_type == 'rebounds':
            # Use only features that exist in training dataset
            self.FEATURE_COLUMNS = [
                'reb_L5',              # Recent rebounds avg
                'minutes_L5',          # Playing time
                'minutes_season',      # [NEW] Long-term role context
                'is_home', 'is_b2b',   # Game context
                'opp_def_rating',      # Opponent defense
                'rest_days',           # Rest/fatigue
            ]
        else:
            raise ValueError(f"Unknown market type: {market_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """Train model with time-series CV."""
        if not LIGHTGBM_AVAILABLE:
            print(f"  [WARN] LightGBM not available, using mean for {self.market_type}")
            self.model = {'type': 'mean', 'value': y.mean()}
            return {'mae': y.std()}

        X_clean = X[self.FEATURE_COLUMNS].copy().fillna(0)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []

        # CV Loop
        for train_idx, val_idx in tscv.split(X_clean):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))

        # Final Train
        train_data = lgb.Dataset(X_clean, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=500)
        
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.FEATURE_COLUMNS, importance))
        
        self.metrics = {'mae': np.mean(mae_scores), 'mae_std': np.std(mae_scores)}
        return self.metrics

    def save(self, path: Path, feature_path: Path) -> None:
        """Save model and features separately for MarketPredictor comapatibility."""
        # 1. Save Model Object (Pickle logic used by MarketPredictor)
        # Note: MarketPredictor uses joblib.load. Pickle is compatible.
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # 2. Save Feature JSON
        import json
        with open(feature_path, 'w') as f:
            json.dump(self.FEATURE_COLUMNS, f)


class ModelTrainer:

    """Orchestrates training of both models."""

    

    def __init__(self, db: Optional[DatabaseManager] = None):

        self.db = db or DatabaseManager()

        self.fe = FeatureEngineer(self.db)

        self.config = get_config()

        

        self.minutes_model = MinutesModel()
        self.ppm_model = PPMModel()
        
        # Market Models
        self.market_models = {
            'points': MarketModel('points'),
            'assists': MarketModel('assists'),
            'rebounds': MarketModel('rebounds')
        }

        

        # Model save paths

        self.model_dir = self.config.project_root / 'models'

        self.model_dir.mkdir(exist_ok=True)

    

    def train_models(self, min_samples: int = 100, seasons: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Train both models.
        
        Args:
            min_samples: Minimum training samples required
            seasons: List of seasons to train on (None = current only, ['ALL'] = all history)
            
        Returns:
            Dict with metrics for both models
        """
        print("\n[Training] Building training dataset...")
        if seasons == ['ALL'] or seasons == 'ALL':
             print("   (Targeting FULL HISTORY from DB)")
        
        # Build training data
        df = self.fe.build_training_dataset(min_games=10, seasons=seasons)

        

        if len(df) < min_samples:

            print(f"   Only {len(df)} samples, need {min_samples}")

            print("  Using available data anyway...")

        

        # Filter rows with valid targets

        df = df.dropna(subset=['target_minutes', 'target_ppm'])

        print(f"  Training samples: {len(df)}")

        

        # =====================

        # Train Minutes Model

        # =====================

        print("\n[Training] Training Minutes Model...")

        

        X = df.copy()

        y_minutes = df['target_minutes']

        

        min_metrics = self.minutes_model.train(X, y_minutes)

        print(f"   Minutes Model - MAE: {min_metrics['mae']:.2f} mins")

        

        # Check feature importance

        if self.minutes_model.feature_importance:

            print("\n  Top Features (Minutes):")

            sorted_imp = sorted(

                self.minutes_model.feature_importance.items(),

                key=lambda x: x[1], reverse=True

            )[:5]

            for feat, imp in sorted_imp:

                print(f"    - {feat}: {imp:.0f}")

        

        # =====================

        # Train PPM Model

        # =====================

        print("\n[Training] Training PPM Model...")

        

        y_ppm = df['target_ppm']

        

        ppm_metrics = self.ppm_model.train(X, y_ppm)

        print(f"   PPM Model - MAE: {ppm_metrics['mae']:.3f}")

        

        if self.ppm_model.feature_importance:

            print("\n  Top Features (PPM):")

            sorted_imp = sorted(

                self.ppm_model.feature_importance.items(),

                key=lambda x: x[1], reverse=True

            )[:5]

            for feat, imp in sorted_imp:

                print(f"    - {feat}: {imp:.0f}")

        

        # Save models

        self.save_models()

        

        return {

            'minutes': min_metrics,

            'ppm': ppm_metrics

        }

    

    def save_models(self) -> None:

        """Save trained models to disk."""

        self.minutes_model.save(self.model_dir / 'minutes_model.pkl')

        self.ppm_model.save(self.model_dir / 'ppm_model.pkl')

        print(f"\n   Models saved to {self.model_dir}")
        
    def train_market_models(self, min_samples: int = 100, seasons: List[str] = None):
        """Train Direct Market Models (Points, Assists, Rebounds)."""
        print("\n[Training] Building Market Model Dataset...")
        df = self.fe.build_training_dataset(min_games=10, seasons=seasons)
        
        results = {}
        
        for market, model in self.market_models.items():
            print(f"\n[Training] Training {market.upper()} Model...")
            target_col = f"target_{market}"
            
            # Filter valid targets
            train_df = df.dropna(subset=[target_col])
            print(f"   Samples: {len(train_df)}")
            
            if len(train_df) < min_samples:
                print("   Not enough samples, skipping.")
                continue
                
            metrics = model.train(train_df, train_df[target_col])
            results[market] = metrics
            print(f"   MAE: {metrics['mae']:.3f}")
            
            # Save
            model_path = self.model_dir / f"model_{market}.pkl"
            feat_path = self.model_dir / f"model_{market}_features.json"
            model.save(model_path, feat_path)
            print(f"   Saved to {model_path}")
            
        return results

    

    def load_models(self) -> bool:

        """Load models from disk.

        

        Returns:

            True if successful, False otherwise

        """

        try:

            self.minutes_model.load(self.model_dir / 'minutes_model.pkl')

            self.ppm_model.load(self.model_dir / 'ppm_model.pkl')

            return True

        except FileNotFoundError:

            return False





# Convenience functions

def get_model_trainer() -> ModelTrainer:

    """Get model trainer instance."""

    return ModelTrainer()

