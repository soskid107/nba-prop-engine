
import pandas as pd
import numpy as np
import logging
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAINER")

def train_model():
    logger.info("Loading training data...")
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        logger.error("training_data.csv not found! Run generate_training_data.py first.")
        return

    logger.info(f"Loaded {len(df)} samples.")
    
    # Features & Target
    # We drop metadata that isn't a feature
    drop_cols = ['target_residual', 'actual_points', 'player_id', 'game_date', 'team', 'opponent'] 
    # Note: 'team' and 'opponent' might not be in the csv if FeatureStore didn't add them, 
    # but generate_training_data might have added them to 'row'? 
    # Let's inspect columns dynamically.
    
    feature_cols = [c for c in df.columns if c not in ['target_residual', 'actual_points', 'player_id', 'game_date', 'team', 'opponent']]
    
    X = df[feature_cols]
    y = df['target_residual']
    
    logger.info(f"Features: {feature_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if HAS_XGB:
        logger.info("Training XGBoost Regressor...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    else:
        logger.info("XGBoost not found. Training Scikit-Learn GradientBoostingRegressor...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
    model.fit(X_train, y_train)
    
    # Evaluation
    preds_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    
    logger.info(f"Residual Model MAE: {mae:.2f}")
    logger.info(f"Residual Model RMSE: {rmse:.2f}")
    
    # Compare to Baseline (Predicting 0 residual)
    baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
    logger.info(f"Baseline (No ML) Residual MAE: {baseline_mae:.2f}")
    
    if mae < baseline_mae:
        logger.info(f"SUCCESS: Model improved error by {baseline_mae - mae:.2f} points per game!")
    else:
        logger.warning("Model did not improve over baseline. Needs more features or data.")
        
    # Save Model
    # XGBoost has internal save, but joblib is generic
    joblib.dump(model, 'src/models/residual_model.pkl')
    logger.info("Model saved to src/models/residual_model.pkl")
    
    # Save feature names for inference safety
    with open('src/models/model_features.json', 'w') as f:
        json.dump(feature_cols, f)

if __name__ == "__main__":
    train_model()
