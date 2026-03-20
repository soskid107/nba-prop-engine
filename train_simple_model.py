"""
Simple Model Training Script
============================
Trains a lightweight XGBoost model on historical NBA player game logs.
Uses only 6 features for maximum simplicity and generalization.

Target: Actual Points
Features: ppg_L5, minutes_L5, is_home, is_b2b, opp_def_rating, ppm_L5
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMPLE_TRAINER")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(db_path: str = 'data/nba_props.db') -> pd.DataFrame:
    """Load and prepare training data from player_logs."""
    logger.info("Loading data from database...")
    
    conn = sqlite3.connect(db_path)
    
    # [FIX] Only train on players who typically get prop lines (12+ ppg average)
    # This matches the population we're predicting for
    query = """
    WITH player_avg AS (
        SELECT player_id, AVG(points) as avg_pts
        FROM player_logs
        WHERE minutes > 10
        GROUP BY player_id
        HAVING AVG(points) >= 12
    )
    SELECT 
        pl.player_id,
        pl.game_date,
        pl.points,
        pl.minutes,
        pl.is_home,
        pl.opponent_abbreviation as opponent,
        pl.team_abbreviation as team
    FROM player_logs pl
    INNER JOIN player_avg pa ON pl.player_id = pa.player_id
    WHERE pl.minutes > 10
    AND pl.game_date >= '2020-12-01'
    ORDER BY pl.player_id, pl.game_date
    """
    
    df = pd.read_sql_query(query, conn)
    logger.info(f"Loaded {len(df):,} game records (prop-line population only)")
    
    conn.close()
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages for each player (L5 window)."""
    logger.info("Computing rolling features...")
    
    # Sort for proper rolling computation
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
    
    # Group by player and compute rolling stats
    # IMPORTANT: shift(1) to avoid data leakage (don't include current game)
    df['ppg_L5'] = df.groupby('player_id')['points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    df['minutes_L5'] = df.groupby('player_id')['minutes'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    df['ppm_L5'] = df.groupby('player_id').apply(
        lambda g: (g['points'].shift(1).rolling(5, min_periods=3).sum() / 
                   g['minutes'].shift(1).rolling(5, min_periods=3).sum())
    ).reset_index(level=0, drop=True)
    
    # Back-to-back detection
    df['prev_game_date'] = df.groupby('player_id')['game_date'].shift(1)
    df['days_rest'] = pd.to_datetime(df['game_date']) - pd.to_datetime(df['prev_game_date'])
    df['is_b2b'] = (df['days_rest'] == timedelta(days=1)).astype(int)
    
    # Convert is_home to int
    df['is_home'] = df['is_home'].astype(int)
    
    # [NEW] Star player detection (elite scorers need different handling)
    df['is_star'] = (df['ppg_L5'] > 20).astype(int)
    
    # [NEW] Season PPG as floor anchor (helps with star regression)
    df['ppg_season'] = df.groupby('player_id')['points'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # [NEW] Head-to-Head: Player's historical avg vs this specific opponent
    # This captures matchup-specific tendencies
    df['ppg_vs_opp'] = df.groupby(['player_id', 'opponent'])['points'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    # Fill NaN (first game vs opponent) with season avg
    df['ppg_vs_opp'] = df['ppg_vs_opp'].fillna(df['ppg_season'])
    
    # [NEW] Days Rest (numeric, capped at 7)
    df['days_rest_num'] = df['days_rest'].dt.days.clip(0, 7).fillna(2)
    
    # [NEW] Minutes Trend (L5 vs L15 slope indicator)
    df['minutes_L15'] = df.groupby('player_id')['minutes'].transform(
        lambda x: x.shift(1).rolling(15, min_periods=5).mean()
    )
    df['minutes_trend'] = (df['minutes_L5'] - df['minutes_L15']).fillna(0)
    # Positive = role expanding, Negative = role shrinking
    
    # [NEW] Recent Shooting % (hot/cold streak)
    # We need FG attempts and makes - check if available
    
    logger.info(f"Features computed. Non-null rows: {df['ppg_L5'].notna().sum():,}")
    
    return df


def add_opponent_features(df: pd.DataFrame, db_path: str = 'data/nba_props.db') -> pd.DataFrame:
    """Add opponent defensive rating and pace."""
    logger.info("Adding opponent features...")
    
    conn = sqlite3.connect(db_path)
    
    # Try to get opponent def rating AND pace from team_advanced_stats
    try:
        opp_query = """
        SELECT team_abbreviation, def_rating, pace
        FROM team_advanced_stats
        WHERE window_type = 'SEASON'
        """
        opp_df = pd.read_sql_query(opp_query, conn)
        
        if len(opp_df) > 0:
            # Merge on opponent
            df = df.merge(
                opp_df.rename(columns={
                    'team_abbreviation': 'opponent', 
                    'def_rating': 'opp_def_rating',
                    'pace': 'opp_pace'
                }),
                on='opponent',
                how='left'
            )
        else:
            # Use league average as fallback
            df['opp_def_rating'] = 110.0
            df['opp_pace'] = 100.0
    except Exception as e:
        logger.warning(f"Could not load opponent stats: {e}")
        df['opp_def_rating'] = 110.0
        df['opp_pace'] = 100.0
    
    # Fill any remaining NaN with league average
    df['opp_def_rating'] = df['opp_def_rating'].fillna(110.0)
    df['opp_pace'] = df['opp_pace'].fillna(100.0)
    
    conn.close()
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare final feature matrix and target."""
    logger.info("Preparing final features...")
    
    # Drop rows with missing features
    feature_cols = [
        'ppg_L5', 'minutes_L5', 'ppm_L5', 'is_home', 'is_b2b', 
        'opp_def_rating', 'is_star', 'ppg_season', 'ppg_vs_opp',
        # New advanced features
        'opp_pace', 'days_rest_num', 'minutes_trend'
    ]
    
    df_clean = df.dropna(subset=feature_cols + ['points'])
    
    X = df_clean[feature_cols].values
    y = df_clean['points'].values
    dates = df_clean['game_date'].values
    
    logger.info(f"Final dataset: {len(X):,} samples, {len(feature_cols)} features")
    
    return X, y, dates, feature_cols


def train_model(X, y, dates):
    """Train XGBoost model with time-based split."""
    logger.info("Training model...")
    
    # Time-based split: last 20% of data for testing
    n = len(X)
    split_idx = int(n * 0.8)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    if HAS_XGB:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,  # Keep it simple!
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    # P50 Accuracy (how often prediction is on correct side of actual)
    # Using the predicted value as P50
    correct_direction = np.sum(
        ((test_preds > y_test - 0.5) & (test_preds < y_test + 0.5)) |
        ((test_preds <= y_test) & (y_test <= test_preds + test_mae)) |
        ((test_preds >= y_test) & (y_test >= test_preds - test_mae))
    )
    # Simpler: within 5 points
    within_5 = np.mean(np.abs(test_preds - y_test) <= 5) * 100
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TRAINING RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Train MAE: {train_mae:.2f} pts")
    logger.info(f"Test MAE:  {test_mae:.2f} pts")
    logger.info(f"Test RMSE: {test_rmse:.2f} pts")
    logger.info(f"Within 5 pts: {within_5:.1f}%")
    logger.info(f"{'='*50}")
    
    return model, test_mae


def save_model(model, feature_cols, output_dir='src/models'):
    """Save trained model and feature names."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = f'{output_dir}/simple_model.pkl'
    features_path = f'{output_dir}/simple_model_features.json'
    
    joblib.dump(model, model_path)
    
    import json
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Features saved to {features_path}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("SIMPLE MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # 1. Load data
    df = load_data()
    
    # 2. Compute rolling features
    df = compute_rolling_features(df)
    
    # 3. Add opponent features
    df = add_opponent_features(df)
    
    # 4. Prepare final features
    X, y, dates, feature_cols = prepare_features(df)
    
    # 5. Train model
    model, test_mae = train_model(X, y, dates)
    
    # 6. Save model
    save_model(model, feature_cols)
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Final Test MAE: {test_mae:.2f} points")
    
    return model, test_mae


if __name__ == "__main__":
    main()
