"""
CLV Feedback Engine (R5)
=========================
Tracks Closing Line Value to measure model predictive power.

CLV = Difference between our line at bet time vs closing line.
  Positive CLV → we got a better number than the market closed at
  Negative CLV → market moved against us (we were wrong)

Key mechanics:
1. Record opening line, our predicted value, and closing line
2. Calculate CLV for each historical bet
3. Correlate CLV with features to identify which signals predict CLV
4. Weight future features by their CLV correlation

If a feature (e.g., "opposing team 2nd game of back-to-back") 
consistently gets positive CLV, weight it higher in future predictions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("CLV")


class CLVFeedbackEngine:
    """
    Tracks and analyzes Closing Line Value to create a feedback loop.
    """
    
    def __init__(self, db=None):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self._feature_clv_weights: Dict[str, float] = {}
    
    def record_bet_snapshot(self, player_id: int, player_name: str,
                            market: str, direction: str,
                            opening_line: float, model_prediction: float,
                            model_edge: float, model_confidence: str,
                            features: Dict[str, Any],
                            game_date: str):
        """
        Record a pre-game snapshot for CLV tracking.
        Called at prediction time.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Ensure table exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS clv_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date TEXT,
                        player_id INTEGER,
                        player_name TEXT,
                        market TEXT,
                        direction TEXT,
                        opening_line REAL,
                        model_prediction REAL,
                        model_edge REAL,
                        model_confidence TEXT,
                        closing_line REAL,
                        actual_result REAL,
                        clv REAL,
                        features_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                import json
                cursor.execute("""
                    INSERT INTO clv_snapshots 
                    (game_date, player_id, player_name, market, direction,
                     opening_line, model_prediction, model_edge, model_confidence,
                     features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_date, player_id, player_name, market, direction,
                      opening_line, model_prediction, model_edge, model_confidence,
                      json.dumps(features, default=str)))
                conn.commit()
        except Exception as e:
            logger.debug(f"CLV snapshot failed: {e}")
    
    def update_closing_lines(self, game_date: str, 
                              closing_data: Dict[int, Dict[str, float]]):
        """
        Update snapshots with closing line data after games start.
        
        closing_data: {player_id: {'closing_line': 22.5, 'actual_result': 25}}
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                for player_id, data in closing_data.items():
                    closing_line = data.get('closing_line')
                    actual = data.get('actual_result')
                    
                    if closing_line is not None:
                        # Calculate CLV
                        cursor.execute("""
                            SELECT id, opening_line, direction FROM clv_snapshots
                            WHERE game_date = ? AND player_id = ?
                            AND closing_line IS NULL
                        """, (game_date, player_id))
                        
                        for row in cursor.fetchall():
                            opening = row['opening_line']
                            direction = row['direction']
                            
                            # CLV: positive = we got a better number
                            if direction == 'OVER':
                                clv = closing_line - opening  # Closing went up → we got lower
                            else:
                                clv = opening - closing_line  # Closing went down → we got higher
                            
                            cursor.execute("""
                                UPDATE clv_snapshots 
                                SET closing_line = ?, actual_result = ?, clv = ?
                                WHERE id = ?
                            """, (closing_line, actual, clv, row['id']))
                
                conn.commit()
        except Exception as e:
            logger.warning(f"CLV update failed: {e}")
    
    def calculate_clv_feature_weights(self, lookback_days: int = 60) -> Dict[str, float]:
        """
        Analyze which features correlate with positive CLV.
        
        Returns:
            Dict[feature_name → weight] where weight > 0 = predictive of positive CLV
        """
        try:
            import json
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT features_json, clv FROM clv_snapshots
                    WHERE clv IS NOT NULL
                    AND game_date >= date('now', ?)
                """, (f'-{lookback_days} days',))
                rows = cursor.fetchall()
            
            if len(rows) < 20:
                return {}  # Need minimum data
            
            # Parse features and CLV
            feature_clv_pairs: Dict[str, List[float]] = {}
            
            for row in rows:
                features = json.loads(row['features_json']) if row['features_json'] else {}
                clv = row['clv'] or 0.0
                
                for fname, fval in features.items():
                    if not isinstance(fval, (int, float)):
                        continue
                    if fname not in feature_clv_pairs:
                        feature_clv_pairs[fname] = []
                    feature_clv_pairs[fname].append((fval, clv))
            
            # Calculate correlation of each feature with CLV
            weights = {}
            for fname, pairs in feature_clv_pairs.items():
                if len(pairs) < 10:
                    continue
                fvals = np.array([p[0] for p in pairs])
                clvs = np.array([p[1] for p in pairs])
                
                if np.std(fvals) > 0 and np.std(clvs) > 0:
                    corr = np.corrcoef(fvals, clvs)[0, 1]
                    if not np.isnan(corr):
                        weights[fname] = float(corr)
            
            self._feature_clv_weights = weights
            return weights
            
        except Exception as e:
            logger.warning(f"CLV feature analysis failed: {e}")
            return {}
    
    def get_clv_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get CLV performance summary.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT clv, direction, model_edge, model_confidence,
                           actual_result, opening_line, closing_line
                    FROM clv_snapshots
                    WHERE clv IS NOT NULL
                    AND game_date >= date('now', ?)
                """, (f'-{lookback_days} days',))
                rows = cursor.fetchall()
            
            if not rows:
                return {'total_bets': 0, 'avg_clv': 0, 'positive_clv_pct': 0}
            
            clvs = [r['clv'] for r in rows]
            
            return {
                'total_bets': len(rows),
                'avg_clv': np.mean(clvs),
                'median_clv': np.median(clvs),
                'positive_clv_pct': sum(1 for c in clvs if c > 0) / len(clvs) * 100,
                'std_clv': np.std(clvs),
                'by_confidence': self._clv_by_group(rows, 'model_confidence'),
            }
        except Exception as e:
            logger.warning(f"CLV summary failed: {e}")
            return {'total_bets': 0, 'avg_clv': 0, 'positive_clv_pct': 0}
    
    def _clv_by_group(self, rows, group_field: str) -> Dict[str, Dict]:
        """Group CLV stats by a field."""
        groups: Dict[str, List[float]] = {}
        for r in rows:
            key = str(r[group_field])
            if key not in groups:
                groups[key] = []
            groups[key].append(r['clv'])
        
        return {
            k: {'avg_clv': np.mean(v), 'count': len(v)}
            for k, v in groups.items()
        }
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get cached CLV-feature weights."""
        if not self._feature_clv_weights:
            self.calculate_clv_feature_weights()
        return self._feature_clv_weights


# Convenience function
def get_clv_engine(db=None):
    """Get CLVFeedbackEngine instance."""
    return CLVFeedbackEngine(db=db)
