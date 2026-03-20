"""
Production Guardian
===================
Enforces strict quality gates for data integrity, model variability, and prediction structure.
Acts as the "Safety Officer" for the pipeline.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..utils.database import DatabaseManager
from ..utils.config import get_config

@dataclass
class GuardianReport:
    """Report strictly for pipeline health status"""
    status: str  # 'GREEN', 'YELLOW', 'RED'
    flags: List[str]
    metrics: Dict[str, Any]
    
    def __str__(self):
        return f"[{self.status}] {len(self.flags)} flags raised"

class ProductionGuardian:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.config = get_config()
        self.flags = []
        
    def check_slate_integrity(self, events: List[Dict]) -> bool:
        """
        Gate 1: Is the slate complete?
        Checks if we have games covering the full window.
        """
        if not events:
            self.flags.append("CRITICAL: No events found for slate window.")
            return False
            
        # Check for duplicate event IDs
        ids = [e.get('event_id') for e in events]
        if len(ids) != len(set(ids)):
            self.flags.append(f"WARN: {len(ids) - len(set(ids))} duplicate event IDs detected.")
            
        return True

    def check_data_freshness(self) -> bool:
        """
        Gate 2: Is the data fresh?
        Checks props, injuries, and team stats.
        """
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check Props
            cursor.execute("SELECT COUNT(*) as cnt FROM player_prop_odds WHERE date(snapshot_time) = ?", (today,))
            props_count = cursor.fetchone()['cnt']
            if props_count < 10:
                self.flags.append(f"CRITICAL: Only {props_count} player props found for today. data ingestion likely failed.")
                return False
                
            # Check Injuries
            cursor.execute("SELECT COUNT(*) as cnt FROM injury_snapshots WHERE report_date = ?", (today,))
            inj_count = cursor.fetchone()['cnt']
            if inj_count < 50: # Expecting 30 teams * ~12 players... usually 400+, but snapshots only store *changes* or active list?
                # Actually injury snapshots might be full list. 50 is conservative lower bound for "active roster info present"
                self.flags.append(f"WARN: Low injury snapshot count ({inj_count}).")
        
        return True

    def check_feature_variability(self, feature_df: pd.DataFrame) -> bool:
        """
        Gate 3: Are features actually varying?
        Detects frozen features (std dev = 0).
        """
        if feature_df.empty:
            return True # Nothing to check
            
        # Key features to check
        critical_cols = ['minutes_L10', 'minutes_L5', 'ppm_L10', 'usg_L10']
        
        for col in critical_cols:
            if col in feature_df.columns:
                if feature_df[col].std() < 0.01:
                    self.flags.append(f"CRITICAL: Feature '{col}' has ZERO VARIANCE. Data pipeline probable failure.")
                    return False
        return True

    def check_prediction_reasons(self, predictions_df: pd.DataFrame) -> bool:
        """
        Gate 4: Are predictions structurally sound?
        pct_in_range logic is post-game, but we can check structural sanity here.
        """
        if predictions_df.empty:
            return True
            
        # Check median vs mean (huge skew alert)
        # Check p10 < p90
        # Check if all predictions are identical
        
        if predictions_df['predicted_mean'].std() < 1.0:
             self.flags.append("CRITICAL: Predicted points have near-zero variance across ALL players. Model collapse.")
             return False
             
        violations = predictions_df[predictions_df['p10'] >= predictions_df['p90']]
        if not violations.empty:
            self.flags.append(f"CRITICAL: {len(violations)} predictions have p10 >= p90.")
            return False

        if 'used_fallback_model' in predictions_df.columns:
            fallback_rate = float(predictions_df['used_fallback_model'].fillna(0).mean())
            if fallback_rate >= 0.50:
                self.flags.append(
                    f"CRITICAL: {fallback_rate:.0%} of predictions used fallback models. Model layer is degraded."
                )
                return False
            if fallback_rate >= 0.20:
                self.flags.append(
                    f"WARN: {fallback_rate:.0%} of predictions used fallback models."
                )

        if 'prediction_health_score' in predictions_df.columns:
            avg_health = float(predictions_df['prediction_health_score'].fillna(1.0).mean())
            if avg_health < 0.55:
                self.flags.append(
                    f"CRITICAL: Prediction health score averaged {avg_health:.2f}. Inputs are too degraded."
                )
                return False
            if avg_health < 0.75:
                self.flags.append(
                    f"WARN: Prediction health score averaged {avg_health:.2f}."
                )

        if 'market_anchor_applied' in predictions_df.columns:
            anchor_rate = float(predictions_df['market_anchor_applied'].fillna(0).mean())
            if anchor_rate > 0.90:
                self.flags.append(
                    f"WARN: {anchor_rate:.0%} of predictions were market-anchored. Model independence is low."
                )
            
        return True

    def _get_coverage(self, days_back: int = 7, min_audited: int = 20) -> float:
        """Get weighted average P10-P90 coverage over recent substantial audit days."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pct_in_p10_p90_range, predictions_audited
                FROM model_performance 
                WHERE predictions_audited >= ?
                ORDER BY audit_date DESC 
                LIMIT ?
            """, (min_audited, days_back))
            rows = cursor.fetchall()
        
        if not rows:
            return -1.0  # No data sentinel

        weighted_total = 0.0
        total_weight = 0.0
        for row in rows:
            weight = max(float(row['predictions_audited'] or 0), 1.0)
            weighted_total += float(row['pct_in_p10_p90_range'] or 0.0) * weight
            total_weight += weight

        return weighted_total / total_weight if total_weight else -1.0

    def check_historical_calibration(self, days_back: int = 7) -> str:
        """
        Gate 5 (R9): Graduated Active Calibration Check.
        
        Analyzes recent model performance to detect over/under confidence
        with graduated defense levels and recent-data override.
        
        Defense Levels (ascending severity):
          STABLE           → No adjustments (coverage 70-85%)
          SLIGHTLY_OVERCONFIDENT → +2% edge, 1.10x std (coverage 65-70%)
          RECOVERING       → +2% edge, 1.15x std (L3 clean but L7 dirty)
          OVERCONFIDENT    → +4% edge, 1.30x std (coverage 55-65%)
          SEVERELY_OVERCONFIDENT → +6% edge, 1.40x std (coverage <55%)
          UNDERCONFIDENT   → -2% edge, 0.90x std (coverage >85%)
        
        Returns:
            Calibration status string
        """
        full_coverage = self._get_coverage(days_back=days_back, min_audited=20)
        
        if full_coverage < 0:
            return 'STABLE'  # No history yet
        
        # Recent-data override: if L3 is clean, allow graduation
        # even if L7 is still dragged down by stale bad data
        recent_coverage = self._get_coverage(days_back=3, min_audited=20)

        if recent_coverage >= 72.0 and full_coverage < 70.0:
            self.flags.append(
                f"CALIBRATION RECOVERING: Recent L3 coverage {recent_coverage:.1f}% "
                f"improved vs L{days_back} {full_coverage:.1f}%. Graduating to mild defense."
            )
            return 'RECOVERING'

        # If recent substantial days are improving materially, step down one level
        if recent_coverage >= 65.0 and full_coverage < 60.0:
            self.flags.append(
                f"CALIBRATION RECOVERING: Recent substantial coverage {recent_coverage:.1f}% "
                f"is improving vs L{days_back} {full_coverage:.1f}%. Holding at moderate defense."
            )
            return 'OVERCONFIDENT'
        
        # Graduated severity levels
        if full_coverage < 52.0:
            self.flags.append(
                f"CALIBRATION ALERT: SEVERELY OVERCONFIDENT "
                f"(Coverage {full_coverage:.1f}% < 52%). Maximum defensive thresholds."
            )
            return 'SEVERELY_OVERCONFIDENT'
        
        if full_coverage < 60.0:
            self.flags.append(
                f"CALIBRATION ALERT: OVERCONFIDENT "
                f"(Coverage {full_coverage:.1f}% < 60%). Widening thresholds."
            )
            return 'OVERCONFIDENT'
        
        if full_coverage < 68.0:
            self.flags.append(
                f"CALIBRATION NOTE: SLIGHTLY OVERCONFIDENT "
                f"(Coverage {full_coverage:.1f}% < 68%). Mild threshold adjustment."
            )
            return 'SLIGHTLY_OVERCONFIDENT'
        
        if full_coverage > 85.0:
            self.flags.append(
                f"CALIBRATION ALERT: UNDERCONFIDENT "
                f"(Coverage {full_coverage:.1f}% > 85%). Narrowing thresholds."
            )
            return 'UNDERCONFIDENT'
        
        return 'STABLE'
        
    def check_distribution_drift(self, current_df: pd.DataFrame, days_back: int = 7) -> bool:
        """
        Gate 6: Distribution Drift Check.
        Compares today's prediction distribution vs historical baseline.
        """
        if current_df.empty:
            return True

        today_mean = current_df['predicted_mean'].mean()
        today_std = current_df['predicted_mean'].std()
        
        hist_stats = self._get_historical_stats(days_back)
        if not hist_stats:
            return True # No history
            
        hist_mean = hist_stats['mean']
        hist_std = hist_stats['std']
        
        # Check Mean Shift (Z-test)
        # We use standard error of the mean? Or just raw distribution shift?
        # Raw shift is better for "Environmental Drift" (e.g. NBA scoring changes)
        z_score = (today_mean - hist_mean) / hist_std if hist_std > 0 else 0
        
        if abs(z_score) > 2.5:
            direction = "HIGHER" if z_score > 0 else "LOWER"
            self.flags.append(
                f"DRIFT ALERT: Predictions are significantly {direction} than usual "
                f"(Z={z_score:.1f}). Today: {today_mean:.1f}, Hist: {hist_mean:.1f}"
            )
            return False
            
        return True

    def _get_historical_stats(self, days_back: int) -> Optional[Dict[str, float]]:
        """Get distribution stats from past N days."""
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(pred_points) as mean_pts, AVG(pred_std) as std_pts
                FROM prediction_log
                WHERE game_date >= ?
            """, (cutoff,))
            row = cursor.fetchone()
            
        if not row or row['mean_pts'] is None:
            return None
            
        return {'mean': row['mean_pts'], 'std': row['std_pts']}

    def save_alerts(self, filepath: str = "ALERTS.log"):
        """Persist Critical/Warn flags to a simple log file."""
        if not self.flags:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filepath, 'a') as f:
            for flag in self.flags:
                f.write(f"[{timestamp}] {flag}\n")

    def generate_health_report(self) -> GuardianReport:
        """
        Compile final status.
        """
        status = 'GREEN'
        criticals = [f for f in self.flags if 'CRITICAL' in f or 'DRIFT' in f]
        warns = [f for f in self.flags if 'WARN' in f or 'CALIBRATION' in f]
        
        if criticals:
            status = 'RED'
        elif warns:
            status = 'YELLOW'
            
        return GuardianReport(
            status=status,
            flags=self.flags,
            metrics={'flag_count': len(self.flags)}
        )
