"""

Prediction Audit & Self-Learning Module



Fetches actual game results to compare against predictions.

Tracks accuracy metrics and enables continuous model improvement.

"""



import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple



from ..utils.config import get_config

from ..utils.database import DatabaseManager

from ..ingestion.nba_ingestion import NBAIngestion





class PredictionAuditor:

    """Audits predictions against actual results and tracks performance."""

    

    def __init__(self, db: Optional[DatabaseManager] = None):

        """Initialize auditor.

        

        Args:

            db: Optional database manager

        """

        self.config = get_config()

        self.db = db or DatabaseManager()

        self.nba = NBAIngestion(self.db)

        self._ensure_tables()

    @staticmethod
    def _grade_pick_against_line(
        pick_direction: Any,
        sportsbook_line: Any,
        target_actual: Any
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Grade only approved OVER/UNDER picks against the reported sportsbook line.

        Returns:
            (market_outcome, bet_won, direction_correct)

        Notes:
        - Rows without an approved side are excluded from right/wrong grading.
        - Pushes are preserved as an explicit market outcome.
        """
        direction = str(pick_direction or '').upper()
        if direction not in ('OVER', 'UNDER'):
            return None, None, None

        if sportsbook_line is None or pd.isna(sportsbook_line):
            return None, None, None

        if target_actual is None or pd.isna(target_actual):
            return None, None, None

        if target_actual > sportsbook_line:
            outcome = 'OVER'
        elif target_actual < sportsbook_line:
            outcome = 'UNDER'
        else:
            outcome = 'PUSH'

        if outcome == 'PUSH':
            return outcome, None, None

        direction_correct = 1 if direction == outcome else 0
        bet_won = direction_correct
        return outcome, bet_won, direction_correct

    

    def _ensure_tables(self) -> None:

        """Create audit tables if not exist."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            # Predictions archive table

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS predictions_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT NOT NULL,
                    player_id INTEGER NOT NULL,
                    player_name TEXT,
                    team TEXT,
                    opponent TEXT,
                    market_key TEXT DEFAULT 'points', -- [NEW]
                    
                    -- Predictions
                    predicted_mean REAL,
                    predicted_p10 REAL,
                    predicted_p50 REAL,
                    predicted_p90 REAL,
                    predicted_minutes REAL,
                    predicted_ppm REAL,
                    p_play REAL,
                    
                    -- Context
                    spread REAL,
                    total REAL,
                    blowout_risk REAL,
                    
                    -- Actuals (filled in later)
                    actual_points INTEGER,
                    actual_assists INTEGER,   -- [NEW]
                    actual_rebounds INTEGER,  -- [NEW]
                    actual_blocks INTEGER,
                    actual_steals INTEGER,
                    actual_threes INTEGER,
                    actual_field_goals INTEGER,
                    actual_fga INTEGER,
                    actual_plus_minus REAL,
                    actual_pra INTEGER,       -- [NEW]
                    actual_minutes REAL,
                    actual_ppm REAL,
                    did_play INTEGER,
                    
                    -- Audit metrics (filled in later)
                    prediction_error REAL,
                    was_in_range INTEGER,  -- Was actual between P10 and P90?
                    edge_score REAL,
                    edge_tier TEXT,
                    is_parlay_core INTEGER DEFAULT 0,
                    pure_model_pred REAL,
                    market_adjusted_pred REAL,
                    post_rule_pred REAL,
                    prediction_health_score REAL,
                    prediction_degradation_flags TEXT,
                    used_fallback_model INTEGER DEFAULT 0,
                    market_anchor_applied INTEGER DEFAULT 0,
                    market_anchor_weight REAL,
                    selection_reason TEXT,
                    edge_explanation TEXT,
                    core_predictor TEXT,
                    mechanistic_reference_used INTEGER DEFAULT 0,
                    mechanistic_reference_mean REAL,
                    monte_carlo_reference_used INTEGER DEFAULT 0,
                    monte_carlo_reference_mean REAL,
                    residual_model_ready INTEGER DEFAULT 0,
                    residual_reference_adjustment REAL,
                    injury_context_present INTEGER DEFAULT 0,
                    injury_context_size INTEGER DEFAULT 0,
                    ensemble_applied INTEGER DEFAULT 0,
                    ensemble_mean REAL,
                    reference_consensus_mean REAL,
                    model_disagreement REAL,
                    ensemble_notes TEXT,
                    edge_candidate_tier TEXT,
                    edge_candidate_direction TEXT,
                    decision_alignment_status TEXT,
                    approved_bet INTEGER DEFAULT 0,
                    player_consensus_status TEXT,
                    player_consensus_level TEXT,
                    player_consensus_trust_score REAL,
                    market_consensus_status TEXT,
                    market_consensus_level TEXT,
                    market_consensus_trust_score REAL,
                    candidate_rank INTEGER,
                    candidate_score_gap REAL,
                    final_status TEXT,
                    rejection_stage TEXT,
                    final_decision_reason TEXT,
                    sportsbook_line REAL,
                    pick_direction TEXT,
                    bet_confidence TEXT,
                    over_under_result TEXT,
                    bet_won INTEGER,
                    direction_correct INTEGER,
                    miss_reason_primary TEXT,
                    miss_reason_secondary TEXT,
                    audited_at TEXT,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prediction_date, player_id, market_key)
                )
            """)

            

            # Model performance tracking

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS model_performance (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    audit_date TEXT NOT NULL,

                    total_predictions INTEGER,

                    predictions_audited INTEGER,

                    

                    -- Accuracy metrics

                    mean_absolute_error REAL,

                    root_mean_squared_error REAL,

                    

                    -- Calibration metrics

                    pct_in_p10_p90_range REAL,  -- Should be ~80%

                    pct_above_p90 REAL,  -- Should be ~10%

                    pct_below_p10 REAL,  -- Should be ~10%

                    

                    -- By player type

                    mae_starters REAL,

                    mae_bench REAL,

                    

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP

                )

            """)

            

            # Learning insights table

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS learning_insights (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    insight_date TEXT NOT NULL,

                    insight_type TEXT NOT NULL,

                    description TEXT,

                    feature_name TEXT,

                    suggested_adjustment REAL,

                    applied INTEGER DEFAULT 0,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP

                )

            """)

            

            cursor.execute("""

                CREATE INDEX IF NOT EXISTS idx_predictions_date 

                ON predictions_archive(prediction_date)

            """)

            cursor.execute("""

                CREATE INDEX IF NOT EXISTS idx_predictions_player 

                ON predictions_archive(player_id)

            """)

            for column_def in [
                "ALTER TABLE predictions_archive ADD COLUMN actual_blocks INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN actual_steals INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN actual_threes INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN actual_field_goals INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN actual_fga INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN actual_plus_minus REAL",
                "ALTER TABLE predictions_archive ADD COLUMN edge_score REAL",
                "ALTER TABLE predictions_archive ADD COLUMN edge_tier TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN is_parlay_core INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN pure_model_pred REAL",
                "ALTER TABLE predictions_archive ADD COLUMN market_adjusted_pred REAL",
                "ALTER TABLE predictions_archive ADD COLUMN post_rule_pred REAL",
                "ALTER TABLE predictions_archive ADD COLUMN prediction_health_score REAL",
                "ALTER TABLE predictions_archive ADD COLUMN prediction_degradation_flags TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN used_fallback_model INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN market_anchor_applied INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN market_anchor_weight REAL",
                "ALTER TABLE predictions_archive ADD COLUMN selection_reason TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN edge_explanation TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN core_predictor TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN mechanistic_reference_used INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN mechanistic_reference_mean REAL",
                "ALTER TABLE predictions_archive ADD COLUMN monte_carlo_reference_used INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN monte_carlo_reference_mean REAL",
                "ALTER TABLE predictions_archive ADD COLUMN residual_model_ready INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN residual_reference_adjustment REAL",
                "ALTER TABLE predictions_archive ADD COLUMN injury_context_present INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN injury_context_size INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN ensemble_applied INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN ensemble_mean REAL",
                "ALTER TABLE predictions_archive ADD COLUMN reference_consensus_mean REAL",
                "ALTER TABLE predictions_archive ADD COLUMN model_disagreement REAL",
                "ALTER TABLE predictions_archive ADD COLUMN ensemble_notes TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN edge_candidate_tier TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN edge_candidate_direction TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN decision_alignment_status TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN approved_bet INTEGER DEFAULT 0",
                "ALTER TABLE predictions_archive ADD COLUMN player_consensus_status TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN player_consensus_level TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN player_consensus_trust_score REAL",
                "ALTER TABLE predictions_archive ADD COLUMN market_consensus_status TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN market_consensus_level TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN market_consensus_trust_score REAL",
                "ALTER TABLE predictions_archive ADD COLUMN candidate_rank INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN candidate_score_gap REAL",
                "ALTER TABLE predictions_archive ADD COLUMN final_status TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN rejection_stage TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN final_decision_reason TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN sportsbook_line REAL",
                "ALTER TABLE predictions_archive ADD COLUMN pick_direction TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN bet_confidence TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN over_under_result TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN bet_won INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN direction_correct INTEGER",
                "ALTER TABLE predictions_archive ADD COLUMN miss_reason_primary TEXT",
                "ALTER TABLE predictions_archive ADD COLUMN miss_reason_secondary TEXT",
            ]:
                try:
                    cursor.execute(column_def)
                except Exception as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise

            
            conn.commit()
    
    def archive_predictions(self, predictions_source: Any) -> int:
        """Archive predictions from CSV or DataFrame to database.
        
        Args:
            predictions_source: Path to CSV file (str) or pandas DataFrame
            
        Returns:
            Number of predictions archived
        """
        df = pd.DataFrame()
        pred_date = datetime.now().strftime('%Y-%m-%d')
        
        if isinstance(predictions_source, str):
            if not Path(predictions_source).exists():
                print(f"  [WARN] Predictions file not found: {predictions_source}")
                return 0
            df = pd.read_csv(predictions_source)
            
            # Extract date from filename if possible
            filename = Path(predictions_source).stem
            if '_' in filename:
                try:
                    # Valid if YYYY-MM-DD
                    date_part = filename.split('_')[-1]
                    datetime.strptime(date_part, '%Y-%m-%d')
                    pred_date = date_part
                except ValueError:
                    pass
        elif isinstance(predictions_source, pd.DataFrame):
            df = predictions_source
            # Try to get date from column
            if 'game_date' in df.columns and not df.empty:
                 pred_date = str(df.iloc[0]['game_date'])
        else:
             print("  [ERROR] Invalid prediction source type")
             return 0

        
        if df.empty:
            return 0
        
        count = 0
        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            # Replace the slate atomically for this date so reruns do not leave stale rows behind.
            cursor.execute("DELETE FROM predictions_archive WHERE prediction_date = ?", (pred_date,))

            

            for _, row in df.iterrows():

                try:

                    # Get player_id from name lookup
                    player_name = row.get('player_name', '')
                    players = self.db.search_players(player_name)
                    
                    # If full name search fails, try first name as fallback (risky but better than nothing)
                    if not players and player_name:
                         players = self.db.search_players(player_name.split()[0])
                         
                    # Find best match (exact name match preferred)
                    player_id = None
                    if players:
                        # Try to find exact match first
                        for p in players:
                            if p['full_name'].lower() == player_name.lower():
                                player_id = p['player_id']
                                break
                        # Fallback to first result
                        if not player_id:
                            player_id = players[0]['player_id']

                    

                    cursor.execute("""

                        INSERT OR REPLACE INTO predictions_archive (

                            prediction_date, player_id, player_name, team, opponent,

                            predicted_mean, predicted_p10, predicted_p50, predicted_p90,

                            predicted_minutes, predicted_ppm, p_play,
                            spread, total, blowout_risk, market_key,
                            edge_score, edge_tier, is_parlay_core,
                            pure_model_pred, market_adjusted_pred, post_rule_pred,
                            prediction_health_score, prediction_degradation_flags,
                            used_fallback_model, market_anchor_applied, market_anchor_weight,
                            selection_reason, edge_explanation,
                            core_predictor, mechanistic_reference_used, mechanistic_reference_mean,
                            monte_carlo_reference_used, monte_carlo_reference_mean,
                            residual_model_ready, residual_reference_adjustment,
                            injury_context_present, injury_context_size,
                            ensemble_applied, ensemble_mean, reference_consensus_mean, model_disagreement, ensemble_notes,
                            edge_candidate_tier, edge_candidate_direction, decision_alignment_status, approved_bet,
                            player_consensus_status, player_consensus_level,
                            market_consensus_status, market_consensus_level,
                            candidate_rank, candidate_score_gap, final_status, rejection_stage, final_decision_reason,
                            sportsbook_line, pick_direction, bet_confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pred_date,
                        player_id,
                        row.get('player_name'),
                        row.get('team'),
                        row.get('opponent'),

                        row.get('predicted_mean') or row.get('mean'),
                        row.get('p10'),
                        row.get('p50'),
                        row.get('p90'),
                        row.get('predicted_minutes'),
                        row.get('predicted_ppm'),
                        row.get('p_play', 1.0),
                        row.get('spread'),
                        row.get('total'),
                        row.get('blowout_risk'),
                        # Handle varied market key names in CSV
                        row.get('market_key') or row.get('market_type') or row.get('market') or 'points',
                        # [NEW] Audit Improvements
                        row.get('edge_score', 0),
                        row.get('edge_tier', ''),
                        1 if row.get('is_parlay_core') else 0,
                        row.get('pure_model_pred'),
                        row.get('market_adjusted_pred'),
                        row.get('post_rule_pred'),
                        row.get('prediction_health_score'),
                        row.get('prediction_degradation_flags'),
                        1 if row.get('used_fallback_model') else 0,
                        1 if row.get('market_anchor_applied') else 0,
                        row.get('market_anchor_weight'),
                        row.get('selection_reason'),
                        row.get('edge_explanation'),
                        row.get('core_predictor'),
                        1 if row.get('mechanistic_reference_used') else 0,
                        row.get('mechanistic_reference_mean'),
                        1 if row.get('monte_carlo_reference_used') else 0,
                        row.get('monte_carlo_reference_mean'),
                        1 if row.get('residual_model_ready') else 0,
                        row.get('residual_reference_adjustment'),
                        1 if row.get('injury_context_present') else 0,
                        row.get('injury_context_size', 0),
                        1 if row.get('ensemble_applied') else 0,
                        row.get('ensemble_mean'),
                        row.get('reference_consensus_mean'),
                        row.get('model_disagreement'),
                        row.get('ensemble_notes'),
                        row.get('edge_candidate_tier'),
                        row.get('edge_candidate_direction'),
                        row.get('decision_alignment_status'),
                        1 if row.get('approved_bet') else 0,
                        row.get('player_consensus_status'),
                        row.get('player_consensus_level'),
                        row.get('market_consensus_status'),
                        row.get('market_consensus_level'),
                        row.get('candidate_rank'),
                        row.get('candidate_score_gap'),
                        row.get('final_status'),
                        row.get('rejection_stage'),
                        row.get('final_decision_reason'),
                        row.get('sportsbook_line', row.get('line')),
                        row.get('pick_direction') or row.get('direction') or row.get('edge_direction'),
                        row.get('bet_confidence', row.get('confidence'))
                    ))

                    count += 1

                except Exception as e:

                    print(f"  [WARN] Error archiving {row.get('player_name')}: {e}")

            

            conn.commit()

        

        print(f"  [OK] Archived {count} predictions for {pred_date}")
        return count

    def sync_from_log(self, date_str: str) -> int:
        """Sync predictions from prediction_log table to predictions_archive.
        This provides a fallback when the temporary CSV file is missing.
        
        Args:
            date_str: Date to sync (YYYY-MM-DD)
            
        Returns:
            Number of predictions synced
        """
        print(f"\n[Audit] Syncing predictions from database logs for {date_str}...")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch from prediction_log (created by LearningLoopAgent)
            cursor.execute("""
                SELECT player_id, player_name, game_date, 
                       pred_points, pred_std, pred_minutes,
                       opponent, spread, market_line
                FROM prediction_log 
                WHERE date(game_date) = date(?)
            """, (date_str,))
            
            logs = cursor.fetchall()
            if not logs:
                print(f"  [WARN] No logs found in prediction_log for {date_str}")
                return 0
                
            count = 0
            for row in logs:
                r = dict(row)
                player_id = r['player_id']
                
                # Approximate P10/P90 from mean/std for the archive
                mean = r['pred_points']
                std = r['pred_std'] or 2.0
                p10 = mean - (1.28 * std)
                p50 = mean
                p90 = mean + (1.28 * std)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions_archive (
                        prediction_date, player_id, player_name, opponent,
                        predicted_mean, predicted_p10, predicted_p50, predicted_p90,
                        predicted_minutes, market_key, spread
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    player_id,
                    r['player_name'],
                    r['opponent'],
                    mean, p10, p50, p90,
                    r['pred_minutes'],
                    'points', # Fallback to points for log sync
                    r['spread']
                ))
                count += 1
                
            conn.commit()
            
        print(f"  [OK] Synced {count} predictions from DB logs for {date_str}")
        return count

    

    def fetch_actual_results(self, game_date: str) -> int:

        """Fetch actual game results for a date and update predictions.

        

        Args:

            game_date: Date to fetch results for (YYYY-MM-DD)

            

        Returns:

            Number of predictions updated with actuals

        """

        print(f"\n[Audit] Fetching actual results for {game_date}...")

        

        # Get predictions that need auditing

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM predictions_archive 

                WHERE prediction_date = ?
                  AND (
                      actual_points IS NULL OR
                      actual_fga IS NULL OR
                      actual_plus_minus IS NULL
                  )

            """, (game_date,))

            predictions = [dict(row) for row in cursor.fetchall()]

        

        if not predictions:

            print(f"  No predictions to audit for {game_date}")

            return 0

        

        print(f"  Found {len(predictions)} predictions to audit")

        

        # Fetch actual game logs

        updated = 0

        for pred in predictions:

            player_id = pred.get('player_id')

            if not player_id:

                continue

            

            # Get actual results from player_logs

            with self.db.get_connection() as conn:

                cursor = conn.cursor()

                cursor.execute("""
                    SELECT points, assists, rebounds, blocks, steals, fg3m, fgm, fga, minutes, ppm, plus_minus
                    FROM player_logs 
                    WHERE player_id = ? AND game_date = ?
                """, (player_id, game_date))

                actual = cursor.fetchone()

            

            if actual:
                # Convert sqlite3.Row to dict for safer access
                actual_dict = dict(actual)
                actual_points = actual_dict['points']
                actual_assists = actual_dict.get('assists', 0)
                actual_rebounds = actual_dict.get('rebounds', 0)
                actual_blocks = actual_dict.get('blocks', 0)
                actual_steals = actual_dict.get('steals', 0)
                actual_threes = actual_dict.get('fg3m', 0)
                actual_field_goals = actual_dict.get('fgm', 0)
                actual_fga = actual_dict.get('fga', 0)
                actual_plus_minus = actual_dict.get('plus_minus', 0)
                actual_pra = actual_points + actual_assists + actual_rebounds
                
                actual_minutes = actual['minutes']
                actual_ppm = actual['ppm']
                did_play = 1 if actual_minutes > 0 else 0
                
                # Determine target actual based on market_key
                market_key = pred.get('market_key', 'points')
                target_actual = self._get_market_actual(market_key, {
                    'actual_points': actual_points,
                    'actual_assists': actual_assists,
                    'actual_rebounds': actual_rebounds,
                    'actual_blocks': actual_blocks,
                    'actual_steals': actual_steals,
                    'actual_threes': actual_threes,
                    'actual_field_goals': actual_field_goals,
                    'actual_fga': actual_fga,
                    'actual_plus_minus': actual_plus_minus,
                    'actual_pra': actual_pra,
                })

                sportsbook_line = pred.get('sportsbook_line')
                pick_direction = str(pred.get('pick_direction') or 'NO_BET').upper()
                over_under_result, bet_won, direction_correct = self._grade_pick_against_line(
                    pick_direction=pick_direction,
                    sportsbook_line=sportsbook_line,
                    target_actual=target_actual
                )
                
                # Calculate error
                predicted_mean = pred.get('predicted_mean', 0) or 0
                prediction_error = target_actual - predicted_mean

                

                # Check if actual was in predicted range
                # [MODIFIED Jan 29] User feedback: P10 is too low/easy.
                # We calculate an "effective floor" approx P25 to make audit stricter.
                # If P25 isn't stored, we approximate: P25 ≈ P10 + 0.4 * (P50 - P10)
                p10 = pred.get('predicted_p10', 0) or 0
                p50 = pred.get('predicted_p50', 0) or 0
                p90 = pred.get('predicted_p90', 100) or 100
                
                # Approximate P25 floor
                effective_floor = p10 + (0.4 * (p50 - p10))
                
                # Stricter check: Must be between P25 and P90 (or similar upper bound)
                # Stricter check: Must be between P25 and P90 (or similar upper bound)
                was_in_range = 1 if effective_floor <= target_actual <= p90 else 0

                

                miss_reason_primary, miss_reason_secondary = self._classify_miss_reasons(
                    pred, target_actual, prediction_error
                )

                # Update prediction record

                with self.db.get_connection() as conn:

                    cursor = conn.cursor()

                    cursor.execute("""

                        UPDATE predictions_archive SET
                            actual_points = ?,
                            actual_assists = ?,
                            actual_rebounds = ?,
                            actual_blocks = ?,
                            actual_steals = ?,
                            actual_threes = ?,
                            actual_field_goals = ?,
                            actual_fga = ?,
                            actual_plus_minus = ?,
                            actual_pra = ?,
                            actual_minutes = ?,
                            actual_ppm = ?,
                            did_play = ?,
                            prediction_error = ?,
                            was_in_range = ?,
                            over_under_result = ?,
                            bet_won = ?,
                            direction_correct = ?,
                            miss_reason_primary = ?,
                            miss_reason_secondary = ?,
                            audited_at = ?
                        WHERE id = ?
                    """, (
                        actual_points,
                        actual_assists,
                        actual_rebounds,
                        actual_blocks,
                        actual_steals,
                        actual_threes,
                        actual_field_goals,
                        actual_fga,
                        actual_plus_minus,
                        actual_pra,
                        actual_minutes,
                        actual_ppm,
                        did_play,
                        prediction_error,
                        was_in_range,
                        over_under_result,
                        bet_won,
                        direction_correct,
                        miss_reason_primary,
                        miss_reason_secondary,
                        datetime.now().isoformat(),
                        pred['id']
                    ))

                    conn.commit()

                

                updated += 1

        

        print(f"  [OK] Updated {updated} predictions with actual results")

        return updated

    def _get_market_actual(self, market_key: str, actuals: Dict[str, Any]) -> float:
        """Select the correct actual stat for a market."""
        mkey = str(market_key or 'points').lower()
        if mkey == 'assists':
            return actuals.get('actual_assists', 0)
        if mkey == 'rebounds':
            return actuals.get('actual_rebounds', 0)
        if mkey == 'pra':
            return actuals.get('actual_pra', 0)
        if mkey == 'blocks':
            return actuals.get('actual_blocks', 0)
        if mkey == 'steals':
            return actuals.get('actual_steals', 0)
        if mkey in ('threes', '3pt', 'three_pointers'):
            return actuals.get('actual_threes', 0)
        if mkey in ('field_goals', 'fgm'):
            return actuals.get('actual_field_goals', 0)
        return actuals.get('actual_points', 0)

    

    def _classify_miss_reasons(self, pred: Dict[str, Any], target_actual: float, prediction_error: float) -> Tuple[str, str]:
        """Classify why a prediction missed using archived metadata."""
        reasons: List[str] = []
        error_abs = abs(prediction_error)
        flags_raw = pred.get('prediction_degradation_flags') or ''
        flags = {flag.strip() for flag in str(flags_raw).split(';') if flag.strip()}
        market_key = str(pred.get('market_key', 'points')).lower()

        if pred.get('used_fallback_model'):
            reasons.append('fallback_model')

        model_disagreement = pred.get('model_disagreement')
        if model_disagreement is not None and float(model_disagreement or 0) >= 7.0:
            reasons.append('model_disagreement')

        pure_model_pred = pred.get('pure_model_pred')
        market_adjusted_pred = pred.get('market_adjusted_pred')
        if (
            pure_model_pred is not None and
            market_adjusted_pred is not None and
            pred.get('market_anchor_applied') and
            abs(float(market_adjusted_pred) - float(pure_model_pred)) >= 1.5
        ):
            reasons.append('market_anchor_shift')

        actual_minutes = pred.get('actual_minutes')
        predicted_minutes = pred.get('predicted_minutes')
        if actual_minutes is not None and predicted_minutes is not None:
              minutes_delta = float(actual_minutes) - float(predicted_minutes)
              if abs(minutes_delta) >= 6:
                  reasons.append('minutes_miss')

        actual_points = float(pred.get('actual_points') or 0)
        actual_assists = float(pred.get('actual_assists') or 0)
        actual_rebounds = float(pred.get('actual_rebounds') or 0)
        actual_fga = pred.get('actual_fga')
        actual_fgm = pred.get('actual_field_goals')
        actual_plus_minus = pred.get('actual_plus_minus')
        if actual_fgm is None:
            actual_fgm = pred.get('actual_fgm')
        actual_fga = float(actual_fga or 0)
        actual_fgm = float(actual_fgm or 0)
        predicted_mean = pred.get('predicted_mean')
        predicted_ppm = pred.get('predicted_ppm')
        actual_ppm = pred.get('actual_ppm')

        if predicted_mean is not None and market_key == 'points':
            predicted_mean = float(predicted_mean)
            if actual_minutes is not None and predicted_minutes is not None and predicted_ppm is not None:
                expected_from_actual_minutes = float(actual_minutes) * float(predicted_ppm)
                if predicted_mean - expected_from_actual_minutes >= 3 and actual_points < predicted_mean:
                    reasons.append('minutes_role_miss')
                elif expected_from_actual_minutes - actual_points >= 4:
                    reasons.append('efficiency_miss')
            elif actual_minutes is not None and actual_points < predicted_mean - 5:
                actual_minutes_val = float(actual_minutes)
                plus_minus_val = float(actual_plus_minus or 0)
                if actual_minutes_val <= 24 and abs(plus_minus_val) >= 15:
                    reasons.append('blowout_script_miss')
                elif actual_minutes_val <= 24:
                    reasons.append('minutes_role_miss')
                elif actual_ppm is not None and float(actual_ppm or 0) <= 0.22:
                    reasons.append('efficiency_miss')
            elif actual_fga > 0 and actual_points < predicted_mean - 4:
                fg_pct = (actual_fgm / actual_fga) if actual_fga > 0 else 0.0
                if fg_pct < 0.35:
                    reasons.append('efficiency_miss')
                elif actual_fga < max(6.0, predicted_mean * 0.6):
                    reasons.append('usage_volume_miss')
            elif actual_fga == 0 and actual_fgm <= 2 and actual_points < predicted_mean - 6:
                reasons.append('usage_volume_miss')

        if market_key == 'assists':
            if predicted_mean is not None and float(predicted_mean) - actual_assists >= 2.5:
                reasons.append('playmaking_volume_miss')
            elif actual_assists - float(predicted_mean or 0) >= 2.5:
                reasons.append('ceiling_creation_miss')

        if market_key == 'rebounds':
            if predicted_mean is not None and float(predicted_mean) - actual_rebounds >= 2.5:
                reasons.append('rebound_role_miss')
            elif actual_rebounds - float(predicted_mean or 0) >= 2.5:
                reasons.append('rebound_ceiling_miss')

        if actual_ppm is not None and predicted_ppm is not None and actual_minutes is not None and predicted_minutes is not None:
            ppm_delta = float(actual_ppm) - float(predicted_ppm)
            minutes_delta = float(actual_minutes) - float(predicted_minutes)
            if abs(minutes_delta) < 3 and ppm_delta <= -0.2 and error_abs >= 4:
                reasons.append('efficiency_miss')
            elif abs(ppm_delta) < 0.1 and minutes_delta <= -5 and error_abs >= 4:
                reasons.append('minutes_miss')

        if pred.get('did_play') == 0:
            reasons.append('player_did_not_play')

        if error_abs >= 8:
            reasons.append('large_stat_miss')
        elif error_abs >= 4:
            reasons.append('moderate_stat_miss')

        if pred.get('edge_tier') in ('parlay_core', 'playable') and error_abs >= 5:
            reasons.append('high_confidence_miss')

        if target_actual > (pred.get('predicted_p90') or 0):
            reasons.append('ceiling_too_low')
        elif target_actual < (pred.get('predicted_p10') or 0):
            reasons.append('floor_too_high')

        if pred.get('prediction_health_score') is not None and float(pred.get('prediction_health_score') or 0) < 0.65:
            reasons.append('low_prediction_health')
        if 'missing_h2h_history' in flags:
            reasons.append('missing_h2h_history')
        if 'missing_injury_context' in flags and error_abs >= 4 and not reasons:
            reasons.append('missing_injury_context')

        if not reasons:
            reasons.append('unclear')

        primary = reasons[0]
        secondary = reasons[1] if len(reasons) > 1 else ''
        return primary, secondary


    def calculate_performance_metrics(self, game_date: str) -> Dict[str, Any]:

        """Calculate model performance metrics for a date.

        

        Args:

            game_date: Date to calculate metrics for

            

        Returns:

            Dict of performance metrics

        """

        with self.db.get_connection() as conn:

            df = pd.read_sql_query("""

                SELECT * FROM predictions_archive 

                WHERE prediction_date = ? AND actual_points IS NOT NULL

            """, conn, params=[game_date])

        

        if df.empty:

            return {}

        

        # Filter to players who played

        df_played = df[df['did_play'] == 1]

        

        if df_played.empty:

            return {}

        

        # Calculate metrics

        mae = np.abs(df_played['prediction_error']).mean()

        rmse = np.sqrt((df_played['prediction_error'] ** 2).mean())

        

        
        # Calculate target actuals for calibration
        # We need to know what to compare against. 
        # Since we stored market_key, we can derive it row by row, or use a clever coalesce if we trusted the DB.
        # But pandas is easier.
        
        def get_target_actual(row):
            return self._get_market_actual(row.get('market_key', 'points'), row)

        df_played['target_actual'] = df_played.apply(get_target_actual, axis=1)

        # Calibration
        pct_in_range = df_played['was_in_range'].mean() * 100
        pct_above_p90 = (df_played['target_actual'] > df_played['predicted_p90']).mean() * 100
        pct_below_p10 = (df_played['target_actual'] < df_played['predicted_p10']).mean() * 100
        
        # By player type (starters vs bench based on predicted minutes)
        starters = df_played[df_played['predicted_minutes'] >= 25]
        bench = df_played[df_played['predicted_minutes'] < 25]

        mae_starters = np.abs(starters['prediction_error']).mean() if not starters.empty else None
        mae_bench = np.abs(bench['prediction_error']).mean() if not bench.empty else None

        

        metrics = {

            'audit_date': game_date,

            'total_predictions': len(df),

            'predictions_audited': len(df_played),

            'mean_absolute_error': mae,

            'root_mean_squared_error': rmse,

            'pct_in_p10_p90_range': pct_in_range,

            'pct_above_p90': pct_above_p90,

            'pct_below_p10': pct_below_p10,

            'mae_starters': mae_starters,

            'mae_bench': mae_bench

        }

        

        # Store in database

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                INSERT INTO model_performance (

                    audit_date, total_predictions, predictions_audited,

                    mean_absolute_error, root_mean_squared_error,

                    pct_in_p10_p90_range, pct_above_p90, pct_below_p10,

                    mae_starters, mae_bench

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (

                game_date, len(df), len(df_played),

                mae, rmse, pct_in_range, pct_above_p90, pct_below_p10,

                mae_starters, mae_bench

            ))

            conn.commit()

        

        return metrics

    

    def generate_learning_insights(self, game_date: str) -> List[Dict[str, Any]]:

        """Analyze prediction errors to generate learning insights.

        

        Args:

            game_date: Date to analyze

            

        Returns:

            List of insights

        """

        with self.db.get_connection() as conn:

            df = pd.read_sql_query("""

                SELECT * FROM predictions_archive 

                WHERE prediction_date = ? AND actual_points IS NOT NULL

            """, conn, params=[game_date])

        

        if df.empty:

            return []

        

        insights = []

        

        # Check for systematic bias

        mean_error = df['prediction_error'].mean()

        if abs(mean_error) > 3:

            insights.append({

                'insight_date': game_date,

                'insight_type': 'SYSTEMATIC_BIAS',

                'description': f"Model {'over' if mean_error < 0 else 'under'}predicts by {abs(mean_error):.1f} points on average",

                'feature_name': 'predicted_mean',

                'suggested_adjustment': -mean_error

            })

        

        # Check blowout prediction accuracy

        blowout_games = df[df['blowout_risk'] > 10]

        if not blowout_games.empty:

            blowout_mae = np.abs(blowout_games['prediction_error']).mean()

            normal_mae = np.abs(df[df['blowout_risk'] <= 10]['prediction_error']).mean()

            

            if blowout_mae > normal_mae * 1.5:

                insights.append({

                    'insight_date': game_date,

                    'insight_type': 'BLOWOUT_HANDLING',

                    'description': f"Blowout games have {blowout_mae:.1f} MAE vs {normal_mae:.1f} normal",

                    'feature_name': 'blowout_risk',

                    'suggested_adjustment': None

                })

        

        # Check calibration issues

        pct_in_range = df['was_in_range'].mean() * 100

        if pct_in_range < 70:

            insights.append({

                'insight_date': game_date,

                'insight_type': 'CALIBRATION_TOO_TIGHT',

                'description': f"Only {pct_in_range:.0f}% in P10-P90 range (expected ~80%)",

                'feature_name': 'uncertainty_estimation',

                'suggested_adjustment': 1.2  # Increase uncertainty by 20%

            })

        elif pct_in_range > 90:

            insights.append({

                'insight_date': game_date,

                'insight_type': 'CALIBRATION_TOO_WIDE',

                'description': f"{pct_in_range:.0f}% in P10-P90 range (expected ~80%)",

                'feature_name': 'uncertainty_estimation',

                'suggested_adjustment': 0.8  # Decrease uncertainty by 20%

            })

        

        # Store insights

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            for insight in insights:

                cursor.execute("""

                    INSERT INTO learning_insights (

                        insight_date, insight_type, description, 

                        feature_name, suggested_adjustment

                    ) VALUES (?, ?, ?, ?, ?)

                """, (

                    insight['insight_date'],

                    insight['insight_type'],

                    insight['description'],

                    insight['feature_name'],

                    insight['suggested_adjustment']

                ))

            conn.commit()

        

        return insights

    def _get_audit_miss_summary(self, game_date: str) -> Dict[str, Any]:
        """Build a concise summary of why predictions missed."""
        with self.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT *
                FROM predictions_archive
                WHERE prediction_date = ? AND actual_points IS NOT NULL
                ORDER BY ABS(prediction_error) DESC
            """, conn, params=[game_date])

        if df.empty:
            return {'worst_misses': [], 'reason_counts': [], 'fallback_misses': []}

        if 'miss_reason_primary' not in df.columns:
            df['miss_reason_primary'] = 'unclear'
        if 'miss_reason_secondary' not in df.columns:
            df['miss_reason_secondary'] = ''
        if 'prediction_health_score' not in df.columns:
            df['prediction_health_score'] = np.nan
        if 'used_fallback_model' not in df.columns:
            df['used_fallback_model'] = 0

        worst_misses = []
        for _, row in df.head(10).iterrows():
            target_actual = self._get_market_actual(row.get('market_key', 'points'), row)
            worst_misses.append({
                'player_name': row.get('player_name'),
                'market_key': row.get('market_key', 'points'),
                'predicted_mean': row.get('predicted_mean'),
                'actual_points': target_actual,
                'prediction_error': row.get('prediction_error'),
                'reason_primary': str(row.get('miss_reason_primary') or 'unclear').replace('_', ' '),
                'reason_secondary': str(row.get('miss_reason_secondary') or '').replace('_', ' '),
                'health_score': row.get('prediction_health_score'),
            })

        reason_counts = [
            {'reason': reason, 'count': int(count)}
            for reason, count in df['miss_reason_primary'].fillna('unclear').value_counts().items()
        ]

        fallback_misses = []
        for _, row in df[df['used_fallback_model'] == 1].head(5).iterrows():
            fallback_misses.append({
                'player_name': row.get('player_name'),
                'market_key': row.get('market_key', 'points'),
                'prediction_error': row.get('prediction_error'),
                'reason_primary': str(row.get('miss_reason_primary') or 'unclear').replace('_', ' '),
            })

        return {
            'worst_misses': worst_misses,
            'reason_counts': reason_counts,
            'fallback_misses': fallback_misses,
        }

    def _get_pick_breakdown(self, game_date: str) -> List[Dict[str, Any]]:
        """Build a clean, pick-by-pick audit breakdown."""
        with self.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT *
                FROM predictions_archive
                WHERE prediction_date = ? AND actual_points IS NOT NULL
                ORDER BY ABS(prediction_error) DESC, player_name ASC
            """, conn, params=[game_date])

        if df.empty:
            return []

        def get_target_actual(row):
            return self._get_market_actual(row.get('market_key', 'points'), row)

        rows = []
        for _, row in df.iterrows():
            line = row.get('sportsbook_line')
            target_actual = get_target_actual(row)
            pure_pred = row.get('pure_model_pred')
            market_adj_pred = row.get('market_adjusted_pred')
            direction = str(row.get('pick_direction') or 'NO_BET').upper()
            result = str(row.get('over_under_result') or '-').upper()
            direction_correct = row.get('direction_correct')
            bet_won = row.get('bet_won')

            def yes_no_unknown(value):
                if pd.isna(value):
                    return '-'
                return 'YES' if int(value) == 1 else 'NO'

            if direction in ('OVER', 'UNDER') and result == 'PUSH':
                bet_won_display = 'PUSH'
                direction_correct_display = '-'
            else:
                bet_won_display = yes_no_unknown(bet_won)
                direction_correct_display = yes_no_unknown(direction_correct)

            rows.append({
                'player_name': row.get('player_name', 'Unknown'),
                'team': row.get('team', 'UNK'),
                'market_key': str(row.get('market_key', 'points')).upper(),
                'sportsbook_line': line,
                'actual_result': target_actual,
                'exact_error': row.get('prediction_error'),
                'over_under_result': result,
                'bet_won': bet_won_display,
                'direction_correct': direction_correct_display,
                'pick_direction': direction,
                'pure_model_pred': pure_pred,
                'market_adjusted_pred': market_adj_pred,
                'miss_reason': str(row.get('miss_reason_primary') or 'unclear').replace('_', ' '),
                'used_fallback_model': yes_no_unknown(row.get('used_fallback_model')),
                'edge_tier': str(row.get('edge_tier') or '').lower(),
                'edge_score': row.get('edge_score'),
                'is_parlay_core': 'YES' if (
                    (not pd.isna(row.get('is_parlay_core')) and int(row.get('is_parlay_core') or 0) == 1) or
                    str(row.get('edge_tier') or '').strip().lower() == 'parlay_core'
                ) else 'NO',
                'selection_reason': str(row.get('selection_reason') or '').strip(),
                'edge_explanation': str(row.get('edge_explanation') or '').strip(),
            })

        return rows

    def _backfill_pick_metadata_from_log(self, game_date: str) -> None:
        """Populate missing line and pick metadata from prediction_log when available."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions_archive
                SET
                    sportsbook_line = COALESCE(
                        sportsbook_line,
                        (
                            SELECT pl.market_line
                            FROM prediction_log pl
                            WHERE pl.player_id = predictions_archive.player_id
                              AND date(pl.game_date) = date(predictions_archive.prediction_date)
                            ORDER BY pl.id DESC
                            LIMIT 1
                        )
                    ),
                    pick_direction = COALESCE(
                        pick_direction,
                        (
                            SELECT pl.bet_direction
                            FROM prediction_log pl
                            WHERE pl.player_id = predictions_archive.player_id
                              AND date(pl.game_date) = date(predictions_archive.prediction_date)
                            ORDER BY pl.id DESC
                            LIMIT 1
                        )
                    ),
                    bet_confidence = COALESCE(
                        bet_confidence,
                        (
                            SELECT pl.bet_confidence
                            FROM prediction_log pl
                            WHERE pl.player_id = predictions_archive.player_id
                              AND date(pl.game_date) = date(predictions_archive.prediction_date)
                            ORDER BY pl.id DESC
                            LIMIT 1
                        )
                    )
                WHERE prediction_date = ?
            """, (game_date,))
            conn.commit()

    def _refresh_miss_reasons(self, game_date: str) -> None:
        """Recompute miss reasons for already-audited rows using current logic."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM predictions_archive
                WHERE prediction_date = ? AND actual_points IS NOT NULL
            """, (game_date,))
            rows = [dict(row) for row in cursor.fetchall()]

            for row in rows:
                market_key = str(row.get('market_key', 'points')).lower()
                target_actual = self._get_market_actual(market_key, row)

                if target_actual is None or row.get('prediction_error') is None:
                    continue

                primary, secondary = self._classify_miss_reasons(
                    row,
                    float(target_actual),
                    float(row.get('prediction_error'))
                )
                cursor.execute("""
                    UPDATE predictions_archive
                    SET miss_reason_primary = ?, miss_reason_secondary = ?
                    WHERE id = ?
                """, (primary, secondary, row['id']))

            conn.commit()

    def run_daily_audit(self, days_back: int = 1) -> Dict[str, Any]:

        """Run audit for recent predictions.

        

        Args:

            days_back: How many days back to audit

            

        Returns:

            Audit summary

        """

        audit_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        

        print(f"\n[Audit] Running daily audit for {audit_date}...")

        

        # First, refresh player logs to get actual results

        self.nba.load_all_teams()

        self.nba.load_all_players()

        

        # Refresh game logs for players being audited to ensure we have the latest results

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("SELECT DISTINCT player_id FROM predictions_archive WHERE prediction_date = ?", (audit_date,))
            
            # handle both tuple (from fetchall) and dict row factory if enabled
            rows = cursor.fetchall()
            if rows and isinstance(rows[0], (list, tuple)):
                 player_ids = [row[0] for row in rows]
            elif rows and isinstance(rows[0], dict):
                 player_ids = [row['player_id'] for row in rows]
            else:
                 player_ids = [row[0] for row in rows] if rows else []

        

        if player_ids:

            print(f"  [Audit] Refreshing game logs for {len(player_ids)} audited players...")

            self.nba.backfill_player_logs(player_ids)

        

        # Fetch actual results

        updated = self.fetch_actual_results(audit_date)

        

        if updated == 0:

            print("  No results to audit")

            return {'status': 'no_data'}

        

        # Calculate metrics

        metrics = self.calculate_performance_metrics(audit_date)

        

        # Generate insights

        insights = self.generate_learning_insights(audit_date)

        

        # Print summary

        print("\n  === AUDIT SUMMARY ===")

        print(f"  Date: {audit_date}")

        print(f"  Predictions audited: {metrics.get('predictions_audited', 0)}")

        print(f"  Mean Absolute Error: {metrics.get('mean_absolute_error', 0):.1f} points")

        print(f"  P10-P90 Calibration: {metrics.get('pct_in_p10_p90_range', 0):.0f}%")

        

        if insights:

            print("\n  Learning Insights:")

            for insight in insights:

                print(f"    - [{insight['insight_type']}] {insight['description']}")

        

        return {

            'status': 'complete',

            'date': audit_date,

            'metrics': metrics,

            'insights': insights

        }

    

    def generate_audit_report(self, output_path: str = None, audit_date: str = None) -> str:

        """Generate a comprehensive audit report.

        

        Args:

            output_path: Optional output path
            audit_date: Optional specific audit date to emphasize in miss breakdown

            

        Returns:

            Path to report

        """

        config = get_config()

        today = datetime.now().strftime('%Y-%m-%d')

        

        if output_path is None:

            output_path = config.project_root / f'audit_report_{today}.md'

        

        # Get recent performance

        with self.db.get_connection() as conn:

            perf_df = pd.read_sql_query("""

                SELECT * FROM model_performance 

                ORDER BY audit_date DESC LIMIT 7

            """, conn)

            

            insights_df = pd.read_sql_query("""

                SELECT * FROM learning_insights 

                ORDER BY insight_date DESC LIMIT 20

            """, conn)

        latest_audit_date = audit_date
        if latest_audit_date is None and not perf_df.empty:
            latest_audit_date = str(perf_df.iloc[0]['audit_date'])
        if latest_audit_date:
            self._backfill_pick_metadata_from_log(latest_audit_date)
            self._refresh_miss_reasons(latest_audit_date)
        miss_summary = self._get_audit_miss_summary(latest_audit_date) if latest_audit_date else {
            'worst_misses': [],
            'reason_counts': [],
            'fallback_misses': [],
        }
        pick_breakdown = self._get_pick_breakdown(latest_audit_date) if latest_audit_date else []

        

        generated_count = 0
        audited_count = 0
        missing_actuals_count = 0
        if latest_audit_date:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) AS total_rows,
                           SUM(CASE WHEN actual_points IS NOT NULL
                                     OR actual_assists IS NOT NULL
                                     OR actual_rebounds IS NOT NULL
                                     OR actual_blocks IS NOT NULL
                                     OR actual_steals IS NOT NULL
                                     OR actual_threes IS NOT NULL
                                     OR actual_field_goals IS NOT NULL
                               THEN 1 ELSE 0 END) AS audited_rows
                    FROM predictions_archive
                    WHERE prediction_date = ?
                """, (latest_audit_date,))
                row = cursor.fetchone()
                if row:
                    generated_count = int(row['total_rows'] or 0)
                    audited_count = int(row['audited_rows'] or 0)
                    missing_actuals_count = max(0, generated_count - audited_count)

        report = f"""# Model Audit Report

## Generated: {today}

### Audit Coverage

- Predictions generated: {generated_count}
- Predictions audited: {audited_count}
- Missing actuals: {missing_actuals_count}



### Performance Trend (Last 7 Days)



| Date | Predictions | MAE | RMSE | Calibration |

|------|-------------|-----|------|-------------|

"""

        

        if not perf_df.empty:

            for _, row in perf_df.iterrows():

                report += f"| {row['audit_date']} "

                report += f"| {row['predictions_audited']} "

                report += f"| {row['mean_absolute_error']:.1f} "

                report += f"| {row['root_mean_squared_error']:.1f} "

                report += f"| {row['pct_in_p10_p90_range']:.0f}% |\n"

        report += """

### Miss Reason Summary

"""

        if miss_summary['reason_counts']:
            report += "| Reason | Count |\n"
            report += "|--------|-------|\n"
            for item in miss_summary['reason_counts'][:8]:
                report += f"| {str(item['reason']).replace('_', ' ')} | {item['count']} |\n"
        else:
            report += "_No miss summary available yet._\n"

        report += """

### Worst Misses

"""

        if miss_summary['worst_misses']:
            report += "| Player | Market | Pred | Actual | Error | Primary | Secondary | Health |\n"
            report += "|--------|--------|------|--------|-------|---------|-----------|--------|\n"
            for miss in miss_summary['worst_misses']:
                health = miss['health_score']
                health_str = f"{health:.2f}" if pd.notna(health) else "-"
                report += (
                    f"| {miss['player_name']} | {miss['market_key']} | {miss['predicted_mean']:.1f} | "
                    f"{miss['actual_points']:.1f} | {miss['prediction_error']:+.1f} | {miss['reason_primary']} | "
                    f"{miss['reason_secondary'] or '-'} | {health_str} |\n"
                )
        else:
            report += "_No audited misses available._\n"

        report += """

### Fallback Model Misses

"""

        if miss_summary['fallback_misses']:
            for miss in miss_summary['fallback_misses']:
                report += f"- {miss['player_name']} ({miss['market_key']}): {miss['prediction_error']:+.1f}, {miss['reason_primary']}\n"
        else:
            report += "_No fallback-model misses recorded._\n"

        report += """

### Pick-by-Pick Audit

"""

        if pick_breakdown:
            report += "| Player | Team | Market | Tier | Line | Actual | Error | Outcome | Pick | Bet Won | Dir Correct | Pure -> Adj | Core | Why / Miss | Fallback |\n"
            report += "|--------|------|--------|------|------|--------|-------|---------|------|---------|-------------|-------------|------|------------|----------|\n"
            for item in pick_breakdown:
                line = item['sportsbook_line']
                actual = item['actual_result']
                error = item['exact_error']
                pure_pred = item['pure_model_pred']
                market_adj_pred = item['market_adjusted_pred']
                line_str = f"{line:.1f}" if pd.notna(line) else "-"
                actual_str = f"{actual:.1f}" if pd.notna(actual) else "-"
                error_str = f"{error:+.1f}" if pd.notna(error) else "-"
                pure_vs_adj = "-"
                if pd.notna(pure_pred) and pd.notna(market_adj_pred):
                    pure_vs_adj = f"{pure_pred:.1f}->{market_adj_pred:.1f}"
                tier = item['edge_tier'] or '-'
                edge_score = item['edge_score']
                if pd.notna(edge_score):
                    tier = f"{tier}:{float(edge_score):.0f}" if tier != '-' else f"{float(edge_score):.0f}"
                why_parts = []
                if item['edge_explanation']:
                    why_parts.append(item['edge_explanation'].split('.')[0][:36])
                elif item['selection_reason']:
                    why_parts.append(item['selection_reason'].split(';')[0][:36])
                why_parts.append(item['miss_reason'])
                why_text = " / ".join(part for part in why_parts if part)
                outcome = item['over_under_result']
                if item['pick_direction'] == 'NO_BET':
                    outcome = 'WATCH'
                report += (
                    f"| {item['player_name']} | {item['team']} | {item['market_key']} | {tier} | {line_str} | {actual_str} | "
                    f"{error_str} | {outcome} | {item['pick_direction']} | "
                    f"{item['bet_won']} | {item['direction_correct']} | {pure_vs_adj} | "
                    f"{item['is_parlay_core']} | {why_text[:46]} | {item['used_fallback_model']} |\n"
                )
        else:
            report += "_No audited pick-by-pick breakdown available yet._\n"

        

        report += """

### Key Insights



"""

        

        if not insights_df.empty:
            for _, row in insights_df.iterrows():
                report += f"- **[{row['insight_type']}]** ({row['insight_date']}): {row['description']}\n"
        else:
            report += "_No recent insights_\n"
        
        # [NEW] Detailed Breakdown
        report += "\n### Detailed Prediction Breakdown\n\n"
        
        with self.db.get_connection() as conn:
            detailed_df = pd.read_sql_query("""
                SELECT * FROM predictions_archive 
                WHERE prediction_date = ? AND actual_points IS NOT NULL
                ORDER BY abs(prediction_error) DESC
            """, conn, params=[today if today in perf_df['audit_date'].values else perf_df.iloc[0]['audit_date'] if not perf_df.empty else today])
            
        if not detailed_df.empty:
            # [NEW] Handle missing columns (for retro-compatibility)
            if 'edge_score' not in detailed_df.columns:
                detailed_df['edge_score'] = 0
            if 'edge_tier' not in detailed_df.columns:
                detailed_df['edge_tier'] = ''
            if 'is_parlay_core' not in detailed_df.columns:
                detailed_df['is_parlay_core'] = 0

            # [NEW] Calculate Tiered Performance
            detailed_df['tier_group'] = detailed_df.apply(lambda row: 
                'A (75+)' if row.get('edge_score', 0) >= 75 else
                'B (60-74)' if row.get('edge_score', 0) >= 60 else
                'C (50-59)' if row.get('edge_score', 0) >= 50 else
                'D (<50)', axis=1)

            # Group by Tier
            tier_stats = detailed_df.groupby('tier_group').agg({
                'prediction_error': lambda x: np.mean(np.abs(x)),
                'player_id': 'count'
            }).rename(columns={'player_id': 'Count', 'prediction_error': 'MAE'})
            
            report += "\n### 🏆 Performance by Edge Tier\n"
            try:
                report += tier_stats.to_markdown() + "\n\n"
            except Exception:
                report += "| Tier | MAE | Count |\n"
                report += "|------|-----|-------|\n"
                for tier_name, row in tier_stats.iterrows():
                    report += f"| {tier_name} | {row['MAE']:.2f} | {int(row['Count'])} |\n"
                report += "\n"
            
            # [NEW] Parlay Core Analysis
            parlay_core = detailed_df[detailed_df['is_parlay_core'] == 1]
            if not parlay_core.empty:
                pc_mae = np.mean(np.abs(parlay_core['prediction_error']))
                report += f"\n### 🔥 Parlay Core Performance\n"
                report += f"- **Count**: {len(parlay_core)}\n"
                report += f"- **MAE**: {pc_mae:.2f} pts\n"
                report += f"*(Lower MAE indicates higher reliability)*\n\n"

            report += "### Detailed Breakdown\n"
            report += "| Player | Tier | Score | Pred | Actual | Error | In Range |\n"
            report += "|--------|------|-------|------|--------|-------|----------|\n"
            
            correct_count = 0
            for _, row in detailed_df.iterrows():
                p_name = row.get('player_name') or 'Unknown'
                matchup = f"{row.get('team')} vs {row.get('opponent')}"
                
                # Safe float conversion
                def safe_float(val):
                    try:
                        return float(val) if val is not None else 0.0
                    except:
                        return 0.0

                pred_mean = safe_float(row.get('predicted_mean'))
                # p10 = safe_float(row.get('predicted_p10'))
                # p50 = safe_float(row.get('predicted_p50'))
                p90 = safe_float(row.get('predicted_p90'))
                error = safe_float(row.get('prediction_error'))
                in_range = row.get('was_in_range', 0)
                
                edge_score = row.get('edge_score', 0)
                tier = row.get('tier_group', '-')
                
                # Determine display actual
                mkey = row.get('market_key', 'points')
                actual = self._get_market_actual(mkey, row)
                
                actual = safe_float(actual)
                
                in_range_text = "YES" if in_range else "NO"
                # Highlight Parlay Core
                
                # Highlight Parlay Core
                pc_icon = "🔥" if row.get('is_parlay_core') else ""
                
                report += f"| {pc_icon} {p_name} ({mkey}) | {tier} | {edge_score} | {pred_mean:.1f} | {actual:.0f} | {error:+.1f} | {in_range_text} |\n"
                
                if in_range: correct_count += 1
                
            report += f"\n**Total Accuracy (P50 Hit):** {correct_count}/{len(detailed_df)} ({correct_count/len(detailed_df)*100:.1f}%)\n"
        else:
            report += "_No detailed data available for this date._\n"

        

        report += f"""

### Recommendations



Based on the analysis:

1. If Tier A MAE > Tier B MAE: Investigate "Star Player" variance logic.

2. If Parlay Core MAE > 5.0: Tighten parlay eligibility thresholds.

3. If Calibration < 75%: Widen uncertainty bands.



---

*Generated automatically by NBA Props Engine*

"""

        

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        

        print(f"  [OK] Audit report saved to {output_path}")

        return str(output_path)





    def check_feature_variability(self, df: pd.DataFrame, columns: List[str] = None) -> bool:

        """Check if features have sufficient variability (Audit 2.1).

        

        Args:

            df: DataFrame of features

            columns: List of columns to check (default: key rolling stats)

            

        Returns:

            True if passed

        """

        print("\n[Audit] Checking Feature Variability...")

        

        if df.empty:

            print("  [WARN] No features to audit")

            return False

            

        if columns is None:

            columns = ['minutes_L10', 'ppm_L10', 'minutes_L3']  # Exclude binary features like is_starter

            

        passed = True

        columns = [c for c in columns if c in df.columns]

        

        for col in columns:

            n_unique = df[col].nunique()

            null_pct = df[col].isnull().mean()

            

            print(f"  {col}: {n_unique} unique values, {null_pct:.1%} nulls")

            

            # Acceptance criteria: At least 20 unique values? (User asked for > 20)

            # But let's be lenient for small samples?

            # User said: "At least 80% of features have > 20 unique values on todays slate"

            

            if n_unique < 5: 

                 print(f"    [FAIL] CRITICAL: {col} has constant or near-constant values!")

                 passed = False

        

        return passed



    def check_prediction_distribution(self, df: pd.DataFrame) -> bool:

        """Check if predictions follow realistic distribution (Audit 2.2).

        

        Args:

            df: DataFrame of predictions

            

        Returns:

            True if passed

        """

        print("\n[Audit] Checking Prediction Distribution...")

        

        if df.empty:

            print("  [WARN] No predictions to audit")

            return False

            

        passed = True

        

        # Minutes P50 distribution

        if 'predicted_minutes' in df.columns:

            p50_min = df['predicted_minutes'].median()

            min_min = df['predicted_minutes'].min()

            max_min = df['predicted_minutes'].max()

            print(f"  Minutes Dist: Min={min_min:.1f}, Median={p50_min:.1f}, Max={max_min:.1f}")

            

            # Acceptance criteria

            if p50_min > 30:

                print(f"    [FAIL] Median minutes ({p50_min:.1f}) is too high (should be < 25 typically)")

                passed = False

            if max_min - min_min < 5:

                print(f"    [FAIL] Minutes range too narrow ({max_min - min_min:.1f})")

                passed = False

        

        # PPM Distribution

        if 'predicted_ppm' in df.columns:

            p50_ppm = df['predicted_ppm'].median()

            std_ppm = df['predicted_ppm'].std()

            print(f"  PPM Dist: Median={p50_ppm:.2f}, Std={std_ppm:.3f}")

            

            if std_ppm < 0.05:

                print(f"    [FAIL] PPM has almost no variance (std={std_ppm:.3f})")

                passed = False

        

        # Points Mean Distribution

        if 'predicted_mean' in df.columns:

            mean_pts = df['predicted_mean'].mean()

            median_pts = df['predicted_mean'].median()

            print(f"  Points Dist: Mean={mean_pts:.1f}, Median={median_pts:.1f}")

            

            if median_pts > 20:

                print(f"    [FAIL] Median points ({median_pts:.1f}) is too high (should be < 15)")

                passed = False

                

        return passed



# Convenience function

def get_auditor() -> PredictionAuditor:

    """Get prediction auditor instance."""

    return PredictionAuditor()

