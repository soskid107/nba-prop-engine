"""

Monte Carlo Simulation Engine



Combines Minutes and PPM models to generate probabilistic point forecasts.

Handles injury uncertainty through mixture models.

"""



import numpy as np

import pandas as pd

from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple



from ..utils.config import get_config

from ..utils.database import DatabaseManager

from ..models.feature_engineering import FeatureEngineer
from ..models.training import ModelTrainer
from ..models.usage_model import UsageModel
from ..models.efficiency_model import EfficiencyModel, FreeThrowModel
from ..models.variance_model import VarianceModel, PaceAdjuster
from ..ingestion.injury_ingestion import InjuryIngestion
from ..ingestion.odds_ingestion import OddsIngestion
from ..ingestion.nba_ingestion import NBAIngestion





class SimulationEngine:

    """Monte Carlo simulation for player point predictions."""

    

    def __init__(self, db: Optional[DatabaseManager] = None, **kwargs):

        """Initialize simulation engine.

        

        Args:

            db: Optional database manager
            **kwargs: Ablation flags

        """

        self.config = get_config()

        self.db = db or DatabaseManager()
        
        # [ABLATION] Flags to disable specific components for testing
        self.ablation_flags = kwargs.get('ablation_flags', {})

        self.n_sims = self.config.n_simulations

        

        # Initialize components
        self.fe = FeatureEngineer(self.db)
        self.trainer = ModelTrainer(self.db)
        self.usage_model = UsageModel(self.db)
        self.efficiency_model = EfficiencyModel(self.db)
        self.ft_model = FreeThrowModel(self.db)
        self.variance_model = VarianceModel(self.db)  # Player-specific variance
        self.pace_adjuster = PaceAdjuster(self.db)    # Pace multiplier
        
        # [R2] Bayesian minutes model for sparse-data players
        try:
            from ..models.bayesian_minutes import BayesianMinutesModel
            self.bayesian_minutes = BayesianMinutesModel(self.db)
        except Exception:
            self.bayesian_minutes = None
        self.injury = InjuryIngestion(self.db)
        self.odds = OddsIngestion(self.db)
        self.nba = NBAIngestion(self.db)
        
        # Cache for team injuries (populated in predict_todays_slate)
        self._team_injuries: dict = {}

        

        # Set random seed for reproducibility

        np.random.seed(self.config.random_seed)

    

    def load_models(self) -> bool:

        """Load trained models.

        

        Returns:

            True if successful

        """

        return self.trainer.load_models()

    

    def simulate_player_points(self, player_id: int,
                                team_abbr: str,
                                opponent_abbr: str,
                                game_date: str,
                                p_play: float = 1.0,
                                team_injuries: Dict[int, float] = None,
                                market_line: float = None) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a player's points.
        
        NEW: Conservative Baseline Approach + Market Discrepancy Adjustment
        Points = Baseline + Bounded Adjustments + Market Correction
        
        Args:
            player_id: NBA player ID
            team_abbr: Player's team
            opponent_abbr: Opponent team
            game_date: Game date
            p_play: Probability of playing (from injury status)
            team_injuries: Dict of teammate injuries {player_id: p_play}
            market_line: Optional - The current sportsbook line for points (external truth)
            
        Returns:
            Dict with simulation results
        """
        # Default empty injuries if not provided
        if team_injuries is None:
            team_injuries = self._team_injuries.get(team_abbr, {})
        
        # Build feature vector
        features = self.fe.build_features_for_player(
            player_id=player_id,
            team_abbr=team_abbr,
            opponent_abbr=opponent_abbr,
            game_date=game_date,
            p_play=p_play
        )
        
        # 1. Calculate Conservative Baseline
        # Weights: 35% L5, 25% L10, 25% Season, 15% Role Baseline
        
        ppg_L5 = features.get('points_L5', features.get('ppg_L5', 0))
        ppg_L10 = features.get('points_L10', features.get('ppg_L10', 0))
        ppg_season = features.get('ppg_season', features.get('points_L30', 0)) # Fallback if season missing
        
        # Role Baseline Lookup
        player_role = self.usage_model.classify_player_role(player_id, team_abbr)
        role_baselines = {
            'volume_star': 26.0,
            'star': 22.0,
            'secondary_star': 18.0,
            'third_option': 14.0,
            'starter': 12.0,
            'microwave_scorer': 12.0,
            'six_man': 12.0,
            'role_player': 8.0,
            'bench_scorer': 8.0,
            'rotation': 6.0,
            'deep_bench': 4.0
        }
        role_base = role_baselines.get(player_role, 8.0)
        
        # Handle missing data cases
        if ppg_L5 == 0: ppg_L5 = ppg_L10 if ppg_L10 > 0 else role_base
        if ppg_L10 == 0: ppg_L10 = ppg_season if ppg_season > 0 else role_base
        if ppg_season == 0: ppg_season = role_base
        
        # NEW: Heavily weight L5 to match real recent performance
        # Old: 35% L5 + 25% L10 + 25% Season + 15% Role
        # New: 50% L5 + 25% L10 + 15% Season + 10% Role (more responsive to recent form)
        baseline = (0.50 * ppg_L5) + (0.25 * ppg_L10) + (0.15 * ppg_season) + (0.10 * role_base)
        
        # [R2] Bayesian minutes adjustment for sparse-data players
        bayes_minutes_mult = 1.0
        if not self.ablation_flags.get('disable_bayes_minutes'): 
            try:
                if self.bayesian_minutes:
                    bm_result = self.bayesian_minutes.predict_minutes_bayesian(
                        player_id, team_abbr, n_samples=500
                    )
                    # Only adjust if prior is doing heavy lifting (sparse data)
                    if bm_result.get('data_weight', 1.0) < 0.7:
                        # Get L5 minutes avg for comparison
                        min_L5 = features.get('minutes_L5', features.get('avg_minutes_L5', 0))
                        if min_L5 > 0 and bm_result['mean'] > 0:
                            # Bayesian mean might differ from L5 — ratio as multiplier
                            ratio = bm_result['mean'] / min_L5
                            bayes_minutes_mult = 1.0 + max(-0.15, min(0.15, ratio - 1.0))
            except Exception:
                pass
        
        # 2. Calculate Bounded Adjustments (Multipliers)
        
        # A. Usage/Injury Adjustment
        usage_bump = 0.0
        original_usage = self.usage_model.get_baseline_usage(player_id)
        if original_usage > 0:
            predicted_usage_dist = self.usage_model.predict_distribution(
                player_id=player_id,
                team_abbr=team_abbr,
                injuries=team_injuries,
                n_samples=100 # Low sample for mean estimation
            )
            predicted_usage = np.mean(predicted_usage_dist)
            usage_bump_pct = (predicted_usage - original_usage) / original_usage
            # Cap usage bump: max ±12%
            usage_bump_pct = max(-0.12, min(0.12, usage_bump_pct))
            usage_mult = 1.0 + usage_bump_pct
        else:
            usage_mult = 1.0
            
        # B. Pace Adjustment
        pace_mult_raw = self.pace_adjuster.calculate_pace_multiplier(team_abbr, opponent_abbr)
        # Cap Pace: max ±8%
        pace_mult = 1.0 + max(-0.08, min(0.08, pace_mult_raw - 1.0))
        
        # C. Efficiency/Defense Adjustment
        eff_mult_raw, _ = self.efficiency_model.predict_efficiency_multiplier(
            player_id, 
            opponent_abbr,
            team_abbr=team_abbr,
            injuries=team_injuries
        )
        # Cap Defense: max ±10%
        eff_mult = 1.0 + max(-0.10, min(0.10, eff_mult_raw - 1.0))
        
        # D. Blowout Adjustment
        spread = features.get('spread', 0.0) or 0.0
        blowout_risk = min(0.60, abs(spread) / 40.0)
        blowout_adjustment = 1.0
        if blowout_risk > 0.15:
            # Expected reduction due to blowout risk
            # If 20% risk of 25% min reduction -> 0.2 * 0.75 + 0.8 * 1.0 = 0.95
            blowout_adjustment = 1.0 - (blowout_risk * 0.15) # conservative limit
            # Cap blowout: max -15%
            blowout_adjustment = max(0.85, blowout_adjustment)

        # E. [R4] Matchup Model — 3-level cascade (H2H → Archetype-Scheme → DvP)
        matchup_mult = 1.0
        if not self.ablation_flags.get('disable_matchup_model'):
            try:
                from ..models.matchup_model import MatchupModel
                mm = MatchupModel(db=self.db)
                matchup_result = mm.get_matchup_multiplier(player_id, opponent_abbr, market='points')
                matchup_mult = matchup_result.get('multiplier', 1.0)
                matchup_mult = 1.0 + max(-0.10, min(0.10, matchup_mult - 1.0))
            except Exception:
                pass
        
        # F. [R8] Teammate Impact Graph — scoring changes from absent teammates
        teammate_mult = 1.0
        if not self.ablation_flags.get('disable_teammate_impact'):
            try:
                from ..models.teammate_graph import TeammateImpactGraph
                tg = TeammateImpactGraph(db=self.db)
                impact = tg.get_injury_impact_multiplier(player_id, team_abbr, team_injuries)
                teammate_mult = impact.get('multiplier', 1.0)
                teammate_mult = 1.0 + max(-0.15, min(0.15, teammate_mult - 1.0))
            except Exception:
                pass

        # G. [R4] Shooting Form Reversion (Slump/Hot Streak)
        form_mult = 1.0
        if not self.ablation_flags.get('disable_shooting_form'):
            try:
                form_res = self.variance_model.get_shooting_form_reversion(player_id)
                form_mult = form_res.get('multiplier', 1.0)
            except Exception:
                pass

        # G. Market Discrepancy Adjustment (The "Insider Info" Fix)
        # [CRITICAL UPDATE] Ablation testing (2026-02-17) showed this HURTS performance (MAE 6.03 -> 5.72 when removed).
        # The market line often leads us astray or we over-react to bad lines.
        # DISABLING by default until further research.
        market_adjustment = 0.0
        
        # if not self.ablation_flags.get('disable_market_adjustment'):
        #     if market_line and market_line > 5.0:
        #          # Calculate raw difference from our conservative baseline
        #          implied_diff = market_line - baseline
        #          
        #          # Threshold: +4.0 points diff
        #          if implied_diff > 4.0:
        #              # We trust the market for 50% of the diff
        #              market_adjustment = implied_diff * 0.5
        #              market_adjustment = min(market_adjustment, 8.0)

        # 3. Combine to Final Mean
        # All multipliers are bounded to prevent extreme compounding
        combined_mult = usage_mult * pace_mult * eff_mult * blowout_adjustment * matchup_mult * teammate_mult * bayes_minutes_mult * form_mult
        
        # SAFETY: Floor the combined multiplier to prevent over-compression
        combined_mult = max(0.80, min(1.25, combined_mult))
        
        adjusted_mean = (baseline * combined_mult) + market_adjustment
        
        # NEW: Hard Ceiling Cap - Never predict more than L5 * 1.25 (unless market line explicitly higher)
        ceiling_mult = 1.25
        floor_val = 10.0
        
        l5_ceiling = ppg_L5 * ceiling_mult
        if market_line and market_line > l5_ceiling:
            # Market knows something we don't - allow up to market line + buffer
            # Allow 15% upside on top of the market line to enable OVER edges
            l5_ceiling = market_line * 1.15
        adjusted_mean = min(adjusted_mean, max(l5_ceiling, floor_val))  # Floor to not kill low-volume players
        
        # 4. Generate Distribution
        # Use variance model for std dev
        adjusted_std, player_archetype = self.variance_model.calculate_player_variance(player_id)
        
        # Ensure variance scales with mean (CV sanity)
        # Min CV 0.25 (consistent) to 0.45 (volatile)
        implied_cv = adjusted_std / adjusted_mean if adjusted_mean > 0 else 0
        target_cv = max(0.25, min(0.45, implied_cv))
        if implied_cv < 0.25:
             adjusted_std = adjusted_mean * 0.25
        elif implied_cv > 0.45:
             adjusted_std = adjusted_mean * 0.45

        # Monte Carlo Sampling
        points_samples = np.random.normal(adjusted_mean, adjusted_std, self.n_sims)
        
        # Apply Floor (0) and Hard Ceiling (Wilt Rule) non-linearly
        points_samples = np.maximum(0, points_samples)
        
        # Handle "Did Not Play" risk (if p_play < 1.0)
        if p_play < 1.0:
            play_mask = np.random.random(self.n_sims) < p_play
            points_samples = np.where(play_mask, points_samples, 0)

        # Calculate statistics
        results = {
            'player_id': player_id,
            'team': team_abbr,
            'opponent': opponent_abbr,
            'game_date': game_date,
            'p_play': p_play,
            
            # Point predictions
            'predicted_mean': np.mean(points_samples),
            'predicted_std': np.std(points_samples),
            'p10': np.percentile(points_samples, 10),
            'p25': np.percentile(points_samples, 25),
            'p50': np.percentile(points_samples, 50),
            'p75': np.percentile(points_samples, 75),
            'p90': np.percentile(points_samples, 90),
            
            # Components for Transparency
            'baseline_points': baseline,
            'predicted_minutes': features.get('minutes_L10', 0), # Pass-through for report
            'predicted_ppm': 0, # Not calculated in this method anymore
            'predicted_usage': 0, # Not calculated
            
            'components': {
               'baseline': baseline,
               'usage_mult': usage_mult,
               'pace_mult': pace_mult,
               'eff_mult': eff_mult,
               'matchup_mult': matchup_mult,
               'teammate_mult': teammate_mult,
               'form_mult': form_mult,
               'blowout_adj': blowout_adjustment,
               'market_adj': market_adjustment
            },

            'player_role': player_role,
            'player_archetype': player_archetype, # Passed to Auditor
            'adjusted_variance': adjusted_std,
            
            # Context
            'spread': spread,
            'blowout_risk': blowout_risk,
        }
        
        return results

    

    def run_feature_audit(self, auditor: Any) -> bool:

        """Run feature variability audit on today's slate (Audit 2.1).

        

        Args:

            auditor: PredictionAuditor instance

            

        Returns:

            True if audit passes, False otherwise

        """

        print("\n[Simulation] Running Pre-Flight Feature Audit...")

        today = datetime.now().strftime('%Y-%m-%d')

        

        # Get today's games

        games = self.nba.get_todays_games()

        if not games:

            print("  [WARN] No games today, skipping audit")

            return True

        

        # Ensure odds are loaded

        # [OPTIMIZATION] Skipped redundant fetch
        # self.odds.fetch_todays_odds()

        

        all_features = []

        failed_players = []  # Track failures for soft-fail

        total_players = 0

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            for game in games:

                home_id = game.get('home_team_id')

                away_id = game.get('away_team_id')

                

                # Get team abbreviations

                cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (home_id,))

                row = cursor.fetchone()

                home_abbr = row['abbreviation'] if row else 'UNK'

                

                cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (away_id,))

                row = cursor.fetchone()

                away_abbr = row['abbreviation'] if row else 'UNK'

                

                # Iterate teams

                for team_id, team_abbr, opp_abbr in [

                    (home_id, home_abbr, away_abbr),

                    (away_id, away_abbr, home_abbr)

                ]:

                    roster = self.nba.get_team_roster(team_id)

                    

                    for player in roster:

                        player_id = player.get('player_id')

                        if not player_id:

                            continue

                        

                        total_players += 1

                        

                        # Build features with error tracking

                        try:

                            features = self.fe.build_features_for_player(

                                player_id=player_id,

                                team_abbr=team_abbr,

                                opponent_abbr=opp_abbr,

                                game_date=today

                            )

                            all_features.append(features)

                        except Exception as e:

                            failed_players.append((player_id, str(e)))

        

        # === INSTRUMENTATION ===

        print(f"\n  [AUDIT] Total players attempted: {total_players}")

        print(f"  [AUDIT] Successfully built features: {len(all_features)}")

        print(f"  [AUDIT] Failed players: {len(failed_players)}")

        

        if failed_players:

            print(f"  [AUDIT] First 5 failures:")

            for pid, err in failed_players[:5]:

                print(f"    - Player {pid}: {err[:50]}...")

        

        # === SOFT-FAIL MODE ===

        # If we have at least 50% valid features, proceed with warning

        if len(all_features) < total_players * 0.5:

            print(f"  [FAIL] Too few valid features ({len(all_features)} / {total_players})")

            return False

        

        if failed_players:

            print(f"  [WARN] Proceeding despite {len(failed_players)} failed players (soft-fail mode)")

        

        df = pd.DataFrame(all_features)

        print(f"  Generated features for {len(df)} players")

        

        return auditor.check_feature_variability(df)





    def predict_todays_slate(self) -> pd.DataFrame:

        """Generate predictions for all players in today's games.

        

        Returns:

            DataFrame with predictions for each player

        """

        today = datetime.now().strftime('%Y-%m-%d')

        print(f"\n[Simulation] Generating predictions for {today}...")

        

        # Get today's games and players

        games = self.nba.get_todays_games()

        

        if not games:

            print("   No games scheduled today")

            return pd.DataFrame()

        

        print(f"  Found {len(games)} games")

        

        # Get odds

        self.odds.fetch_todays_odds()

        

        # Get injuries

        injuries = self.injury.fetch_injuries_from_web()

        injury_map = {i['player_id']: i['p_play'] for i in injuries if i.get('player_id')}

        

        # Get team rosters

        predictions = []

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            for game in games:

                home_id = game.get('home_team_id')

                away_id = game.get('away_team_id')

                

                # Get team abbreviations

                cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (home_id,))
                home_row = cursor.fetchone()
                home_abbr = home_row['abbreviation'] if home_row else 'UNK'
                
                cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (away_id,))
                away_row = cursor.fetchone()
                away_abbr = away_row['abbreviation'] if away_row else 'UNK'
                
                # Get rosters and predict for each player
                for team_id, team_abbr, opp_abbr in [
                    (home_id, home_abbr, away_abbr),
                    (away_id, away_abbr, home_abbr)
                ]:
                    roster = self.nba.get_team_roster(team_id)
                    
                    # Build team-specific injury map for usage redistribution
                    team_injuries = {}
                    for player in roster:
                        pid = player.get('player_id')
                        if pid and pid in injury_map:
                            team_injuries[pid] = injury_map[pid]
                    
                    # Store in cache for usage model
                    self._team_injuries[team_abbr] = team_injuries
                    
                    for player in roster:
                        player_id = player.get('player_id')
                        if not player_id:
                            continue
                        
                        # Get injury status
                        p_play = injury_map.get(player_id, 1.0)
                        
                        try:
                            result = self.simulate_player_points(
                                player_id=player_id,
                                team_abbr=team_abbr,
                                opponent_abbr=opp_abbr,
                                game_date=today,
                                p_play=p_play,
                                team_injuries=team_injuries  # Pass team injuries for usage model
                            )
                            result['player_name'] = player.get('full_name', 'Unknown')
                            predictions.append(result)
                        except Exception as e:
                            print(f"  [WARN] Error predicting {player.get('full_name')}: {e}")
        
        df = pd.DataFrame(predictions)
        
        if not df.empty:
            # Sort by predicted mean points
            df = df.sort_values('predicted_mean', ascending=False)

            print(f"   Generated {len(df)} player predictions")

        

        return df

    

    def generate_predictions_csv(self, output_path: str = None) -> str:

        """Generate predictions and save to CSV.

        

        Args:

            output_path: Optional output path

            

        Returns:

            Path to saved CSV

        """

        df = self.predict_todays_slate()

        

        if df.empty:

            print("   No predictions to save")

            return ""

        

        # Select columns for output
        output_cols = [
            'player_name', 'team', 'opponent', 'game_date',
            'p_play', 'spread', 
            'predicted_mean', 'p10', 'p50', 'p90',
            'predicted_minutes', 'predicted_ppm',
            # NEW: Usage tracking columns
            'predicted_usage', 'usage_bump', 'player_role',
            'blowout_risk', 'pace_edge', 'efficiency_edge'
        ]

        

        # Filter to available columns

        output_cols = [c for c in output_cols if c in df.columns]

        output_df = df[output_cols].copy()

        

        # Round numeric columns

        numeric_cols = output_df.select_dtypes(include=[np.number]).columns

        output_df[numeric_cols] = output_df[numeric_cols].round(2)

        

        # Save

        if output_path is None:

            today = datetime.now().strftime('%Y-%m-%d')

            output_path = self.config.project_root / f'predictions_{today}.csv'

        

        output_df.to_csv(output_path, index=False)

        print(f"   Predictions saved to {output_path}")

        

        return str(output_path)





# Convenience function

def get_simulation_engine() -> SimulationEngine:

    """Get simulation engine instance."""

    return SimulationEngine()

