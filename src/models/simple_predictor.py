"""
Market Predictor
================
Generic XGBoost-based predictor for multiple NBA player prop markets.
Supports: Points, Assists, Rebounds.
"""

import logging
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger("MARKET_PREDICTOR")


class MarketPredictor:
    """
    Production-grade generic predictor for multiple markets.
    
    Dynamically selects features based on target market:
    - Points: ppg_L5, minutes_L5, ppm_L5...
    - Assists: ast_L5, minutes_L5, ast_std_L5...
    - Rebounds: reb_L5, minutes_L5, reb_std_L5...
    """
    
    # Configuration for each market type
    MARKET_CONFIG = {
        'points': {
            'clamp_min': 0, 'clamp_max': 60, 
            'base_std': 6.0,
            'features': ['ppg_L5', 'minutes_L5', 'ppm_L5', 'is_home', 'is_b2b', 'opp_def_rating']
        },
        'assists': {
            'clamp_min': 0, 'clamp_max': 25, 
            'base_std': 2.5,
            # Must match training.py feature set
            'features': ['ast_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        },
        'rebounds': {
            'clamp_min': 0, 'clamp_max': 30, 
            'base_std': 3.5,
            # Must match training.py feature set
            'features': ['reb_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        },
        # [PHASE 13] New Markets
        'threes': {
            'clamp_min': 0, 'clamp_max': 15,
            'base_std': 1.5,
            'features': ['threes_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        },
        'blocks': {
            'clamp_min': 0, 'clamp_max': 10,
            'base_std': 1.0,
            'features': ['blocks_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        },
        'steals': {
            'clamp_min': 0, 'clamp_max': 10,
            'base_std': 1.0,
            'features': ['steals_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        },
        'field_goals': {
            'clamp_min': 0, 'clamp_max': 25,
            'base_std': 3.0,
            'features': ['fgm_L5', 'minutes_L5', 'is_home', 'is_b2b', 'opp_def_rating', 'rest_days']
        }
    }
    
    def __init__(self, 
                 target_market: str = 'points',
                 model_dir: str = None):
        """
        Initialize predictor for a specific market.
        
        Args:
            target_market: 'points', 'assists', or 'rebounds'
            model_dir: Directory containing .pkl models (auto-detected if None)
        """
        self.market = target_market.lower()
        if self.market == 'fgm': self.market = 'field_goals' # Alias
        if self.market not in self.MARKET_CONFIG:
            logger.warning(f"Unknown market '{self.market}', defaulting to points config")
            self.market = 'points'
            
        self.config = self.MARKET_CONFIG[self.market]
        
        # Determine paths - check 'models/' first, then 'src/models/' as fallback
        model_path = None
        features_path = None
        
        # Priority 1: Look for model_{market}.pkl in 'models/' (new multi-market models)
        primary_dir = 'models'
        primary_path = f"{primary_dir}/model_{self.market}.pkl"
        primary_features = f"{primary_dir}/model_{self.market}_features.json"
        
        if Path(primary_path).exists():
            model_path = primary_path
            features_path = primary_features
        else:
            # Priority 2: Look in 'src/models/' (legacy location)
            fallback_dir = model_dir or 'src/models'
            fallback_path = f"{fallback_dir}/model_{self.market}.pkl"
            fallback_features = f"{fallback_dir}/model_{self.market}_features.json"
            
            if Path(fallback_path).exists():
                model_path = fallback_path
                features_path = fallback_features
            elif self.market == 'points':
                # Priority 3: Legacy simple_model for points
                legacy_path = f"{fallback_dir}/simple_model.pkl"
                legacy_features = f"{fallback_dir}/simple_model_features.json"
                if Path(legacy_path).exists():
                    model_path = legacy_path
                    features_path = legacy_features

        self.model = None
        self.feature_names = []
        if model_path:
            self._load_model(model_path, features_path)
        
    def _load_model(self, model_path: str, features_path: str):
        """Load trained model and feature names."""
        try:
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded {self.market} model from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}. Using fallback logic.")
                
            if Path(features_path).exists():
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Features: {self.feature_names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def predict(self, 
                player_context: Dict[str, Any],
                match_context: Dict[str, Any],
                market_line: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate prediction for a player.
        
        Args:
            player_context: Dict with player stats (from DataGatherer)
            match_context: Dict with game context
            market_line: Optional sportsbook line to use as anchor
            
        Returns:
            Dict with prediction, confidence interval, etc.
        """
        # Extract features
        features = self._extract_features(player_context, match_context)
        degradation_flags: List[str] = []
        health_notes: List[str] = []
        model_loaded = self.model is not None
        market_anchor_applied = False
        market_anchor_weight = 0.0
        
        # 1. Prediction (Model or Fallback)
        if self.model:
            try:
                # Align features to model expectations
                X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
                raw_pred = float(self.model.predict(X)[0])
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raw_pred = self._get_baseline_avg(player_context)
                degradation_flags.append('model_inference_failed')
                health_notes.append(str(e))
        else:
            raw_pred = self._get_baseline_avg(player_context)
            degradation_flags.append('model_fallback_baseline')
            
        # 2. Clamping
        raw_pred = max(self.config['clamp_min'], min(self.config['clamp_max'], raw_pred))
        pure_model_pred = raw_pred
        adjusted_pred = pure_model_pred

        rest_days = player_context.get('rest_days')
        if rest_days is None:
            degradation_flags.append('missing_rest_days')
        injury_context_present = (
            'team_injuries' in player_context or
            'team_injuries' in match_context
        )
        injury_context = player_context.get('team_injuries', match_context.get('team_injuries', {}))
        if not injury_context_present:
            degradation_flags.append('missing_injury_context')
        if not match_context.get('h2h_history'):
            degradation_flags.append('missing_h2h_history')
        
        # 3. Market Line Blending
        # Keep the base predictor mostly independent and let the calibrator
        # handle the main market anchoring step. Only degraded predictions get
        # a light pre-anchor here.
        if market_line is not None and market_line > 0:
            is_star = features.get('is_star', 0)
            regime_flags = match_context.get('regime_flags', [])
            degraded_prediction = (
                not model_loaded or
                'model_inference_failed' in degradation_flags or
                len(degradation_flags) >= 3
            )

            if degraded_prediction:
                market_anchor_weight = 0.15 if is_star else 0.08
                degradation_flags.append('light_market_anchor_due_to_degraded_prediction')
                if 'SYSTEM_REDUCE_MARKET_ANCHOR' in regime_flags:
                    market_anchor_weight *= 0.5
                    degradation_flags.append('market_anchor_reduced_by_policy')
                if 'SYSTEM_STRICT_INJURY_CONTEXT' in regime_flags and not injury_context_present:
                    market_anchor_weight = 0.0
                    degradation_flags.append('market_anchor_disabled_no_injury_context')
                adjusted_pred = (1 - market_anchor_weight) * pure_model_pred + market_anchor_weight * market_line
                market_anchor_applied = market_anchor_weight > 0
        final_pred = adjusted_pred

        player_role = str(player_context.get('player_role', 'rotation'))
        recent_logs = player_context.get('recent_logs', []) or []
        if self.market == 'points':
            ppg_l5 = float(player_context.get('ppg_L5') or player_context.get('points_L5') or 0.0)
            ppg_l15 = float(player_context.get('points_L15') or ppg_l5 or 0.0)
            points_max_l5 = float(player_context.get('points_max_L5') or ppg_l5 or 0.0)
            usage_proxy_l5 = float(player_context.get('usage_proxy_L5') or 0.0)
            usage_proxy_delta = float(player_context.get('usage_proxy_delta') or 0.0)
            recent_points = [float(g.get('points') or 0.0) for g in recent_logs[:5] if g.get('points') is not None]
            hot_games = sum(1 for pts in recent_points if pts >= max(25.0, market_line or 0.0))
            recent_peak = max(recent_points) if recent_points else points_max_l5
            star_profile = player_role == 'star' or ppg_l5 >= 22.0 or usage_proxy_l5 >= 0.85

            recent_form_adj = 0.0
            # Weight recent form more than medium-term form when a scorer is trending up.
            recent_form_adj += max(-2.0, min(2.5, (ppg_l5 - ppg_l15) * 0.30))
            recent_form_adj += max(-1.0, min(1.5, usage_proxy_delta * 10.0))

            if player_context.get('points_trend') == 'up':
                recent_form_adj += 0.8 if star_profile else 0.4
            elif player_context.get('points_trend') == 'down':
                recent_form_adj -= 0.6 if star_profile else 0.3

            if star_profile:
                ceiling_gap = max(0.0, recent_peak - max(final_pred, ppg_l5))
                recent_form_adj += min(2.2, ceiling_gap * 0.12)
                if hot_games >= 2:
                    recent_form_adj += 0.8
            elif hot_games >= 2:
                recent_form_adj += 0.4

            if abs(recent_form_adj) >= 0.15:
                final_pred += recent_form_adj
                degradation_flags.append('recent_form_adjusted_prediction')
                health_notes.append(f"recent_form_adj={recent_form_adj:+.2f}")
          
        # NEW: L5 Ceiling Cap - More generous for AST/REB (L5 * 1.40)
        l5_baseline = self._get_baseline_avg(player_context)
        ceiling_mult = 1.40  # Allow up to 40% above L5 avg
        l5_ceiling = l5_baseline * ceiling_mult
        
        # Always trust market if available and higher than our ceiling
        if market_line and market_line > l5_ceiling:
            l5_ceiling = market_line
        
        # Market-specific floor values to not kill low-volume players
        market_floors = {
            'assists': 3.0,
            'rebounds': 4.0,
            'threes': 0.5,
            'blocks': 0.5,
            'steals': 0.5,
            'field_goals': 2.0,
            'points': 5.0,
        }
        floor_value = market_floors.get(self.market, 3.0)
        if self.market == 'points':
            if player_role == 'star':
                ceiling_mult = 1.65
            elif player_role == 'starter':
                ceiling_mult = 1.50
            l5_ceiling = max(l5_ceiling, float(player_context.get('points_max_L5') or l5_baseline))
        final_pred = min(final_pred, max(l5_ceiling, floor_value))
        post_rules_pred = final_pred
        
        # [FIX] Apply Teammate Impact (Post-Model Adjustment)
        # Verify connectivity: Orchestrator passes 'usage_impact' or 'teammate_impact' in match_context
        teammate_impact = match_context.get('usage_impact') or match_context.get('teammate_impact')
        if teammate_impact and self.market == 'points':
            boost = teammate_impact.get('expected_points_boost', 0)
            if boost and abs(boost) > 0.1:
                logger.info(f"Applying teammate injury boost: +{boost:.1f} pts")
                final_pred += boost
                
        # Also apply boosts for other markets if available in impact
        if teammate_impact:
            if self.market == 'assists':
                boost = teammate_impact.get('total_assists_boost', 0)
                if boost: final_pred += boost
            elif self.market == 'rebounds':
                boost = teammate_impact.get('total_rebounds_boost', 0)
                if boost: final_pred += boost
            elif self.market == 'minutes': # If we predicted minutes
                boost = teammate_impact.get('total_minutes_boost', 0)
                if boost: final_pred += boost

        lineup_context = match_context.get('lineup_context') or player_context.get('lineup_context') or {}
        if lineup_context:
            role = str(lineup_context.get('player_role', player_context.get('player_role', 'rotation')))
            usage_delta = float(lineup_context.get('usage_delta', 0.0) or 0.0)
            minutes_delta = float(lineup_context.get('minutes_delta', 0.0) or 0.0)
            assists_delta = float(lineup_context.get('assists_delta', 0.0) or 0.0)
            rebounds_delta = float(lineup_context.get('rebounds_delta', 0.0) or 0.0)
            volatility_score = float(lineup_context.get('volatility_score', 0.0) or 0.0)
            role_change = bool(lineup_context.get('role_change', False))
            usage_spike = bool(lineup_context.get('usage_spike', False))
            significant_absence_cluster = bool(lineup_context.get('significant_absence_cluster', False))
            lineup_adj = 0.0

            if self.market == 'points':
                lineup_adj += usage_delta * 18.0
                lineup_adj += max(-1.5, min(2.5, minutes_delta * 0.22))
                if usage_spike:
                    lineup_adj += 0.8
                if role_change and role in ('starter', 'star'):
                    lineup_adj += 1.0
            elif self.market == 'assists':
                lineup_adj += assists_delta
                lineup_adj += max(-0.6, min(1.0, usage_delta * 5.0))
                if role_change and role in ('starter', 'star'):
                    lineup_adj += 0.3
            elif self.market == 'rebounds':
                lineup_adj += rebounds_delta
                lineup_adj += max(-0.5, min(0.8, minutes_delta * 0.10))
            elif self.market == 'threes':
                lineup_adj += max(-0.4, min(0.8, usage_delta * 4.5))
                if significant_absence_cluster and role in ('starter', 'star'):
                    lineup_adj += 0.2
            elif self.market in ('blocks', 'steals'):
                lineup_adj += max(-0.2, min(0.3, minutes_delta * 0.04))
            elif self.market == 'field_goals':
                lineup_adj += max(-0.8, min(1.5, usage_delta * 7.0))
                lineup_adj += max(-0.5, min(1.0, minutes_delta * 0.10))

            if abs(lineup_adj) > 0.1:
                final_pred += lineup_adj
                degradation_flags.append('lineup_context_adjusted_prediction')
                health_notes.append(f"lineup_adj={lineup_adj:+.2f}")
            
        # 4. Uncertainty Estimation
        uncertainty = self._estimate_uncertainty(player_context)
        regime_flags = match_context.get('regime_flags', [])
        if 'SYSTEM_WIDEN_MINUTES_UNCERTAINTY' in regime_flags:
            uncertainty *= 1.10
            degradation_flags.append('minutes_uncertainty_widened_by_policy')
        if lineup_context and float(lineup_context.get('volatility_score', 0.0) or 0.0) >= 0.35:
            uncertainty *= 1.12
            degradation_flags.append('lineup_volatility_widened_uncertainty')
        if len(degradation_flags) >= 2:
            uncertainty *= 1.10
        if 'model_fallback_baseline' in degradation_flags or 'model_inference_failed' in degradation_flags:
            uncertainty *= 1.15
        
        # 5. Build Distribution
        samples = np.random.normal(final_pred, uncertainty, 1000)
        samples = np.clip(samples, self.config['clamp_min'], self.config['clamp_max'])
        health_score = max(0.0, 1.0 - 0.15 * len(set(degradation_flags)))
        
        return {
            'mean': final_pred,
            'std': uncertainty,
            'p10': float(np.percentile(samples, 10)),
            'p25': float(np.percentile(samples, 25)),
            'p50': float(np.percentile(samples, 50)),
            'p75': float(np.percentile(samples, 75)),
            'p90': float(np.percentile(samples, 90)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'samples': samples, # Keep for orchestrator simulations
            'raw_model_pred': pure_model_pred if model_loaded else None,
            'pure_model_pred': pure_model_pred,
            'market_adjusted_pred': adjusted_pred,
            'post_rule_pred': post_rules_pred,
            'market_line': market_line,
            'features_used': features,
            'market_type': self.market,
            'h2h_avg': self._get_h2h_avg(match_context.get('h2h_history', [])),
            'prediction_health': {
                'model_loaded': model_loaded,
                'used_fallback_model': not model_loaded or 'model_inference_failed' in degradation_flags,
                'market_anchor_applied': market_anchor_applied,
                'market_anchor_weight': market_anchor_weight,
                'degradation_flags': sorted(set(degradation_flags)),
                'health_score': round(health_score, 2),
                'notes': health_notes,
            }
        }
    
    # Market-to-DB-column mapping for H2H, hit rate, etc.
    MARKET_DB_MAP = {
        'points': 'points',
        'assists': 'assists',
        'rebounds': 'rebounds',
        'threes': 'fg3m',
        'blocks': 'blk',
        'steals': 'stl',
        'field_goals': 'fgm',
    }
    
    def _get_h2h_avg(self, history: List[Dict]) -> float:
        """Calculate average for this market against specific opponent."""
        if not history:
            return None
        
        db_col = self.MARKET_DB_MAP.get(self.market, self.market)
        values = []
        for game in history:
            if not isinstance(game, dict):
                continue
            val = game.get(db_col, game.get(self.market, 0))
            values.append(val)
            
        return sum(values) / len(values) if values else None
    
    def _extract_features(self, 
                          player_context: Dict[str, Any],
                          match_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract generic features + market-specific stats."""
        
        # --- Base Context ---
        is_home = 1 if match_context.get('is_home', False) else 0
        is_b2b = 1 if match_context.get('is_b2b', False) else 0
        opp_def_rating = match_context.get('opp_def_rating', 110.0) or 110.0
        
        # Star heuristic (Points-based usually works best for "Is this a main guy?")
        # Support both naming conventions: ppg_L5 (legacy) and points_L5 (from DataGatherer)
        ppg_L5 = player_context.get('ppg_L5', player_context.get('points_L5', player_context.get('ppg', 15.0)))
        is_star = 1 if (ppg_L5 or 0) > 20 else 0
        minutes_L5 = player_context.get('minutes_L5', 25.0)
        minutes_season = player_context.get('minutes_season', minutes_L5) # [NEW] Long-term role context

        # --- Points Features ---
        ppm_L5 = player_context.get('ppm_L5', 0.5)
        
        # --- Assists Features ---
        ast_L5 = player_context.get('ast_L5', player_context.get('assists_L5', 2.0))
        ast_std = player_context.get('ast_std_L5', 1.0)
        
        # --- Rebounds Features ---
        reb_L5 = player_context.get('reb_L5', player_context.get('rebounds_L5', 4.0))
        reb_std = player_context.get('reb_std_L5', 2.0)
        
        # --- New Market Features ---
        threes_L5 = player_context.get('threes_L5', player_context.get('fg3m_L5', 1.0))
        blocks_L5 = player_context.get('blocks_L5', player_context.get('blk_L5', 0.5))
        steals_L5 = player_context.get('steals_L5', player_context.get('stl_L5', 0.5))
        fgm_L5 = player_context.get('fgm_L5', player_context.get('field_goals_L5', 5.0))
        rest_days = player_context.get('rest_days', 1)
        
        # --- Advanced Context (from feature engineering) ---
        opp_pace = match_context.get('opp_pace', 100.0)
        lineup_context = match_context.get('lineup_context') or player_context.get('lineup_context') or {}
        
        return {
            # Shared
            'minutes_L5': float(minutes_L5 or 25.0),
            'minutes_season': float(minutes_season or 25.0),
            'is_home': float(is_home),
            'is_b2b': float(is_b2b),
            'opp_def_rating': float(opp_def_rating),
            'is_star': float(is_star),
            'opp_pace': float(opp_pace or 100.0),
            'rest_days': float(rest_days or 1),
            'lineup_usage_delta': float(lineup_context.get('usage_delta', 0.0) or 0.0),
            'lineup_minutes_delta': float(lineup_context.get('minutes_delta', 0.0) or 0.0),
            'lineup_volatility_score': float(lineup_context.get('volatility_score', 0.0) or 0.0),
            'lineup_missing_rotation_count': float(lineup_context.get('missing_rotation_count', 0) or 0),
            'lineup_role_change': float(1 if lineup_context.get('role_change') else 0),
            'lineup_usage_spike': float(1 if lineup_context.get('usage_spike') else 0),
            
            # Points Specific
            'ppg_L5': float(ppg_L5 or 15.0),
            'ppm_L5': float(ppm_L5 or 0.5),
            
            # Assists Specific
            'ast_L5': float(ast_L5 or 2.0),
            'ast_std_L5': float(ast_std),
            
            # Rebounds Specific
            'reb_L5': float(reb_L5 or 4.0),
            'reb_std_L5': float(reb_std),
            
            # Threes
            'threes_L5': float(threes_L5 or 1.0),
            
            # Blocks
            'blocks_L5': float(blocks_L5 or 0.5),
            
            # Steals
            'steals_L5': float(steals_L5 or 0.5),
            
            # Field Goals Made
            'fgm_L5': float(fgm_L5 or 5.0),
        }
        
    def _get_baseline_avg(self, player_context: Dict[str, Any]) -> float:
        """Get naive recent average for the target market."""
        if self.market == 'assists':
            return player_context.get('ast_L5') or player_context.get('assists_L5') or 2.0
        elif self.market == 'rebounds':
            return player_context.get('reb_L5') or player_context.get('rebounds_L5') or 4.0
        elif self.market == 'threes':
            return player_context.get('threes_L5') or player_context.get('fg3m_L5') or 1.0
        elif self.market == 'blocks':
            return player_context.get('blocks_L5') or player_context.get('blk_L5') or 0.5
        elif self.market == 'steals':
            return player_context.get('steals_L5') or player_context.get('stl_L5') or 0.5
        elif self.market == 'field_goals':
            return player_context.get('fgm_L5') or player_context.get('field_goals_L5') or 5.0
        else: # points
            return player_context.get('ppg_L5') or player_context.get('points_L5') or 15.0

    def _estimate_uncertainty(self, player_context: Dict[str, Any]) -> float:
        """Estimate standard deviation for the distribution."""
        if self.market == 'assists':
            base = player_context.get('ast_std_L5') or 1.5
            return max(1.0, min(5.0, base))
        elif self.market == 'rebounds':
            base = player_context.get('reb_std_L5') or 2.5
            return max(1.5, min(8.0, base))
        elif self.market == 'threes':
            base = player_context.get('threes_std_L5') or player_context.get('fg3m_std_L5') or 1.0
            return max(0.5, min(3.0, base))
        elif self.market == 'blocks':
            base = player_context.get('blocks_std_L5') or player_context.get('blk_std_L5') or 0.8
            return max(0.3, min(2.5, base))
        elif self.market == 'steals':
            base = player_context.get('steals_std_L5') or player_context.get('stl_std_L5') or 0.6
            return max(0.3, min(2.5, base))
        elif self.market == 'field_goals':
            base = player_context.get('fgm_std_L5') or player_context.get('field_goals_std_L5') or 2.0
            return max(1.0, min(5.0, base))
        else: # points
            base = player_context.get('ppg_std') or player_context.get('points_std') or 6.0
            if str(player_context.get('player_role', 'rotation')) == 'star':
                base *= 1.08
            if player_context.get('points_trend') == 'up':
                base *= 1.05
            if float(player_context.get('usage_proxy_delta') or 0.0) >= 0.08:
                base *= 1.05
            return max(4.0, min(11.5, base))
