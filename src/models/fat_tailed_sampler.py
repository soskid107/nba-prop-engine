"""
Fat-Tailed Distribution Engine (R1)
=====================================
Replaces Gaussian sampling with data-driven distributions.

NBA scoring data has heavy tails and asymmetric skew:
- A player averaging 22 PPG might hit 45 once per 20 games
- But almost never scores below 8
- Gaussian assumptions systematically misprice P10/P90

This module fits the best distribution per player from:
1. Student-t (heavy tails, few games)
2. Skew-Normal (asymmetric distributions)
3. Kernel Density Estimation (non-parametric, 20+ games)

Integration: Replaces `np.random.normal` in VarianceModel and MonteCarloEngine.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from scipy import stats as sp_stats

logger = logging.getLogger("FAT_TAIL")


# Distribution fit results cache
_player_dist_cache: Dict[int, Tuple[str, Any]] = {}


class FatTailedSampler:
    """
    Replace Gaussian with data-driven distributions for Monte Carlo sampling.
    
    Key insight: NBA props have asymmetric risk. Unders hit differently than overs.
    A player who averages 24 PPG has a scoring distribution that is:
    - Right-skewed for volume scorers (can explode to 40+)
    - Left-truncated for all players (can't score negative)
    - Heavy-tailed compared to Gaussian (extreme games happen more than bell curve predicts)
    """
    
    def __init__(self, db=None):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self._cache: Dict[int, Dict[str, Any]] = {}
    
    def get_player_scoring_history(self, player_id: int, market: str = 'points',
                                    window: int = 30) -> np.ndarray:
        """Pull recent game actuals for distribution fitting."""
        col_map = {
            'points': 'points',
            'assists': 'assists', 
            'rebounds': 'rebounds',
        }
        col = col_map.get(market, 'points')
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {col} FROM player_logs 
                WHERE player_id = ? AND {col} IS NOT NULL AND minutes > 5
                ORDER BY game_date DESC 
                LIMIT ?
            """, (player_id, window))
            rows = cursor.fetchall()
        
        if not rows:
            return np.array([])
        
        return np.array([r[col] for r in rows], dtype=float)
    
    def fit_player_distribution(self, player_id: int, market: str = 'points',
                                 window: int = 30) -> Dict[str, Any]:
        """
        Fit the best distribution to a player's actual scoring history.
        
        Selection logic:
        - < 8 games: Student-t with df=4 (heavy tails, conservative)
        - 8-19 games: Student-t (fit to data)
        - 20+ games: Compare Student-t vs Skew-Normal vs KDE, pick best by NLL
        
        Returns:
            Dict with 'dist_type', 'params', 'mean', 'std', 'skewness', 'kurtosis'
        """
        cache_key = f"{player_id}_{market}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        games = self.get_player_scoring_history(player_id, market, window)
        
        if len(games) < 3:
            # Absolute minimum — use default conservative Student-t
            result = {
                'dist_type': 'student_t_default',
                'params': {'df': 4, 'loc': 15.0, 'scale': 5.0},
                'mean': 15.0,
                'std': 5.0,
                'skewness': 0.0,
                'kurtosis': 3.0,  # Excess kurtosis of t(df=4) ≈ ∞, but capped
                'sample_size': len(games),
            }
            self._cache[cache_key] = result
            return result
        
        data_mean = np.mean(games)
        data_std = np.std(games, ddof=1) if len(games) > 1 else 5.0
        data_skew = float(sp_stats.skew(games)) if len(games) >= 8 else 0.0
        data_kurt = float(sp_stats.kurtosis(games)) if len(games) >= 8 else 0.0
        
        if len(games) < 8:
            # Few games — Student-t with heavy tails (df=4)
            # Use empirical mean/std but conservative tail behavior
            result = {
                'dist_type': 'student_t_prior',
                'params': {'df': 4, 'loc': data_mean, 'scale': data_std * 0.8},
                'mean': data_mean,
                'std': data_std,
                'skewness': data_skew,
                'kurtosis': data_kurt,
                'sample_size': len(games),
            }
            self._cache[cache_key] = result
            return result
        
        # 8+ games: Fit competing distributions
        candidates = {}
        
        # 1. Student-t (captures heavy tails)
        try:
            t_params = sp_stats.t.fit(games)
            t_nll = sp_stats.t.nnlf(t_params, games)
            candidates['student_t'] = {'nll': t_nll, 'params': t_params}
        except Exception:
            pass
        
        # 2. Skew-Normal (captures asymmetry)
        try:
            sn_params = sp_stats.skewnorm.fit(games)
            sn_nll = sp_stats.skewnorm.nnlf(sn_params, games)
            candidates['skew_normal'] = {'nll': sn_nll, 'params': sn_params}
        except Exception:
            pass
        
        # 3. KDE (non-parametric — needs 20+ games for stability)
        if len(games) >= 20:
            try:
                kde = sp_stats.gaussian_kde(games, bw_method='silverman')
                # KDE doesn't have NLL in same form; use cross-validation score
                # Approximate: mean log-likelihood on data
                log_lik = np.mean(np.log(kde(games) + 1e-10))
                kde_nll = -log_lik * len(games)
                candidates['kde'] = {'nll': kde_nll, 'params': kde}
            except Exception:
                pass
        
        if not candidates:
            # All fits failed — fallback to empirical normal
            result = {
                'dist_type': 'normal_fallback',
                'params': {'loc': data_mean, 'scale': data_std},
                'mean': data_mean,
                'std': data_std,
                'skewness': data_skew,
                'kurtosis': data_kurt,
                'sample_size': len(games),
            }
            self._cache[cache_key] = result
            return result
        
        # Select best by negative log-likelihood (lower = better fit)
        best_name = min(candidates, key=lambda k: candidates[k]['nll'])
        best = candidates[best_name]
        
        # KDE only wins if it's significantly better than parametric
        if best_name != 'kde' and 'kde' in candidates:
            kde_nll = candidates['kde']['nll']
            best_nll = best['nll']
            # KDE needs to be >10% better to justify its complexity
            if kde_nll < best_nll * 0.90:
                best_name = 'kde'
                best = candidates['kde']
        
        result = {
            'dist_type': best_name,
            'params': best['params'],
            'mean': data_mean,
            'std': data_std,
            'skewness': data_skew,
            'kurtosis': data_kurt,
            'sample_size': len(games),
        }
        
        logger.debug(
            f"Player {player_id} ({market}): Best fit = {best_name} "
            f"(mean={data_mean:.1f}, std={data_std:.1f}, skew={data_skew:.2f}, "
            f"kurt={data_kurt:.2f}, n={len(games)})"
        )
        
        self._cache[cache_key] = result
        return result
    
    def sample(self, player_id: int, predicted_mean: float,
               market: str = 'points', n_samples: int = 5000,
               base_std: float = None) -> np.ndarray:
        """
        Generate samples from fitted distribution, shifted to predicted mean.
        
        The distribution SHAPE comes from historical data (capturing true tails),
        but is SHIFTED so the center matches the model's predicted mean.
        
        Args:
            player_id: NBA player ID
            predicted_mean: Model's point estimate for this game
            market: Market type ('points', 'assists', 'rebounds')
            n_samples: Number of Monte Carlo samples
            base_std: Override std (if model has its own uncertainty estimate)
            
        Returns:
            Array of n_samples positive values
        """
        fit = self.fit_player_distribution(player_id, market)
        dist_type = fit['dist_type']
        params = fit['params']
        
        # Generate raw samples from fitted distribution
        try:
            if dist_type in ('student_t', 'student_t_prior', 'student_t_default'):
                if isinstance(params, dict):
                    raw = sp_stats.t.rvs(
                        df=params['df'], loc=params['loc'], 
                        scale=params['scale'], size=n_samples
                    )
                else:
                    # Tuple from scipy.fit
                    raw = sp_stats.t.rvs(*params, size=n_samples)
                    
            elif dist_type == 'skew_normal':
                raw = sp_stats.skewnorm.rvs(*params, size=n_samples)
                
            elif dist_type == 'kde':
                raw = params.resample(n_samples).flatten()
                
            elif dist_type == 'normal_fallback':
                raw = np.random.normal(params['loc'], params['scale'], n_samples)
                
            else:
                # Ultimate fallback
                raw = np.random.normal(predicted_mean, base_std or 5.0, n_samples)
                return np.maximum(0, raw)
                
        except Exception as e:
            logger.warning(f"Sampling failed for player {player_id}: {e}. Using normal fallback.")
            raw = np.random.normal(predicted_mean, base_std or fit['std'], n_samples)
            return np.maximum(0, raw)
        
        # Shift distribution center to match predicted mean
        # This preserves the SHAPE (tails, skew) while centering on the prediction
        raw_mean = np.mean(raw)
        samples = raw - raw_mean + predicted_mean
        
        # If we have an override std, scale the spread
        if base_std is not None and fit['std'] > 0:
            scale_factor = base_std / fit['std']
            samples = predicted_mean + (samples - predicted_mean) * scale_factor
        
        # Floor at 0 (can't score negative)
        samples = np.maximum(0, samples)
        
        return samples
    
    def get_distribution_info(self, player_id: int, market: str = 'points') -> Dict[str, Any]:
        """
        Get human-readable distribution info for reports/debugging.
        
        Returns:
            Dict with distribution type, shape descriptors, and tail characteristics
        """
        fit = self.fit_player_distribution(player_id, market)
        
        # Classify tail behavior
        if fit['kurtosis'] > 3.0:
            tail_desc = "heavy-tailed (extreme games more likely than normal)"
        elif fit['kurtosis'] < -0.5:
            tail_desc = "light-tailed (consistent, few extreme games)"
        else:
            tail_desc = "normal-tailed"
        
        # Classify skewness
        if fit['skewness'] > 0.5:
            skew_desc = "right-skewed (can explode upward)"
        elif fit['skewness'] < -0.5:
            skew_desc = "left-skewed (floor risk higher)"
        else:
            skew_desc = "symmetric"
        
        return {
            'distribution': fit['dist_type'],
            'sample_size': fit['sample_size'],
            'empirical_mean': fit['mean'],
            'empirical_std': fit['std'],
            'tail_behavior': tail_desc,
            'skew_behavior': skew_desc,
            'kurtosis': fit['kurtosis'],
            'skewness': fit['skewness'],
        }
    
    def clear_cache(self):
        """Clear all cached fits (call after new data ingestion)."""
        self._cache.clear()
        _player_dist_cache.clear()


# Convenience function
def get_fat_tailed_sampler(db=None):
    """Get a FatTailedSampler instance."""
    return FatTailedSampler(db=db)
