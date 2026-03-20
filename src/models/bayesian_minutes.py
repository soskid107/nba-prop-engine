"""
Hierarchical Bayesian Minutes Model (R2)
==========================================
Uses PyMC for partial pooling of minutes predictions.

The core problem: predicting minutes for sparse-data players (rookies,
trade acquisitions, injury returnees). Classical ML treats each player
independently, but Bayesian partial pooling shares information:

  League → Team → Role → Player

A rookie with 5 games borrows strength from:
1. Other players in the same role (role mean)
2. Other players on the same team (team mean)
3. The league-wide average (league mean)

As the player accumulates games, their individual estimate
dominates and the hierarchy matters less.

Key output: Posterior predictive distribution with credible intervals
instead of point estimates, naturally encoding uncertainty from
sample size.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("BAYES_MIN")


class BayesianMinutesModel:
    """
    Hierarchical Bayesian model for player minutes prediction.
    
    Hierarchy:
      league_mean ~ Normal(28, 5)  
      team_offset[team] ~ Normal(0, σ_team)
      role_offset[role] ~ Normal(0, σ_role) 
      player_mean[player] ~ Normal(league_mean + team_offset + role_offset, σ_player)
      observed_minutes ~ Normal(player_mean, σ_obs)
    
    Benefits over ML:
    - Sparse data: Rookie with 3 games → shrinks toward role/team mean
    - Uncertainty: Get credible intervals, not just point estimates
    - Interpretable: Can see how much the prior vs data drives the estimate
    """
    
    # Role priors (typical minutes ranges)
    ROLE_PRIORS = {
        'star':           {'mean': 34.0, 'std': 3.0},
        'secondary_star': {'mean': 32.0, 'std': 3.5},
        'starter':        {'mean': 28.0, 'std': 4.0},
        'third_option':   {'mean': 26.0, 'std': 4.0},
        'bench_scorer':   {'mean': 22.0, 'std': 5.0},
        'role_player':    {'mean': 18.0, 'std': 5.0},
    }
    
    def __init__(self, db=None):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self._posterior_cache: Dict[int, Dict[str, Any]] = {}
        self._pymc_available = self._check_pymc()
    
    def _check_pymc(self) -> bool:
        """Check if PyMC is available."""
        try:
            # Suppress noisy warnings on import
            import warnings
            import logging
            
            # Helper to suppress specific g++ warning from pytensor
            def warn(*args, **kwargs):
                pass
            
            # Suppress arviz logging
            logging.getLogger("arviz").setLevel(logging.ERROR)
            logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="pytensor")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*g\\+\\+ not available.*")
                
                import pymc as pm
            return True
        except ImportError:
            logger.warning("PyMC not available. Using analytical approximation.")
            return False
        except Exception as e:
            logger.warning(f"PyMC unavailable due to environment/setup issue: {e}. Using analytical approximation.")
            return False
    
    def get_player_minutes_history(self, player_id: int, 
                                    window: int = 40) -> np.ndarray:
        """Get recent minutes data for a player."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT minutes FROM player_logs
                WHERE player_id = ? AND minutes > 5
                ORDER BY game_date DESC LIMIT ?
            """, (player_id, window))
            rows = cursor.fetchall()
        
        if not rows:
            return np.array([])
        return np.array([r['minutes'] for r in rows], dtype=float)
    
    def _get_player_role(self, player_id: int, team_abbr: str = None) -> str:
        """Get player role classification."""
        try:
            from .usage_model import UsageModel
            um = UsageModel(db=self.db)
            if team_abbr:
                return um.classify_player_role(player_id, team_abbr)
        except Exception:
            pass
        return 'starter'  # Default
    
    def _get_team_minutes_context(self, team_abbr: str) -> Dict[str, float]:
        """Get team-level minutes distribution context."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT AVG(minutes) as avg_min, 
                           COUNT(DISTINCT player_id) as n_players
                    FROM player_logs
                    WHERE team_abbreviation = ? AND minutes > 5
                    AND game_date >= date('now', '-30 days')
                """, (team_abbr,))
                row = cursor.fetchone()
                
            if row and row['avg_min']:
                return {
                    'team_avg_minutes': row['avg_min'],
                    'n_players': row['n_players'] or 10,
                }
        except Exception:
            pass
        return {'team_avg_minutes': 24.0, 'n_players': 10}
    
    def predict_minutes_bayesian(self, player_id: int,
                                  team_abbr: str = None,
                                  n_samples: int = 2000) -> Dict[str, Any]:
        """
        Predict minutes distribution using Bayesian partial pooling.
        
        For players with 20+ games: individual estimate dominates
        For players with <10 games: role/team prior dominates
        
        Returns:
            Dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'samples',
                  'prior_weight', 'data_weight', 'method'
        """
        cache_key = player_id
        if cache_key in self._posterior_cache:
            return self._posterior_cache[cache_key]
        
        minutes_data = self.get_player_minutes_history(player_id)
        role = self._get_player_role(player_id, team_abbr)
        role_prior = self.ROLE_PRIORS.get(role, self.ROLE_PRIORS['starter'])
        
        team_context = self._get_team_minutes_context(team_abbr) if team_abbr else None
        
        # Choose method based on data availability and PyMC
        # PyMC MCMC is only valuable for sparse-data players (5-15 games)
        # where the prior meaningfully affects the posterior.
        # For 15+ games, analytical conjugate is fast and equally accurate.
        n_games = len(minutes_data)
        if self._pymc_available and 5 <= n_games < 15:
            result = self._fit_pymc(minutes_data, role_prior, team_context, n_samples)
        else:
            result = self._fit_analytical(minutes_data, role_prior, team_context, n_samples)
        
        self._posterior_cache[cache_key] = result
        return result
    
    def _fit_pymc(self, data: np.ndarray, role_prior: Dict,
                   team_context: Optional[Dict],
                   n_samples: int) -> Dict[str, Any]:
        """
        Full PyMC hierarchical model.
        
        Model:
          mu_prior ~ Normal(role_mean, role_std)
          sigma ~ HalfNormal(5)
          mu ~ Normal(mu_prior, sigma_between)
          y ~ Normal(mu, sigma)
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
            
            prior_mean = role_prior['mean']
            prior_std = role_prior['std']
            
            # Adjust prior if team context available
            if team_context:
                team_avg = team_context['team_avg_minutes']
                # Blend role prior with team average (30% team, 70% role)
                prior_mean = 0.7 * prior_mean + 0.3 * team_avg
            
            with pm.Model() as model:
                # Hyperpriors
                mu = pm.Normal('mu', mu=prior_mean, sigma=prior_std)
                sigma = pm.HalfNormal('sigma', sigma=5.0)
                
                # Likelihood
                obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
                
                # Sample — reduced for speed (esp. without C compiler)
                trace = pm.sample(
                    draws=200,
                    tune=200,
                    cores=1,
                    chains=1,
                    progressbar=False,
                    return_inferencedata=True,
                )
            
            # Extract posterior samples
            mu_samples = trace.posterior['mu'].values.flatten()
            sigma_samples = trace.posterior['sigma'].values.flatten()
            
            # Generate predictive samples
            pred_samples = np.random.normal(mu_samples, sigma_samples)
            pred_samples = np.maximum(0, pred_samples)
            
            # Calculate prior vs data weight
            n = len(data)
            data_weight = n / (n + (prior_std / np.std(data, ddof=1)) ** 2) if np.std(data, ddof=1) > 0 else 0.5
            
            return {
                'mean': float(np.mean(mu_samples)),
                'std': float(np.std(pred_samples)),
                'ci_lower': float(np.percentile(pred_samples, 10)),
                'ci_upper': float(np.percentile(pred_samples, 90)),
                'samples': pred_samples[:n_samples],
                'prior_weight': 1.0 - data_weight,
                'data_weight': data_weight,
                'method': 'pymc_hierarchical',
                'n_games': n,
                'posterior_mu_mean': float(np.mean(mu_samples)),
                'posterior_mu_std': float(np.std(mu_samples)),
                'posterior_sigma_mean': float(np.mean(sigma_samples)),
            }
            
        except Exception as e:
            logger.warning(f"PyMC fitting failed: {e}. Falling back to analytical.")
            return self._fit_analytical(data, role_prior, None, n_samples)
    
    def _fit_analytical(self, data: np.ndarray, role_prior: Dict,
                         team_context: Optional[Dict],
                         n_samples: int) -> Dict[str, Any]:
        """
        Analytical conjugate normal-normal posterior (fast fallback).
        
        With known variance (estimated from data or prior):
          posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / total_precision
          posterior_precision = prior_precision + data_precision
        """
        prior_mean = role_prior['mean']
        prior_std = role_prior['std']
        
        if team_context:
            team_avg = team_context['team_avg_minutes']
            prior_mean = 0.7 * prior_mean + 0.3 * team_avg
        
        if len(data) == 0:
            # Pure prior
            samples = np.random.normal(prior_mean, prior_std, n_samples)
            samples = np.maximum(0, samples)
            return {
                'mean': prior_mean,
                'std': prior_std,
                'ci_lower': float(np.percentile(samples, 10)),
                'ci_upper': float(np.percentile(samples, 90)),
                'samples': samples,
                'prior_weight': 1.0,
                'data_weight': 0.0,
                'method': 'prior_only',
                'n_games': 0,
            }
        
        data_mean = np.mean(data)
        data_std = np.std(data, ddof=1) if len(data) > 1 else prior_std
        n = len(data)
        
        # Conjugate normal-normal update
        prior_precision = 1.0 / (prior_std ** 2)
        data_precision = n / (data_std ** 2) if data_std > 0 else 0
        
        total_precision = prior_precision + data_precision
        posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / total_precision
        posterior_std = 1.0 / np.sqrt(total_precision)
        
        # Data weight: how much the posterior is driven by data vs prior
        data_weight = data_precision / total_precision
        
        # Predictive distribution includes both parameter uncertainty and observation noise
        predictive_std = np.sqrt(posterior_std ** 2 + data_std ** 2)
        
        samples = np.random.normal(posterior_mean, predictive_std, n_samples)
        samples = np.maximum(0, samples)
        
        return {
            'mean': float(posterior_mean),
            'std': float(predictive_std),
            'ci_lower': float(np.percentile(samples, 10)),
            'ci_upper': float(np.percentile(samples, 90)),
            'samples': samples,
            'prior_weight': float(1.0 - data_weight),
            'data_weight': float(data_weight),
            'method': 'analytical_conjugate',
            'n_games': n,
        }
    
    def get_minutes_comparison(self, player_id: int, 
                                team_abbr: str = None) -> Dict[str, Any]:
        """
        Compare Bayesian vs simple average minutes prediction.
        Useful for debugging and understanding model behavior.
        """
        minutes_data = self.get_player_minutes_history(player_id)
        bayes_result = self.predict_minutes_bayesian(player_id, team_abbr)
        
        simple_mean = np.mean(minutes_data) if len(minutes_data) > 0 else 0
        simple_std = np.std(minutes_data) if len(minutes_data) > 1 else 0
        
        return {
            'player_id': player_id,
            'n_games': len(minutes_data),
            'simple_mean': float(simple_mean),
            'simple_std': float(simple_std),
            'bayesian_mean': bayes_result['mean'],
            'bayesian_std': bayes_result['std'],
            'bayesian_ci': (bayes_result['ci_lower'], bayes_result['ci_upper']),
            'prior_weight': bayes_result['prior_weight'],
            'data_weight': bayes_result['data_weight'],
            'method': bayes_result['method'],
            'shrinkage': simple_mean - bayes_result['mean'] if simple_mean > 0 else 0,
        }
    
    def clear_cache(self):
        """Clear cached posteriors."""
        self._posterior_cache.clear()


# Convenience function
def get_bayesian_minutes(db=None):
    """Get BayesianMinutesModel instance."""
    return BayesianMinutesModel(db=db)
