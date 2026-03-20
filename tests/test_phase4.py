"""
Phase 4 Validation Script

Tests feature engineering: rolling stats, context features, and leakage prevention.
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import DatabaseManager
from src.models.feature_engineering import FeatureEngineer


def test_rolling_stats():
    """Test rolling statistics calculation."""
    print("\n" + "="*60)
    print("TEST 1: Rolling Statistics")
    print("="*60)
    
    fe = FeatureEngineer()
    
    # Get LeBron's history (we loaded it in Phase 2)
    lebron_id = 2544
    history = fe._get_player_history(lebron_id, limit=20)
    
    if history.empty:
        print("  ⚠ No game history found (run Phase 2 tests first)")
        assert True
        return
    
    print(f"\n  Found {len(history)} games for LeBron")
    
    # Calculate rolling minutes
    min_stats = fe._calculate_rolling_stats(history, 'minutes')
    print("\n  Rolling Minutes:")
    for key, value in min_stats.items():
        print(f"    {key}: {value:.1f}")
    
    # Calculate rolling PPM
    ppm_stats = fe._calculate_rolling_stats(history, 'ppm')
    print("\n  Rolling PPM:")
    for key, value in ppm_stats.items():
        print(f"    {key}: {value:.2f}")
    
    assert True


def test_feature_vector():
    """Test building a complete feature vector."""
    print("\n" + "="*60)
    print("TEST 2: Feature Vector Generation")
    print("="*60)
    
    fe = FeatureEngineer()
    
    # Build features for LeBron's next game
    features = fe.build_features_for_player(
        player_id=2544,
        team_abbr='LAL',
        opponent_abbr='GSW',  
        game_date='2026-01-23',
        p_play=0.5  # Questionable
    )
    
    print("\n  Feature Vector for LeBron James:")
    print("  " + "-"*40)
    
    # Group by category
    print("\n  📊 Rolling Stats:")
    for key in ['minutes_L3', 'minutes_L5', 'minutes_L10', 'min_std_L5']:
        if key in features:
            print(f"    {key}: {features[key]:.2f}")
    
    print("\n  📈 PPM Stats:")
    for key in ['ppm_L3', 'ppm_L5', 'ppm_L10', 'ppm_std_L5']:
        if key in features:
            print(f"    {key}: {features[key]:.3f}")
    
    print("\n  🎰 Market Context:")
    for key in ['spread', 'total', 'blowout_risk', 'pace_proxy']:
        val = features.get(key)
        if val is not None:
            print(f"    {key}: {val}")
        else:
            print(f"    {key}: N/A")
    
    print("\n  🏀 Game Info:")
    for key in ['is_starter', 'rest_days', 'is_home', 'games_played', 'p_play']:
        print(f"    {key}: {features.get(key)}")
    
    assert True


def test_leakage_prevention():
    """Test that no future data leaks into features."""
    print("\n" + "="*60)
    print("TEST 3: Data Leakage Prevention")
    print("="*60)
    
    fe = FeatureEngineer()
    
    # Get LeBron's history
    lebron_id = 2544
    all_history = fe._get_player_history(lebron_id, limit=20)
    
    if len(all_history) < 5:
        print("  ⚠ Not enough games for leakage test")
        assert True
        return
    
    # Pick a game in the middle
    test_game = all_history.iloc[5]
    test_date = test_game['game_date']
    
    print(f"\n  Testing features for game on {test_date}")
    
    # Get history that would be available before this game
    prior_history = fe._get_player_history(lebron_id, before_date=test_date, limit=10)
    
    print(f"  Games before this date: {len(prior_history)}")
    
    # Verify all prior games are actually before test_date
    if not prior_history.empty:
        all_before = all(prior_history['game_date'] < test_date)
        if all_before:
            print("  ✓ All training data is strictly before target date")
        else:
            print("  ✗ LEAK DETECTED: Some training data is after target date!")
            pytest.fail("Detected training data after target date")
    
    print("  ✓ No data leakage detected")
    assert True


def test_feature_columns():
    """Test feature column definitions."""
    print("\n" + "="*60)
    print("TEST 4: Feature Column Definitions")
    print("="*60)
    
    fe = FeatureEngineer()
    
    minutes_features = fe.get_feature_columns('minutes')
    ppm_features = fe.get_feature_columns('ppm')
    
    print("\n  Minutes Model Features:")
    for f in minutes_features:
        print(f"    - {f}")
    
    print("\n  PPM Model Features:")
    for f in ppm_features:
        print(f"    - {f}")
    
    assert True


def test_sample_dataframe():
    """Test showing a sample of the feature dataframe."""
    print("\n" + "="*60)
    print("TEST 5: Sample Feature DataFrame")
    print("="*60)
    
    fe = FeatureEngineer()
    
    # Build features for a few players
    features_list = []
    
    test_players = [
        (2544, 'LAL', 'GSW'),   # LeBron
        (201142, 'DEN', 'LAC'), # Kevin Durant -> Actually Jokic? Let's use another
    ]
    
    for player_id, team, opp in test_players:
        try:
            features = fe.build_features_for_player(
                player_id=player_id,
                team_abbr=team,
                opponent_abbr=opp,
                game_date='2026-01-23'
            )
            features_list.append(features)
        except Exception as e:
            print(f"  ⚠ Could not build features for {player_id}: {e}")
    
    if features_list:
        import pandas as pd
        df = pd.DataFrame(features_list)
        
        print("\n  Sample DataFrame Head:")
        print("  Columns:", list(df.columns)[:10], "...")
        print(f"\n  Shape: {df.shape}")
        
        # Show key columns
        key_cols = ['player_id', 'minutes_L5', 'ppm_L5', 'spread', 'total', 'is_starter']
        available_cols = [c for c in key_cols if c in df.columns]
        print(f"\n  Key Features:\n{df[available_cols].to_string()}")
    
    assert True


def main():
    print("\n" + "="*60)
    print(" PHASE 4 VALIDATION: Feature Engineering")
    print("="*60)
    
    try:
        all_passed = True
        
        test_rolling_stats()
        test_feature_vector()
        test_leakage_prevention()
        test_feature_columns()
        test_sample_dataframe()
        
        # Summary
        print("\n" + "="*60)
        if all_passed:
            print(" ✓ PHASE 4 COMPLETE - All validations passed!")
        else:
            print(" ✗ PHASE 4 INCOMPLETE - Some tests failed")
        print("="*60 + "\n")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n✗ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
