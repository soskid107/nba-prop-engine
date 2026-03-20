
import sys
import os
import sqlite3
from unittest.mock import MagicMock

sys.path.append(os.getcwd())

from src.models.variance_model import VarianceModel
from src.utils.database import DatabaseManager

def test_shooting_form():
    print("Testing Shooting Form Reversion Logic...")
    
    # Mock Database and VarianceModel
    db = MagicMock(spec=DatabaseManager)
    vm = VarianceModel(db)
    
    # Mock get_player_stats_summary to avoid DB calls
    def mock_get_stats(player_id, window=15):
        if window == 30: # Season Baseline
            return {
                'avg_minutes': 30.0,
                'avg_points': 20.0, 
                'ppm': 0.66, # 20 pts / 30 min
                'games': 30
            }
        elif window == 5: # Recent Form
            if player_id == 1: # SLUMP CASE
                # Vol maintained (30min) but EFF dropped (10pts)
                return {
                    'avg_minutes': 30.0,
                    'avg_points': 10.0,
                    'ppm': 0.33,
                    'games': 5
                }
            elif player_id == 2: # HOT CASE
                # Vol maintained (30min) but EFF huge (30pts)
                return {
                    'avg_minutes': 30.0,
                    'avg_points': 30.0,
                    'ppm': 1.0,
                    'games': 5
                }
            elif player_id == 3: # LOW VOL CASE (Benching)
                # Vol dropped (15min)
                return {
                    'avg_minutes': 15.0,
                    'avg_points': 10.0,
                    'ppm': 0.66,
                    'games': 5
                }
        return {}

    vm.get_player_stats_summary = mock_get_stats
    
    # 1. Test Slump (Reversion Candidate)
    print("\n[Case 1] Slump (High Vol, Low Eff)")
    res1 = vm.get_shooting_form_reversion(1)
    print(f"  Result: {res1}")
    assert res1['status'] == 'slump_reversion', "Should detect slump reversion"
    assert res1['multiplier'] > 1.0, "Should boost multiplier"
    
    # 2. Test Hot Streak (Sustainable?)
    print("\n[Case 2] Hot Streak (High Eff)")
    res2 = vm.get_shooting_form_reversion(2)
    print(f"  Result: {res2}")
    assert res2['status'] == 'hot_dampen', "Should detect hot streak"
    assert res2['multiplier'] < 1.0, "Should dampen multiplier"
    
    # 3. Test Low Volume (Benching)
    print("\n[Case 3] Low Volume (Role Reduction)")
    res3 = vm.get_shooting_form_reversion(3)
    print(f"  Result: {res3}")
    assert res3['status'] == 'normal', "Should be normal (handled by baseline)"
    assert res3['multiplier'] == 1.0, "Should not adjust multiplier"

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_shooting_form()
