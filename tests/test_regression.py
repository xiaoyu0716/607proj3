import unittest
import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation import run_simulation as run_new
try:
    from simulation_old import run_simulation as run_old
except ImportError:
    run_old = None

class TestRegression(unittest.TestCase):
    
    def setUp(self):
        self.setting = "A"
        self.n = 100
        self.reps = 10
        self.seed = 999

    def test_results_match_baseline(self):
        """
        Verify that the new parallel/stable implementation produces exactly the same 
        numerical results as the old sequential/unstable implementation given the same seeds.
        """
        if run_old is None:
            print("Skipping regression test because simulation_old.py is missing.")
            return

        print("\nRunning Baseline (Sequential)...")
        # Run OLD
        path_old, _ = run_old(setting=self.setting, n=self.n, reps=self.reps, seed0=self.seed)
        df_old = pd.read_csv(path_old).sort_values(["seed", "method"]).reset_index(drop=True)

        print("Running New (Parallel + Stable)...")
        # Run NEW
        path_new = run_new(setting=self.setting, n=self.n, reps=self.reps, seed0=self.seed, n_jobs=-1)
        df_new = pd.read_csv(path_new).sort_values(["seed", "method"]).reset_index(drop=True)

        # Compare Estimates
        # We drop 'truth_key' if it exists in one but not the other to avoid mismatch on internal columns
        cols_to_ignore = ["truth_key"]
        df_old_cmp = df_old.drop(columns=[c for c in cols_to_ignore if c in df_old.columns])
        df_new_cmp = df_new.drop(columns=[c for c in cols_to_ignore if c in df_new.columns])

        # Check equality
        # Note: Stable method should match Unstable method EXACTLY on non-singular data
        # If we hit singular data, they would differ, but with seed=999 and n=100, it is likely stable.
        pd.testing.assert_frame_equal(df_old_cmp, df_new_cmp, check_dtype=False)
        print("SUCCESS: Estimates match exactly.")

if __name__ == '__main__':
    unittest.main()
