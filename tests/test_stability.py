import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from methods import _ols_fit as ols_unstable, _logit_fit_irls as irls_unstable
from methods_stable import _ols_fit as ols_stable, _logit_fit_irls as irls_stable

class TestNumericalStability(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 100
        
        # Create Singular Matrix X (perfect multicollinearity)
        # X1 is random, X2 is exactly 2 * X1
        x1 = np.random.normal(size=self.n)
        x2 = 2 * x1
        self.X_singular = np.column_stack([x1, x2])
        
        # Target variable
        self.y_continuous = 3 * x1 + np.random.normal(size=self.n)
        self.y_binary = (self.y_continuous > 0).astype(float)

    def test_ols_singularity(self):
        print("\nTesting OLS with Singular Matrix...")
        
        # 1. Unstable method should fail (or be unstable)
        # Note: numpy.linalg.solve sometimes raises LinAlgError, sometimes returns garbage for singular matrices depending on LAPACK
        try:
            beta_unstable = ols_unstable(self.X_singular, self.y_continuous)
            print("Unstable OLS: Returned result (might be garbage)")
        except np.linalg.LinAlgError:
            print("Unstable OLS: Crashed as expected (LinAlgError)")
        except Exception as e:
            print(f"Unstable OLS: Crashed with {e}")

        # 2. Stable method should handle it via pseudo-inverse fallback
        try:
            beta_stable = ols_stable(self.X_singular, self.y_continuous)
            print("Stable OLS: Successfully returned result")
            self.assertTrue(np.all(np.isfinite(beta_stable)), "Stable OLS result contains NaNs or Infs")
        except Exception as e:
            self.fail(f"Stable OLS failed unexpectedly: {e}")

    def test_irls_singularity(self):
        print("\nTesting IRLS (Logistic) with Singular Matrix...")
        
        # 1. Unstable method
        try:
            beta_unstable = irls_unstable(self.X_singular, self.y_binary, max_iter=10)
            print("Unstable IRLS: Returned result")
        except np.linalg.LinAlgError:
            print("Unstable IRLS: Crashed as expected (LinAlgError)")
        except Exception as e:
            print(f"Unstable IRLS: Crashed with {e}")
            
        # 2. Stable method
        try:
            beta_stable = irls_stable(self.X_singular, self.y_binary, max_iter=10)
            print("Stable IRLS: Successfully returned result")
            self.assertTrue(np.all(np.isfinite(beta_stable)), "Stable IRLS result contains NaNs or Infs")
        except Exception as e:
            self.fail(f"Stable IRLS failed unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
