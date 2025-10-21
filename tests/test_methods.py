import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dgps import generate_data
from methods import run_all_methods


def test_methods_setting_A_correct_spec_unbiased(cfg_A):
    # Large n to reduce Monte Carlo error
    n = 20000
    beta_true = np.array(cfg_A["beta_true"], dtype=float)
    alpha_true = np.array(cfg_A["alpha_true"], dtype=float)
    data, truth, _, _ = generate_data(n=n, setting="A", seed=123, beta=beta_true, alpha=alpha_true)
    res = run_all_methods(data, "A", beta_true, alpha_true, np.zeros(3), np.zeros(2))
    mu = truth["mu_true"]
    # OR true and DR true should be close to mu
    assert abs(res["OR"]["estimate"] - mu) < 0.03
    assert abs(res["DR"]["estimate"] - mu) < 0.03


def test_methods_setting_B_correct_spec_unbiased(cfg_B):
    n = 20000
    beta_true = np.array(cfg_B["beta_true"], dtype=float)
    alpha_true = np.array(cfg_B["alpha_true"], dtype=float)
    data, truth, _, _ = generate_data(n=n, setting="B", seed=10001, beta=beta_true, alpha=alpha_true)
    res = run_all_methods(data, "B", beta_true, alpha_true, np.zeros(3), np.zeros(2))
    ate = truth["ATE_true"]
    # IPW true and DR true should be close to ATE
    assert abs(res["IPW"]["estimate"] - ate) < 0.1
    assert abs(res["DR"]["estimate"] - ate) < 0.05
