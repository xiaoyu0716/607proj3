import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dgps import generate_data


def test_setting_A_dgp_properties(cfg_A):
    n = 3000
    beta = np.array(cfg_A["beta_true"], dtype=float)
    alpha = np.array(cfg_A["alpha_true"], dtype=float)

    data, truth, b_used, a_used = generate_data(n=n, setting="A", seed=123, beta=beta, alpha=alpha)

    # Basic columns
    assert set(["V1","V2","V3","V4","Y","R"]).issubset(set(data.columns))
    # R is binary
    assert set(np.unique(data["R"]).tolist()).issubset({0,1})

    # V distribution rough checks
    for v in ["V1","V2","V3","V4"]:
        m = data[v].mean()
        s2 = data[v].var()
        assert abs(m) < 0.1
        assert 0.7 < s2 < 1.3

    # Truth checks: mu_true = beta0 + beta1 (E[V1^2]=1, others 0)
    mu_true = truth["mu_true"]
    assert np.isclose(mu_true, beta[0] + beta[1])


def test_setting_B_dgp_properties(cfg_B):
    n = 3000
    beta = np.array(cfg_B["beta_true"], dtype=float)
    alpha = np.array(cfg_B["alpha_true"], dtype=float)

    data, truth, b_used, a_used = generate_data(n=n, setting="B", seed=100, beta=beta, alpha=alpha)

    # Columns
    assert set(["A","V1","V2","V3","V4","Y"]).issubset(set(data.columns))
    # A is binary
    assert set(np.unique(data["A" ]).tolist()).issubset({0,1})

    # V checks
    for v in ["V1","V2","V3","V4"]:
        m = data[v].mean()
        s2 = data[v].var()
        assert abs(m) < 0.1
        assert 0.7 < s2 < 1.3

    # Truth checks: ATE_true = beta1
    assert np.isclose(truth["ATE_true"], beta[1])
