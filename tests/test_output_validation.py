import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from simulation import run_simulation


def test_output_files_created(tmp_path):
    # Run a very small simulation
    est_path, met_path = run_simulation(setting="A", n=200, reps=5, seed0=999)
    assert os.path.exists(est_path)
    assert os.path.exists(met_path)

    df_est = pd.read_csv(est_path)
    df_met = pd.read_csv(met_path)
    # sanity checks
    assert {"method","estimate","truth","bias"}.issubset(set(df_est.columns))
    assert {"method","bias","variance","IQR"}.issubset(set(df_met.columns))

    # Bias within some reasonable range around 0 for DR when true
    dr_row = df_met[df_met["method"] == "DR"].iloc[0]
    assert abs(dr_row["bias"]) < 0.2


def test_reproducibility():
    # Same seed reproduces identical outputs
    p1 = run_simulation(setting="A", n=500, reps=3, seed0=4242)
    p2 = run_simulation(setting="A", n=500, reps=3, seed0=4242)
    est1, met1 = p1
    est2, met2 = p2
    assert pd.read_csv(est1).equals(pd.read_csv(est2))
    assert pd.read_csv(met1).equals(pd.read_csv(met2))
