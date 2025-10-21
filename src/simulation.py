import os
import sys
import numpy as np
import pandas as pd
import json

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from dgps import generate_data
from methods import run_all_methods
from metrics import evaluate


def run_simulation(setting="A", n=1000, reps=100, seed0=1, beta_true=None, alpha_true=None, beta_false=None, alpha_false=None):
    # Load configuration
    project_root = os.path.dirname(THIS_DIR)
    config_path = os.path.join(project_root, "config", "params.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg_set = cfg["settings"][setting]
    V_config = cfg_set.get("V", None)

    # Resolve parameters from config if not provided
    if setting == "A":
        if beta_true is None:
            beta_true = np.array(cfg_set.get("beta_true", [0.0, 1.0, 2.5, 3.0]))
        if alpha_true is None:
            alpha_true = np.array(cfg_set.get("alpha_true", [-1.0, 1.0, 0.0, 0.0, -1.0]))
        if beta_false is None:
            beta_false = np.array(cfg_set.get("beta_false", [1.0, 2.5, 3.0]))  # [1, V1, V2^2]
        if alpha_false is None:
            alpha_false = np.array(cfg_set.get("alpha_false", [1.0, -1.0]))  # [I1, I3] no intercept
    elif setting == "B":
        if beta_true is None:
            beta_true = np.array(cfg_set.get("beta_true", [0.0, 2.0, 3.0, 2.0, -4.0]))
        if alpha_true is None:
            alpha_true = np.array(cfg_set.get("alpha_true", [-3.0, 2.5, 3.0, 1.0, -3.0]))
        if beta_false is None:
            beta_false = np.array(cfg_set.get("beta_false", [0.0, 2.0, 3.0]))  # [1, V1*A, V2^2*(1-A)]
        if alpha_false is None:
            alpha_false = np.array(cfg_set.get("alpha_false", [0.0, 2.0]))  # [I3, V4] no intercept

    # Prepare results directory and raw incremental file
    results_dir = os.path.join(os.path.dirname(THIS_DIR), "results")
    os.makedirs(results_dir, exist_ok=True)
    raw_path = os.path.join(results_dir, f"raw_estimates_{setting}.csv")
    # Start fresh raw file for this run
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
        except OSError:
            pass

    rows = []
    truth_value = None
    truth_key = None
    for r in range(reps):
        seed = seed0 + r
        data, truth, beta_used, alpha_used = generate_data(
            n=n,
            setting=setting,
            seed=seed,
            beta=beta_true,
            alpha=alpha_true,
            V_config=V_config,
        )
        if truth_value is None:
            if "mu_true" in truth:
                truth_value = float(truth["mu_true"])  # type: ignore
                truth_key = "mu_true"
            elif "ATE_true" in truth:
                truth_value = float(truth["ATE_true"])  # type: ignore
                truth_key = "ATE_true"
            else:
                raise ValueError("Truth dict must contain 'mu_true' or 'ATE_true'.")

        res = run_all_methods(data, setting, beta_true, alpha_true, beta_false, alpha_false)
        # Collect current replicate rows
        batch = []
        for m, v in res.items():
            batch.append({
                "seed": seed,
                "method": m,
                "estimate": float(v.get("estimate", np.nan)),
                "bias": float(v.get("estimate", np.nan) - truth_value),
                "alpha": v.get("alpha"),
                "beta": v.get("beta"),
                "propensity_spec": v.get("propensity_spec"),
                "outcome_spec": v.get("outcome_spec"),
                "setting": setting,
                "n": n,
                "truth": truth_value,
            })
        # Append to in-memory
        rows.extend(batch)
        # Write intermediate raw outputs (append mode)
        df_batch = pd.DataFrame(batch)
        df_batch.to_csv(raw_path, mode="a", header=not os.path.exists(raw_path), index=False)
        # Print last 3-5 lines as real-time snapshot
        preview = df_batch[["seed", "method", "estimate", "bias"]].tail(5)
        print("[Intermediate] setting=", setting, "rep=", r+1, "/", reps)
        print(preview.to_string(index=False))

    df_est = pd.DataFrame(rows)
    est_path = os.path.join(results_dir, f"estimates_{setting}.csv")
    df_est.to_csv(est_path, index=False)

    metric_rows = []
    method_names = sorted(df_est["method"].unique())
    for m in method_names:
        vals = df_est.loc[df_est["method"] == m, "estimate"].to_numpy()
        metrics = evaluate(vals, {truth_key: truth_value})  # type: ignore
        metric_rows.append({
            "method": m,
            "setting": setting,
            "n": n,
            "reps": reps,
            "truth": truth_value,
            **metrics,
        })

    df_metrics = pd.DataFrame(metric_rows)
    met_path = os.path.join(results_dir, f"metrics_{setting}.csv")
    df_metrics.to_csv(met_path, index=False)

    return est_path, met_path


if __name__ == "__main__":
    run_simulation(setting="A", n=1000, reps=100, seed0=1)
    run_simulation(setting="B", n=1000, reps=100, seed0=10001)
