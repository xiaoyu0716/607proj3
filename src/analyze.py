import os
import sys
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
RESULTS = os.path.join(PROJECT_ROOT, "results")

# Recompute metrics from estimates files to ensure decoupled analysis step

def compute_metrics(estimates_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    truth = estimates_df["truth"].iloc[0]
    for method in sorted(estimates_df["method"].unique()):
        vals = estimates_df.loc[estimates_df["method"] == method, "estimate"].to_numpy()
        bias = float(np.mean(vals) - truth)
        variance = float(np.var(vals, ddof=1))
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = float(q75 - q25)
        rows.append({
            "method": method,
            "truth": truth,
            "bias": bias,
            "variance": variance,
            "IQR": iqr,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # analyze both settings if available
    for setting in ["A", "B"]:
        est_path = os.path.join(RESULTS, f"estimates_{setting}.csv")
        if not os.path.exists(est_path):
            continue
        df_est = pd.read_csv(est_path)
        df_met = compute_metrics(df_est)
        df_met.insert(1, "setting", setting)
        df_met.to_csv(os.path.join(RESULTS, f"metrics_{setting}.csv"), index=False)
        print(f"Wrote metrics for setting {setting}")

if __name__ == "__main__":
    main()
