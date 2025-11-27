import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Ensure we can import modules from current directory
sys.path.append(os.path.dirname(__file__))

# Import New (Parallel)
from simulation import run_simulation as run_new
# Import Old (Sequential)
try:
    from simulation_old import run_simulation as run_old
except ImportError:
    print("Error: src/simulation_old.py not found. Cannot compare.")
    sys.exit(1)

def test_correctness():
    print("\n=== 1. Correctness Check ===")
    setting = "A"
    n = 100
    reps = 10
    seed = 999
    
    print(f"Running Sequential (Old) n={n}, reps={reps}...")
    path_old, _ = run_old(setting=setting, n=n, reps=reps, seed0=seed)
    df_old = pd.read_csv(path_old).sort_values(["seed", "method"]).reset_index(drop=True)
    
    print(f"Running Parallel (New) n={n}, reps={reps}...")
    path_new = run_new(setting=setting, n=n, reps=reps, seed0=seed, n_jobs=-1)
    df_new = pd.read_csv(path_new).sort_values(["seed", "method"]).reset_index(drop=True)
    
    # Compare
    try:
        cols_to_ignore = ["truth_key"]
        df_old_cmp = df_old.drop(columns=[c for c in cols_to_ignore if c in df_old.columns])
        df_new_cmp = df_new.drop(columns=[c for c in cols_to_ignore if c in df_new.columns])

        pd.testing.assert_frame_equal(df_old_cmp, df_new_cmp, check_dtype=False)
        print("SUCCESS: Results match exactly between Sequential and Parallel versions.")
    except AssertionError as e:
        print("FAILURE: Results do not match!")
        print(e)

def benchmark_performance():
    print("\n=== 2. Performance Benchmark ===")
    # To show parallel benefits, use a larger n
    n = 10000
    # Fewer reps for the large n to keep runtime reasonable
    reps_list = [10, 20, 50] 
    
    times_old = []
    times_new = []
    
    print(f"Fixed n={n}. Varying reps: {reps_list}")
    
    for r in reps_list:
        print(f"\n--- reps={r} ---")
        
        # Time Old
        start = time.time()
        run_old(setting="A", n=n, reps=r, seed0=1)
        dur_old = time.time() - start
        times_old.append(dur_old)
        print(f"Sequential: {dur_old:.4f} s")
        
        # Time New
        start = time.time()
        run_new(setting="A", n=n, reps=r, seed0=1, n_jobs=-1)
        dur_new = time.time() - start
        times_new.append(dur_new)
        print(f"Parallel:   {dur_new:.4f} s")
        
        speedup = dur_old / dur_new if dur_new > 0 else 0
        print(f"Speedup:    {speedup:.2f}x")

    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(len(reps_list))
    width = 0.35
    
    plt.bar(x - width/2, times_old, width, label='Sequential', color='salmon')
    plt.bar(x + width/2, times_new, width, label='Parallel', color='skyblue')
    
    plt.xlabel('Number of Replicates (reps)')
    plt.ylabel('Total Runtime (seconds)')
    plt.title(f'Performance Comparison (n={n})')
    plt.xticks(x, reps_list)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for i in range(len(reps_list)):
        speedup = times_old[i] / times_new[i]
        y_pos = max(times_old[i], times_new[i])
        plt.text(x[i], y_pos + (y_pos*0.05), f"{speedup:.1f}x", ha='center', fontweight='bold', color='black')
        
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'benchmark_comparison_n{n}.png')
    plt.savefig(save_path)
    print(f"\nBenchmark plot saved to: {save_path}")

def check_stability(save_path=None):
    # Capture logs to write to file
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    log("\n=== 3. Stability Check (Singular Matrix) ===")
    
    try:
        from methods import _ols_fit as ols_unstable
        from methods_stable import _ols_fit as ols_stable
    except ImportError:
        log("Could not import methods for stability check.")
        return

    # 1. Test with Collinearity
    log("Scenario A: Perfect Collinearity (x2 = 2*x1)")
    n = 100
    x1 = np.random.normal(size=n)
    x2 = 2 * x1 
    X_collinear = np.column_stack([x1, x2])
    y = 3 * x1 + np.random.normal(size=n)

    log("  [Unstable Method]:")
    try:
        beta = ols_unstable(X_collinear, y)
        log(f"    Result: Returned a result (likely unreliable) -> {beta[:5]}")
    except Exception as e:
        log(f"    Result: CRASHED ({e})")
        
    log("  [Stable Method]:")
    try:
        beta = ols_stable(X_collinear, y)
        log(f"    Result: SUCCESS (Handled via pseudo-inverse) -> {beta[:5]}")
    except Exception as e:
        log(f"    Result: FAILED ({e})")

    # 2. Test with Zero Column
    log("\nScenario B: Zero Column (Matrix Rank Deficient)")
    X_zero = np.column_stack([x1, np.zeros(n)])
    
    log("  [Unstable Method]:")
    try:
        beta = ols_unstable(X_zero, y)
        log(f"    Result: Returned a result -> {beta[:5]}")
    except np.linalg.LinAlgError:
        log("    Result: CRASHED (LinAlgError) -> Caught as expected")
    except Exception as e:
        log(f"    Result: CRASHED ({e})")

    log("  [Stable Method]:")
    try:
        beta = ols_stable(X_zero, y)
        log(f"    Result: SUCCESS (Handled via pseudo-inverse) -> {beta[:5]}")
    except Exception as e:
        log(f"    Result: FAILED ({e})")
    
    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(logs))
        print(f"\nStability report saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tests.")
    parser.add_argument("--test", type=str, default="all", choices=["correctness", "speed", "stability", "all"], help="Which test to run")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.test in ["correctness", "all"]:
        test_correctness()
    
    if args.test in ["speed", "all"]:
        benchmark_performance()
        
    if args.test in ["stability", "all"]:
        stability_path = os.path.join(results_dir, "stability_report.txt")
        check_stability(save_path=stability_path)
