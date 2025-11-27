
import time
import cProfile
import pstats
import io
import sys
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from simulation import run_simulation

# Redirect stderr to capture warnings
class WarningCatcher:
    def __init__(self):
        self.warnings = []
    
    def write(self, message):
        if "Warning" in message or "warning" in message:
            self.warnings.append(message)
        sys.__stderr__.write(message)
    
    def flush(self):
        sys.__stderr__.flush()

def measure_total_runtime():
    print("\n--- Measuring Total Runtime (Baseline) ---")
    start_time = time.time()
    # Run a smaller version for baseline to avoid waiting too long, 
    # or run the full default if needed. The prompt asks for "current simulation".
    # Default in simulation.py is n=1000, reps=100. 
    # We will run specific settings to be consistent.
    run_simulation(setting="A", n=1000, reps=50, seed0=1)
    run_simulation(setting="B", n=1000, reps=50, seed0=10001)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total runtime for A&B (n=1000, reps=50): {duration:.4f} seconds")
    return duration

def profile_code():
    print("\n--- Running cProfile ---")
    pr = cProfile.Profile()
    pr.enable()
    # Run just one setting for profiling to identify bottlenecks
    run_simulation(setting="A", n=1000, reps=10, seed0=1)
    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Print top 20 lines
    print(s.getvalue())
    return s.getvalue()

def complexity_analysis():
    print("\n--- Running Complexity Analysis ---")
    ns = [100, 500, 1000, 2000, 5000]
    times = []
    
    for n in ns:
        start = time.time()
        # Run a small number of reps just to gauge 'per run' time or total time
        run_simulation(setting="A", n=n, reps=5, seed0=1)
        end = time.time()
        avg_time = (end - start) / 5 # Average time per rep roughly
        times.append(avg_time)
        print(f"n={n}: {avg_time:.4f} seconds/rep")
    
    # Plotting log-log
    plt.figure(figsize=(8, 6))
    plt.loglog(ns, times, 'o-', label='Measured Time')
    
    # Fit a line: log(t) = k * log(n) + c
    log_n = np.log(ns)
    log_t = np.log(times)
    coeffs = np.polyfit(log_n, log_t, 1)
    poly = np.poly1d(coeffs)
    slope = coeffs[0]
    
    plt.loglog(ns, np.exp(poly(log_n)), '--', label=f'Fit (slope={slope:.2f})')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Time per Replicate (s)')
    plt.title('Computational Complexity Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('../results/complexity_baseline.png')
    print(f"Complexity plot saved to results/complexity_baseline.png. Slope: {slope:.2f}")
    return ns, times, slope

def check_stability():
    print("\n--- Checking Stability ---")
    # Capture warnings
    original_stderr = sys.stderr
    catcher = WarningCatcher()
    sys.stderr = catcher
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Run a case that might trigger instability (small n or specific seed?)
        # Or just run the standard one.
        run_simulation(setting="A", n=1000, reps=10, seed0=1)
    
    sys.stderr = original_stderr
    
    if len(catcher.warnings) > 0:
        print("Warnings captured:")
        for msg in catcher.warnings:
            print(msg)
    else:
        print("No stderr warnings captured.")

import argparse

# ... existing imports ...

if __name__ == "__main__":
    # Ensure results dir exists for plots
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run baseline benchmarks.")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["runtime", "profile", "complexity", "stability", "all"], 
                        help="Which benchmark mode to run")
    args = parser.parse_args()
    
    if args.mode in ["runtime", "all"]:
        measure_total_runtime()
        
    if args.mode in ["profile", "all"]:
        profile_output = profile_code()
        
    if args.mode in ["complexity", "all"]:
        complexity_analysis()
        
    if args.mode in ["stability", "all"]:
        check_stability()
