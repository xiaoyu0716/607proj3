# Baseline Performance Analysis

## 1. Total Runtime
The current simulation runs sequentially (single-threaded).
- **Settings**: n=1000, reps=50 (for both Setting A and B)
- **Total Runtime**: ~2.03 seconds
- **Estimated time for default config (reps=100)**: ~4-5 seconds.
- **Estimated time for larger scale (e.g., n=5000, reps=100)**: ~92 seconds (based on complexity analysis).

## 2. Profiling Results
Profiling was performed using `cProfile` on Setting A (n=1000, reps=10).

**Key Bottlenecks identified:**
1. **Linear Algebra Operations**: `numpy.linalg.solve` and matrix multiplications (`@` operator) in `_logit_fit_irls` and `_ols_fit`.
2. **Data Generation**: Random number generation in `dgps.py` accounts for a significant portion when $n$ is large.
3. **Sequential Execution**: The `run_simulation` loop runs replicates one after another, utilizing only a single CPU core.

## 3. Computational Complexity
We measured the runtime per replicate for varying sample sizes $n \in \{100, 500, 1000, 2000, 5000\}$.

- **Empirical Relationship**: The runtime scales with $n$ approximately as $O(n^{1.29})$.
- **Slope**: 1.29 (on log-log plot).
- **Implication**: As $n$ increases, runtime grows slightly faster than linearly. For very large $n$, this will become a significant bottleneck.


## 4. Numerical Stability
- **Current Status**: No `RuntimeWarning` or convergence issues were captured during the baseline run.
- **Potential Risks**: The `_sigmoid` function in `methods.py` uses `np.clip` to avoid overflow, which is good. However, manual implementation of IRLS (Iteratively Reweighted Least Squares) in `_logit_fit_irls` can be unstable if the design matrix is singular or perfect separation occurs.

## 5. Planned Optimizations
Based on this baseline, the following optimizations are planned:
1. **Parallelization**: Implement `multiprocessing` to run replicates in parallel. This should provide a near-linear speedup proportional to the number of CPU cores.
2. **Array Programming / Vectorization**: Review `dgps.py` to ensure all random number generation is fully vectorized.
3. **Algorithmic Improvements**: Verify if IRLS can be replaced or optimized, though parallelization is the highest priority.
