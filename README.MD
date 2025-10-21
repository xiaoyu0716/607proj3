# Unit 2 Project – Simulation Study (Bang & Robins, 2005)

## Brief description
- **Goal**: Reproduce and extend findings from Bang & Robins (2005) on doubly robust (DR) estimation versus inverse probability weighting (IPW) and outcome regression (OR).
- **Settings**: 
  - `A`: Missing-data, nonlongitudinal model (estimand: population mean `μ`).
  - `B`: Treatment-effect model (estimand: average treatment effect `ATE`).
- **Methods**: IPW, OR, DR with true/false model specifications. Code lives in `src/` and results in `results/`.

## Setup
- Python 3.10+ recommended.
- Create a virtual environment and install dependencies:
  
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  ```

## How to run
- The Makefile automates the full pipeline. From the project root:

  ```bash
  make all       # runs: simulate -> analyze -> figures
  ```

- Individual steps if needed:
  - `make simulate`  # run simulations and write `results/estimates_*.csv` and `results/raw_estimates_*.csv`
  - `make analyze`   # recompute `results/metrics_*.csv` from estimates
  - `make figures`   # generate plots into `results/figures/`
  - `make clean`     # remove generated CSVs and PNGs
  - `make test`      # run unit tests

Outputs are written to `results/` and figures to `results/figures/`.

## Estimated runtime
- Default settings in `src/simulation.py` run `n=1000`, `reps=100` for both settings A and B.
- On a typical laptop, end-to-end (`make all`) takes about **3–6 minutes**. Runtime varies by CPU and BLAS.

## Summary of key findings
- Doubly robust estimators remain near-unbiased under single-model misspecification, while IPW and OR show noticeable bias when their respective models are false. Visualizations in `results/figures/` highlight DR’s smaller bias and tighter spread under misspecification.

## References and reports
- Paper: `Bang-DoublyRobustEstimation-2005.pdf`.
- Project writeups: `ADEMP.md`, `Analysis.md` (and `Analysis.pdf`).