## Aims

- I chose the paper "Doubly Robust Estimation in Missing Data and Causal Inference Models" by Heejung Bang and James M. Robins published on Biometrics in 2005.
- This paper studies the performance of different methods in estimating the average treatment effect (ATE) in the presence of model misspecification and expectation in nonlongitudinal models. The hypotheses are:
    - The double robust estimator is more robust to model misspecification than the other methods requiring only one of the model assumption to be correct.
    - The IPW estimator and OR estimator are less  robust requiring the specific model assumption to be correct.
    

## Data generation mechanism

- **Covariates**: `V=(V1,V2,V3,V4)`.
 - Default `V ~ N(0, I4)`; configurable via `config/params.json` under `settings.{A|B}.V` with `mean` and `cov`.
 - Implemented in `src/dgps.py: generate_data(..., V_config=...)`.

### **Setting A (Missing data / nonlongitudinal)**
 - Outcome (true): `Y = β0 + β1·(V1^2) + β2·V2 + β3·(V2·V3) + ε`, `ε ~ N(0,1)`.
 - Missingness: `logit P(R=1|V) = α0 + α1·I1 + α2·I2 + α3·I3 + α4·(I1·I2)` with `Ij=1(Vj>0)`.
 - Observed data: only `R=1` have `Y` observed. Truth: `μ_true = β0 + β1`.
 - Parameters from `config/params.json: settings.A.{beta_true, alpha_true}`.

### **Setting B (Treatment effect)**
 - Treatment: `logit P(A=1|V) = α0 + α1·I1 + α2·I2 + α3·I3 + α4·(I1·I2)`.
 - Outcome (true): `Y = β0 + β1·(V1^2)·A + β2·V2·A + β3·(V2·V3)·(1-A) + β4·V3·(1-A) + ε`.
 - Truth: `ATE_true = β1`.
 - Parameters from `config/params.json: settings.B.{beta_true, alpha_true}`.

### **Seeding & size**:
 - `run_simulation()` in `src/simulation.py`: `seed = seed0 + r`, `n` samples per replicate, `reps` replicates.

## Estimands

- **Setting A**: `μ = E[Y]` (population mean under missingness at random).
- **Setting B**: `ATE = E[Y(1) - Y(0)]`.

## Methods

- Implemented in `src/methods.py`. Each method can use a “true” or “false” design.

- **IPW (Inverse Probability Weighting)**
 - Fit propensity `π̂(V)` by logistic regression using designs:
   - True: `[1, I1, I2, I3, I1·I2]`.
   - False (A): `[I1, I3]` (no intercept); False (B): `[I3, V4]` (no intercept).
 - Estimators:
   - A: `mean(R·Y / π̂(V))`.
   - B: `mean(A·Y/π̂ − (1−A)·Y/(1−π̂))`.

- **OR (Outcome Regression)**
 - Linear regression with designs:
   - A true: `[1, V1^2, V2, V2·V3]`; A false: `[1, V1, V2^2]`.
   - B true: `[1, V1^2·A, V2·A, V2·V3·(1−A), V3·(1−A)]`; B false: `[1, V1·A, V2^2·(1−A)]`.
 - Estimators:
   - A: predict `ŝ(V)` for all and average.
   - B: predict `ŝ1(V)` and `ŝ0(V)` then average `ŝ1 − ŝ0`.

- **DR (Doubly Robust)**
 - Combine OR and IPW:
   - A: `mean( ŝ(V) + R·(Y − ŝ(V))/π̂(V) )`.
   - B: `mean( A·(Y − ŝ1)/π̂ − (1−A)·(Y − ŝ0)/(1−π̂) + ŝ1 − ŝ0 )`.
 - Variants: `DR` (both true), `DR.ofal` (OR false), `DR.pfal` (PS false), `DR.o&pfal` (both false).

## Performance measures

- Computed per method from replicate estimates (see `src/simulation.py`, `src/analyze.py`).
 - **Bias**: `mean(estimate) − truth`.
 - **Variance**: sample variance across replicates.
 - **IQR**: 75th − 25th percentile of estimates.

- **Files** (in `results/`):
 - Raw per-replicate: `raw_estimates_{A,B}.csv` (saved during simulation).
 - Aggregated per replicate: `estimates_{A,B}.csv`.
 - Summaries: `metrics_{A,B}.csv`.

- **Visualizations** (see `src/visualization.py`, saved to `results/figures/`):
 - Bias comparison (per setting + combined).
 - Boxplots of estimates with truth line.

- **Testing & validation** (see `tests/`):
 - DGP checks (columns, binary indicators, mean/var of V, closed-form truths).
 - Method correctness under correct specification (large n ≈ truth).
 - Output validation (files exist, columns present, DR bias small).
 - Reproducibility (same seeds → identical CSV outputs).