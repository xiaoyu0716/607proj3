# Simulation pipeline automation
# Usage:
#   make all       # run simulate -> analyze -> figures
#   make simulate  # run simulations and save raw results
#   make analyze   # recompute metrics from estimates/raws
#   make figures   # generate visualizations
#   make clean     # remove generated files
#   make test      # run test suite
#
#   make profile        # Run profiling on representative simulation
#   make complexity     # Run computational complexity analysis (timing vs n)
#   make benchmark      # Run timing comparison: baseline vs optimized
#   make parallel       # Run optimized version with parallelization
#   make stability-check # Check for warnings/convergence issues across conditions
#   make regression-test # Verify optimized code produces equivalent results

PYTHON := python
PROJECT_ROOT := $(shell pwd)
RESULTS := $(PROJECT_ROOT)/results
FIGURES := $(RESULTS)/figures

.PHONY: all simulate analyze figures clean test profile complexity benchmark parallel stability-check regression-test

all: simulate analyze figures

simulate:
	@$(PYTHON) -m src.simulation

analyze:
	@$(PYTHON) -m src.analyze

figures:
	@mkdir -p $(FIGURES)
	@$(PYTHON) -m src.visualization

clean:
	@echo "Cleaning results..."
	@rm -f $(RESULTS)/estimates_*.csv $(RESULTS)/metrics_*.csv $(RESULTS)/raw_estimates_*.csv
	@rm -f $(FIGURES)/*.png

test:
	@$(PYTHON) -m pytest -q

# --- Unit 3 Optimization Targets ---

profile:
	@$(PYTHON) src/benchmark_baseline.py --mode profile

complexity:
	@$(PYTHON) src/benchmark_baseline.py --mode complexity

benchmark:
	@$(PYTHON) src/benchmark_comparison.py --test speed

parallel:
	@$(PYTHON) src/simulation.py

stability-check:
	@$(PYTHON) src/benchmark_comparison.py --test stability

regression-test:
	@$(PYTHON) tests/test_regression.py
