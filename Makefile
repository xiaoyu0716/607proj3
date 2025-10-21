# Simulation pipeline automation
# Usage:
#   make all       # run simulate -> analyze -> figures
#   make simulate  # run simulations and save raw results
#   make analyze   # recompute metrics from estimates/raws
#   make figures   # generate visualizations
#   make clean     # remove generated files
#   make test      # run test suite

PYTHON := python
PROJECT_ROOT := $(shell pwd)
RESULTS := $(PROJECT_ROOT)/results
FIGURES := $(RESULTS)/figures

.PHONY: all simulate analyze figures clean test

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

