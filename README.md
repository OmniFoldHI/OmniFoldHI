# OmniFoldHI

This repository contains the data and code used to produce the results presented in our work: [arXiv:2507.06291](https://arxiv.org/abs/2507.06291).

## Repository Structure

The repository is organized as follows:

- **`data/`** — Event-by-event jet observables used in the analysis. See the paper for details on event generation.
- **`src/`** — Implementation of the iterative Bayesian unfolding and the OmniFoldHI algorithms.
- **`analysis/`** — Scripts for running the analysis and reproducing the paper's results.
- **`figures/`** — Output plots for all 18 observables (supplementary to the publication).

## Setup Instructions

Before running the analysis, ensure all required packages are installed. You can set up the environment using:

```bash
conda env create -f environment.yml -n omnifoldHI_env
conda activate omnifoldHI_env
```

Unpack the data files:
```bash
cd data
tar -xf *.zip
```

## Plotting Jet Observables
To generate plots of the truth and measured jet observables
```bash
cd analysis/plot_data
python plot_data.py
open ../figures/data.pdf
```

## Run the Main Analysis
```bash
cd analysis
python analysis_multifoldHI.py
open ../figures/omnifoldHI_pages.pdf
```
This script reproduces Figure 6 of the paper by unfolding 3, 7, 12, and 18 observables using OmniFoldHI, including both statistical and systematic uncertainties. The configuration for each unfolding is specified in the corresponding `config_anl_XXX.json` file. An unfolding with uncertainties takes approximately 30 minutes.
