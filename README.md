# Bayesian Hierarchical Modeling of Air Pollution

## Overview
This project implements a Bayesian hierarchical model (BHM) to analyze air pollution data (PM2.5) across multiple monitoring stations in Italy. The focus is on multiparameter inference with hyperpriors, estimated via Markov Chain Monte Carlo (MCMC) methods.

The objective is to:
- capture heterogeneity across stations,
- model temporal patterns (month, day-of-week),
- study posterior behavior and convergence properties.

---

## Project Structure
```
bayesian_analysis/
│
├── data/
│ ├── air_quality_complete_dataset.parquet
│ └── metadata.xlsx
│
├── paper/
│ ├── figures/
│ └── paper_pm25.Rmd
│
├── scripts/
│ ├── build_dataset.py
│ ├── load_dataset.ipynb
│ └── model_pm25.Rmd
│
├── README.md
└── .gitignore
```
---

## Dataset
- Source: European Environment Agency (EEA)
- Pollutant: PM2.5
- Observations: ~31,000
- Stations: ~116

### Preprocessing
- Filtered by validity and verification
- Log transformation applied:
  y = log(PM2.5)
- Derived features:
  - month
  - day_of_week
---

## Model Specification

$$
\begin{aligned}
y_i &\sim \mathcal{N}(X_i \beta + \alpha_j, \sigma^2) \\
\alpha_j &\sim \mathcal{N}(0, \tau^2) \\
\beta &\sim \mathcal{N}(b_0, B_0) \\
\sigma^2 &\sim \text{Inverse-Gamma}(a_\sigma, b_\sigma) \\
\tau^2 &\sim \text{Inverse-Gamma}(A_\tau)
\end{aligned}
$$

---

## Inference Method

Inference is performed via MCMC sampling using a Metropolis-within-Gibbs algorithm:
- Gibbs updates for:
  - regression coefficients beta
  - variance sigma^2
  - random effects alpha_j
- Random-walk Metropolis step for tau

### Tuning
- Proposal standard deviation tuned to achieve:
  acceptance rate ≈ 20–40%
- Final choice:
  proposal_sd = 0.2
---

## Diagnostics

The following diagnostics are performed:
- Trace plots
- Autocorrelation (ACF)
- Effective Sample Size (ESS)
- Geweke diagnostic
- Posterior predictive checks (PPC)

---

## Results

The model captures:
- station-level heterogeneity
- systematic temporal variation
- improved fit after log transformation

Posterior distributions show:
- stable convergence
- reasonable mixing

---
