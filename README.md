# Bayesian Hierarchical Modeling of Air Pollution

## Overview

This project implements a Bayesian hierarchical model (BHM) to analyze daily PM2.5 concentrations across multiple monitoring stations in Italy. The response variable is log-transformed to reduce skewness, and station-specific random effects are used to capture persistent heterogeneity across monitoring sites.

The focus is on multiparameter Bayesian inference with hyperpriors, estimated through a custom Metropolis-within-Gibbs MCMC sampler.

The objectives are to:

- model temporal patterns in PM2.5 concentrations using month and day-of-week effects;
- capture station-level heterogeneity through hierarchical random effects;
- assess posterior behavior, convergence, and model adequacy using MCMC diagnostics and posterior predictive checks.

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
│ └── paper_pm25.pdf
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

---

## Diagnostics

Convergence and mixing of the MCMC sampler are assessed using both graphical and numerical diagnostics:

- **Trace plots** to verify stationarity of the chains after burn-in  
- **Autocorrelation functions (ACF)** to evaluate mixing efficiency  
- **Effective Sample Size (ESS)** to quantify the amount of independent information in the chains  
- **Geweke diagnostic** to assess equality of early and late segments of the chain  
- **Heidelberger–Welch diagnostic** to test stationarity and determine an appropriate burn-in period  
- **Posterior predictive checks (PPC)** to evaluate model adequacy in reproducing key features of the data  

Results indicate satisfactory convergence for all parameters. While some station-specific effects exhibit lower sampling efficiency, no evidence of pathological behavior or non-stationarity is detected.

---

## Results

The model successfully captures the main sources of variation in PM2.5 concentrations:

- **Station-level heterogeneity**, through random effects capturing persistent local differences  
- **Systematic temporal variation**, with a clear seasonal pattern and mild day-of-week effects  
- **Improved distributional fit**, achieved via log transformation of the outcome  

Posterior inference shows:

- **Stable convergence** across all parameter blocks  
- **Good mixing** for regression coefficients and variance components  
- **Heterogeneous but acceptable ESS**, with lower values concentrated among some station-level effects  
- **Strong posterior predictive performance**, with replicated data closely matching observed distributions and summary statistics  

Overall, the Bayesian hierarchical specification provides a coherent and flexible framework for modeling air pollution dynamics in the presence of unbalanced panel data.

---
