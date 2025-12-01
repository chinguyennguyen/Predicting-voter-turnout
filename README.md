# Predicting Voter Turnout Transitions in Sweden

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)]()
[![Stata](https://img.shields.io/badge/Stata-Data%20Cleaning-1a5490)]()

## Abstract

Using Sweden’s population register of 7.4 million eligible voters, this project predicts voting behavior in the 2018 and 2022 municipal elections. Instead of modeling turnout in a single year, the focus is on **transitions in voting habits** across time. The analysis trains models under different sampling strategies (stratified and balanced) and varying training sizes, using baseline covariates from 2018 such as education, family composition, and personal background.

The best-performing model (stratified sampling on 4 million observations) is then used to construct a municipality-level map of voting predictability. To understand why some municipalities are more predictable than others, two initial factors are examined: **entropy of the outcome distribution** and **test sample size**. Higher entropy reduces accuracy, while larger test sizes increase it (both significant at the 5% level). Next steps include analyzing heterogeneity in covariates, studying covariate–target correlations, and augmenting the model with temporal variables—income dynamics, life events, mobility, and shifts in local context not captured by baseline characteristics.

## Overview

This repository contains the code, analysis pipeline, and exported outputs for a population-level study of voting-behavior transitions in Sweden. All production models are trained inside the MONA secure environment at Statistics Sweden, while this public repository includes only non-sensitive components: scripts, configurations, visualizations, and summary results.

> **Important:** All production results are computed in Statistics Sweden’s secure MONA environment. Only aggregated outputs (tables, visuals) are exported.

---

## Voting Categories (2018 → 2022)

| Code | Description | Count (approx.) |
|------|-------------|-----------------|
| VV   | Voted both years | 5.7M |
| VN   | Voted in 2018 only | 570k |
| NV   | Voted in 2022 only | 318k |
| NN   | Did not vote either year | 680k |

The classes are highly imbalanced. This is a methodological challenge for the project.

---

## Research Questions

### **RQ1 — Methodological: Sampling Strategy & Scaling**
How different sampling strategies handle severe class imbalance at scale, and how predictive performance changes with training size.

Two settings are evaluated:

- **RQ1a: Method Comparison**  
  Training sizes: 50k–800k  
  Methods: Balanced vs stratified  
  Seeds: 3 per configuration

- **RQ1b: Stratified Scaling**  
  Training sizes: 1M–4M  
  Methods: Stratified only  
  Seeds: 3 per configuration

Balanced sampling is constrained by the smallest class (~210k observations), capping the feasible balanced dataset at ~840k.

---

### **RQ2 — Descriptive: Geographic Predictability**
How predictable voting-habit transitions are across Sweden’s 290 municipalities, and where geographic clustering appears.

Approach:
- Use the best-performing model from RQ1  
- Predict on the test set  
- Aggregate performance by municipality  
- Report metrics: balanced accuracy, per-class F1, precision, recall  

A choropleth map highlights spatial patterns in predictability.

---

### **RQ3 — Explanatory: What Drives Predictability?**
Which features of municipalities—such as outcome entropy, sample size, and covariate patterns—help explain variation in predictability.

Two dimensions are explored:

1. **Task Complexity**
   - Outcome diversity (entropy)
   - Sample size
   - Potential class imbalance

2. **Signal Strength**
   - Feature–outcome correlations  
   - Heterogeneity in covariates  
   - Demographic and economic variation  

Initial results:  
Higher entropy reduces accuracy; larger test samples improve it. Both effects are statistically significant.

---


### Repository Structure
```
Predicting-voter-turnout/
│
├── data/
│   ├── alla_kommuner.shp              # Sweden municipal shapefile (for RQ2)
│   ├── municipality_code.csv          # Matching municipal code to name
│   ├── README.md                      # README
│   └── synthetic_data_generator.py    # Generate 10k synthetic observations
│
├── python_scripts/
│   ├── local_scripts/                 # Local testing on synthetic data
│   │   ├── data_summary.py            # Generate dataset statistics
│   │   ├── feature_importance.py      # Feature importance and confusion matrix
│   │   ├── rq1_analyses.py            # RQ1 results analysis
│   │   ├── rq2_analyses.py            # RQ2 results analysis
│   │   └── rq3_analyses.py            # RQ3 results analysis
│   │
│   ├── mona_scripts/                  # Production scripts for MONA
│   │   ├── config.py                  # Configuration management
│   │   ├── data_loader.py             # Data loading & preprocessing
|   |   ├── mona_data_summary.py       # One-time data statistics
│   │   ├── mona_rq1.py                # RQ1 experiments (12-14 hours to run)
│   │   ├── mona_rq2.py                # RQ2 
│   │   ├── mona_rq3.py                # RQ3
│   │   ├── sampler.py                 # Sampling strategies
│   │   ├── trainer.py                 # XGBoost training
│   │   └── utils.py                   # Utility functions
│   │
│   └── visualisation_scripts/         # Visualization generation using real results from MONA
|       ├── rq1.py                     # RQ1 plots & figures
|       ├── r21.py                     # RQ2 plots & figures
│       └── rq3.py                     # RQ3 plots & figures
│
├── outputs_mona/                       # Results from MONA production runs
│   ├── tables/                         # CSV results
│   └── plots/                          # Visualizations
│
├── outputs_synthetic/                 # Results from local testing
│   ├── tables/                        # CSV results (test runs)
│   ├── plots/                         # Visualizations (test runs)
│   └── models/                        # Test models 
│
├── main_findings.ipynb                         # Jupyter analysis
│
├── .gitignore                         # Excludes MONA models, sensitive data
├── LICENSE
├── requirements.txt
└── README.md
```

Models trained in MONA cannot be exported; all results here are aggregated or synthetic.

---

## Key Findings

See `main_findings.ipynb` for full details.

---

## Future Directions

### **Model Extensions**
- Temporal features (income, employment, mobility)
- Local economic and demographic change
- Event-based features (family transitions, moves)

### **Explanatory Modeling**
A structured decomposition will assess:
- **Task complexity:** outcome entropy, heterogeneity, nonlinearities  
- **Signal strength:** feature sufficiency and alignment  

Both dimensions are needed to understand municipal-level variation.

---

## About the Research

This work forms a chapter of my PhD dissertation on political participation in Sweden.

**Author:** Chi Nguyen  
**Department:** Economics, University of Gothenburg  
**Email:** chi.nguyen@economics.gu.se  
**Status:** Early-stage research  

---

*This repository contains only non-sensitive code and exported aggregated outputs. All confidential data work is performed inside MONA.*
