# Predicting Voter Turnout Transitions in Sweden

**Using machine learning to understand why people change their voting habits**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)]()
[![Stata](https://img.shields.io/badge/Stata-Data%20Cleaning-1a5490)]()

## Overview

This project tackles a fundamental question in democratic participation: **What makes some people's voting behavior predictable while others remain unpredictable?** 

Using Sweden's complete population register (7.8 million voters), I predict voting behavior transitions between the 2018 and 2022 municipal elections. Rather than asking "who votes?" in a single election, this research examines **behavioral change**—who starts voting, who stops, and what makes habits stable or fluid.

**Research Questions:**
1. Where is voting behavior most predictable? (Geographic performance analysis)
2. Does more training data improve prediction of rare behavioral transitions?
3. What demographic factors predict stability vs. change in voting habits?

## Contributions

First population-level machine learning study of voting transitions using complete administrative data. Most studies examine single elections; this captures behavioral dynamics across time.

**Methodology** Real-world demonstration of:
- Handling extreme class imbalance (3% minority class)
- Large-scale data processing (7.8M observations)
- Detecting and analyzing geographic model bias

## Key Findings

### Performance Patterns
- **Stable behavior is predictable:** 94% precision for consistent voters (VV)
- **Transitions are hard:** Only 5.8% precision for new voters (NV), 12.6% for stopped voters (VN)
- **Critical insight:** Demographics predict *who has habits*, not *who changes their habits*

### Geographic Bias Discovered
- Model F1 scores vary dramatically: 0.50-0.75 across Sweden's 290 municipalities
- Northern rural areas: harder to predict (smaller samples, different patterns)
- Urban southern Sweden: more predictable (larger samples, urban voting dynamics)
- **Implication:** Model performance reflects real regional differences + data availability issues

### Training Scale Analysis
- Performance plateaus around 500K-1M training samples
- Diminishing returns beyond 1M observations
- **Practical impact:** Don't need the full 7.8M dataset for stable performance

### What Drives Predictions?
SHAP analysis reveals:
- **Foreign birth status:** 56% of model importance
- **Age and income:** Secondary predictors
- **Education and employment:** Moderate effects
- **Geographic location:** Captures local political culture

**The puzzle:** These static features predict 2018 behavior well, but fail to predict *changes* by 2022. This suggests behavioral transitions require temporal variables (income changes, life events, residential moves) not captured in baseline demographics.

## Methodology

### The Prediction Challenge

I predict four voting patterns:
- **VV (81%)**: Voted both years → stable voters
- **NN (8.4%)**: Voted neither year → stable non-voters  
- **VN (7%)**: Voted 2018, not 2022 → stopped voting
- **NV (3.1%)**: Didn't vote 2018, voted 2022 → started voting

### Solution: Balanced Sampling Strategy

**Training approach:**
- Balanced sampling: equal observations per class
- Ensures model learns from all patterns, not just majority class
- Multiple random seeds for robustness

**Testing approach:**
- Natural class distribution (81%-8%-7%-3%)
- Reflects real-world prediction scenarios
- Per-class metrics (precision, recall, F1) instead of just accuracy

### Model Choice: XGBoost

- Native categorical feature support
- Handles imbalanced data effectively with proper sampling
- Fast training on large datasets
- Interpretable via SHAP analysis

### Three Core Experiments

**1. Geographic Analysis**
- Municipality-level performance mapping (290 municipalities)
- Choropleth visualizations revealing regional patterns
- Investigation of urban-rural performance gaps

**2. Training Scale Study**
- Tested training sizes: 100K, 250K, 500K, 1M, 2M, 4M samples
- Fixed test set for fair comparison
- Power law regression to model scaling effects

**3. Feature Importance**
- Permutation importance for overall rankings
- SHAP values for individual prediction explanations
- Analysis of what makes transitions unpredictable

## Technical Implementation

### Workflow Architecture

This project uses a **two-stage pipeline** that leverages the strengths of different tools:

**Stage 1: Data Preparation (Stata)**
- Raw data processing from Statistics Sweden's administrative registers
- Variable construction and feature engineering
- Data quality checks and validation
- Handling missing values and outliers
- Export to CSV format for machine learning

**Stage 2: Machine Learning (Python)**
- Model training and evaluation
- Feature importance analysis
- Geographic performance mapping
- Statistical analysis of results

This separation allows:
- Using Stata's strengths for administrative data handling
- Using Python's ML ecosystem for modeling
- Clear handoff point between data prep and analysis
- Reproducible workflow across both platforms

### Python Project Structure
```
├── data/
│   ├── sample_data.csv        # Synthetic data (demo only)
│   └── README.md              # Data documentation
├── src/
│   ├── data_loading.py        # Load cleaned CSV data
│   ├── models.py              # XGBoost model class
│   ├── evaluation.py          # Metrics, confusion matrices
│   ├── visualization.py       # Choropleth maps, SHAP plots
│   ├── config.py              # Hyperparameters, paths
│   └── pipeline.py            # End-to-end ML pipeline
├── notebooks/                 # Exploratory analysis
├── results/                   # Performance metrics & visualizations
├── tests/                     # Unit tests
└── requirements.txt           # Python dependencies
```

**Note:** Stata data cleaning scripts are not included in this repository.

### Machine Learning Engineering
**Key engineering practices:**
- Modular, production-ready architecture
- Type hints and comprehensive docstrings
- Logging for debugging and monitoring
- Configuration management (separate from code)
- Reproducible results (fixed random seeds)
- Version control for models and experiments

### Statistical Rigor
- Proper train/test splitting (no data leakage)
- Stratified sampling by geographic region
- Multiple evaluation metrics beyond accuracy
- Statistical testing for scaling effects
- Per-class performance analysis (not just aggregate)

### Data Pipeline
- Loads pre-cleaned CSV data from Stata
- Efficient processing of 7.8M observations
- Handles Swedish administrative data structure
- Categorical feature preparation for XGBoost
- Municipality-level aggregation for geographic analysis
- Memory-efficient sampling strategies

## Data Access & Privacy

**Important:** This project uses confidential Swedish administrative data accessed through Statistics Sweden's MONA secure remote access system.

### Privacy Measures
- Data never leaves secure MONA environment
- All data cleaning and ML analysis conducted on approved secure platform
- Only aggregated, non-identifiable results published
- Research approved by Swedish Ethical Review Authority
- Compliance with GDPR and Swedish data protection laws

### Synthetic Data Provided
The `data/sample_data.csv` file:
- Has identical column structure to cleaned real data
- Contains randomly generated values
- Allows Python code to run for demonstration
- **Produces meaningless results** (not representative)

Real analysis uses 7.8M population records with 29 demographic, economic, and geographic variables after Stata preprocessing.

### Data Processing Notes
**Original data format:** Stata (.dta) files from Swedish administrative registers

**Preprocessing (Stata):**
- Variable construction from raw registers
- Merging multiple data sources
- Quality checks and validation
- Missing value handling
- Export to CSV for ML pipeline

**ML Pipeline (Python):**
- Loads cleaned CSV data
- Feature encoding for XGBoost
- Model training and evaluation
- Results visualization and analysis

## Getting Started

### Prerequisites
```bash
# Python environment
Python 3.8+
pip install -r requirements.txt
```

### Run Demo with Synthetic Data
```bash
# Load pre-cleaned synthetic data and run ML pipeline
python src/pipeline.py --data data/sample_data.csv --output results/

# Note: Synthetic data produces random results for demonstration only
# Real analysis conducted on MONA system with population data
```

### Results Location
```
results/
├── model_performance/        # Confusion matrices, per-class metrics
├── geographic_analysis/      # Municipality-level performance maps
├── feature_importance/       # SHAP values, permutation importance
└── scaling_analysis/         # Training size experiments
```

## Future Directions

### Immediate Next Steps
Current models use only 2018 baseline demographics. To predict *transitions*, I need temporal change variables:

**Planned additions:**
- Income changes (2019-2022)
- Employment status transitions
- Residential moves (geographic mobility)
- Life events (marriage, divorce, children)
- Neighborhood changes (gentrification, demographic shifts)

**Hypothesis:** Behavioral transitions require capturing *change*, not just static characteristics.

### Model Improvements
- Address geographic sampling bias (northern municipalities)
- Ensemble methods combining XGBoost with other algorithms
- Deep learning for complex feature interactions

### Research Extensions
- Causal inference: Do specific interventions change voting?
- Subgroup analysis: Different predictors for different demographics?
- Policy implications: Where are voter mobilization efforts most effective?

## About This Research

This project is Chapter 3 of my PhD dissertation on political participation in Sweden.

**Author:** Chi Nguyen  
**Institution:** Economics Department, University of Gothenburg  
**Contact:** chi.nguyen@economics.gu.se  

**Status:** Early-stage dissertation research (2024-2025)

I'm actively developing this project and welcome discussions about:
- Methodology and technical approaches
- Extensions and applications to other contexts
- Collaboration opportunities
- Industry applications of similar techniques

If you find this research interesting or want to discuss the technical approach, please reach out!

**Note:** This is academic research using sensitive administrative data. All results are preliminary and subject to revision. The code demonstrates technical implementation; substantive findings are still under development.
