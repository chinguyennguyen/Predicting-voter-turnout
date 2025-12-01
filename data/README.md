# Data Directory

This directory contains supporting files for running the project locally, including:
- A **synthetic dataset generator** that creates data with the same schema as the MONA dataset (40,000 observations).
- A **municipality code–name mapping** used in visualization scripts.
- A **shapefile** containing Sweden’s municipal boundaries (used in Research Question 2 on geographic predictability).

**Important:** The synthetic data is purely artificial. Variables are generated randomly and are used only to verify that scripts run end-to-end. No meaningful analysis can be conducted using this synthetic dataset.

---

## Structure

```
data/
├── alla_kommuner/              # Sweden municipality shapefile (for RQ2)
│   ├── alla_kommuner.shp
│   ├── alla_kommuner.dbf
│   ├── alla_kommuner.prj
│   ├── alla_kommuner.shx
│   ├── alla_kommuner.sbn
│   ├── alla_kommuner.sbx
│   └── alla_kommuner.shp.xml
│
├── municipality_code.csv       # Municipality code ↔ name mapping
├── synthetic_data_generator.py # Script to generate synthetic data
├── synthetic_data.dta          # Example synthetic dataset (safe to commit)
└── README.md
```

---

## Data Sources (Real Data, Not Included Here)

The actual research uses confidential Swedish administrative data accessed through the MONA secure environment:

- **Voting Register** – Official turnout records for the 2018 and 2022 municipal elections  
- **Administrative Registers**, including:
  - Demographics  
  - Socioeconomic indicators  
  - Household structure  
  - Geographic identifiers  

Sample: Individuals appearing in the population register with complete demographic and socioeconomic information in 2018. Current models use only **baseline 2018 characteristics** to predict turnout transitions.

---

## Data Protection

**IMPORTANT: No sensitive or administrative data is stored in this repository.**

All real data remains exclusively inside MONA due to:
- GDPR regulations  
- Statistics Sweden (SCB) access restrictions  
- Research ethics and confidentiality requirements  

Only **synthetic data**, **shapefiles**, and **code** are stored here.

---

## Usage

To test the code locally:

1. Run `synthetic_data_generator.py`
2. It will create a synthetic dataset with the correct schema
3. Use this dataset with scripts in:

```
python_scripts/local_scripts/
```

---

## Data Dictionary (Synthetic & Real Variables)

### Target Variable
- **`y_mun`** — Voting transition category (VV, VN, NV, NN)

### Numerical Variables
- `fgang18` — First-time voter in 2018  
- `female` — Indicator for being female  
- `age_2018` — Age in 2018  
- `foreigner` — Indicator for having foreign-born parents  
- `schooling_years` — Years of schooling  
- `share_sa` — Share of social assistance in income  
- `share_labor_income` — Share of labor income in total income  
- `total_income` — Total annual income  
- `barn0_6`, `barn7_17`, `barn_above18` — Children in household by age group  

### Categorical Variables
- `birth_continent` — Continent category  
- `employment_status` — Employment category  
- `sector` — Employment sector  
- `marital_status` — Marital category  
- `Kommun` — Municipality (290 total)
