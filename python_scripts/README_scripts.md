# Python Scripts

This directory contains all Python scripts used for data preparation, model training, analyses, and visualization. The workflow is designed to ensure full reproducibility while allowing safe local testing using synthetic data before running computations on MONA.

---

## Folder Structure

```
python_scripts/
│
├── local_scripts/                 # Local testing with synthetic data
│   ├── data_summary.py
│   ├── feature_importance.py
│   ├── rq1_analyses.py
│   ├── rq2_analyses.py
│   └── rq3_analyses.py
│
├── mona_scripts/                  # Scripts executed on MONA
│   ├── __pycache__/
│   ├── config.py                   # Shared configuration for MONA/local
│   ├── data_loader.py              # Shared data loader for MONA/local
│   ├── mona_data_summary.py        
│   ├── mona_feature_importance.py
│   ├── mona_rq1.py
│   ├── mona_rq2.py
│   ├── mona_rq3.py
│   ├── sampler.py                  # Shared sampler logic
│   ├── trainer.py                  # Shared training routines
│   └── utils.py                    # Shared helper functions
│
└── visualisation_scripts/         # Visualization scripts (post‑MONA)
    ├── rq1.py
    ├── rq2.py
    └── rq3.py
```

---

## Workflow Overview

1. **Generate synthetic data**  
   - Use the `synthetic_data_generator` in the `data/` folder.  
   - This data allows you to test code safely without exposing real data.

2. **Test locally**  
   - Use scripts inside `local_scripts/` to ensure code runs correctly on synthetic data.  
   - Local scripts mirror the logic of MONA scripts as closely as possible.

3. **Prepare MONA scripts**  
   - After confirming correctness locally, adapt the corresponding `mona_*` scripts.  
   - These are designed specifically for MONA’s environment.

4. **Run jobs on MONA**  
   - Upload the updated MONA scripts and execute analyses.  
   - Download outputs from MONA into the `outputs_mona/` directory.

5. **Visualize results**  
   - Use scripts in `visualisation_scripts/` to produce figures and summaries based on MONA outputs.

---

## Notes

- `config.py`, `data_loader.py`, `sampler.py`, `trainer.py`, and `utils.py` serve as shared components across both local and MONA workflows.
- The scripts evolve over time as analysis needs change.
- Local and MONA versions should remain aligned to ensure smooth transitions.
- Scripts are generated with the help of large language models (ChatGPT and Claude), guided through prompts and task descriptions.
