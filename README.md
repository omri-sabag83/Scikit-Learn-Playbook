# Scikit-Learn Playbook

A module-by-module learning playbook for using scikit-learn in analyst work:
**driver analysis, honest model evaluation, and risk scoring**. Six numbered
notebooks, each building on the one before it, ending in a calibrated, evaluated
risk score.

The program, progress table and per-module detail are in
[CURRICULUM.md](CURRICULUM.md).

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `01_modelling_workflow.ipynb` | Splits, pipelines, cross-validation, data leakage |
| 2 | `02_regression_driver_analysis.ipynb` | Linear and logistic regression; statsmodels for inference |
| 3 | `03_tree_permutation_importance.ipynb` | Random Forest, interpreted with permutation importance |
| 4 | `04_imbalanced_metrics.ipynb` | Precision, recall, ROC-AUC vs. PR-AUC |
| 5 | `05_calibration.ipynb` | Reliability diagrams, Brier score, calibration |
| 6 | `06_risk_score.ipynb` | Points scorecard, risk bands, trust checks |

## Datasets

Both are from the UCI Machine Learning Repository, licensed **CC BY 4.0**
(Creative Commons Attribution 4.0), and committed in `data/raw/`:

- **Default of Credit Card Clients**: Yeh, I.-C., & Lien, C.-H. (2009).
  *The comparisons of data mining techniques for the predictive accuracy of
  probability of default of credit card clients.* Expert Systems with
  Applications. [UCI page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
  Converted from the original `.xls` to CSV; content unchanged.
- **Bank Marketing**: Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.* Proceedings of ESM 2011 (the citation the dataset requests).
  [UCI page](https://archive.ics.uci.edu/dataset/222/bank+marketing).
  `bank-full.csv` plus the dataset's two documentation files.

## Reproduce

```bash
conda activate base                 # tested env: Python 3.13.5, versions in requirements.txt
# elsewhere: conda create -n sklearn_playbook python=3.13 && pip install -r requirements.txt

python data/get_data.py             # optional: rebuild data/raw/ from UCI (checksummed)
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

All randomness is seeded, so re-running gives identical outputs. `common.py`
holds the shared loaders and the locked train/validation/test split.

## Process

A self-directed learning program: I set the objectives, scope and review
standard; Claude served as tutor and pair-programmer, building each module
for my review. Every modelling claim went through a structured rigor
checklist, and the final risk-score result was checked by an independent
verifier that saw only the claim and the data.
