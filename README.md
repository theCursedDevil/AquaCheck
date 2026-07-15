# AquaCheck

A statistical and machine-learning analysis of the `water_potability.csv` dataset (3,276 samples, 9 physicochemical indicators), built to help water-safety decision-makers understand what actually predicts potability, and screen samples for follow-up testing.

## Contents

| File / folder | What it is |
|---|---|
| `AquaCheck_Risk_Analysis_Report.docx` | Full report: EDA, cleaning, linear-baseline diagnostics (VIF, coefficients, likelihood-ratio test, Hosmer-Lemeshow), nonlinear model comparison and tuning, calibration, SHAP interpretability, limitations, recommendations — all charts below are embedded in it |
| `water_potability_clean.csv` | Cleaned dataset (missing values imputed by within-class median; 0 missing values remain) |
| `model_summary.json` | Every statistic computed (baseline logistic regression numbers, all 4 candidate models' AUC, calibration Brier scores, SHAP top features, tuned hyperparameters for each model) in raw machine-readable form |
| `eda_charts/` | All 11 chart images (PNG) used in the report, full resolution, for reuse in slides or other documents |
| `analysis_scripts/` | The 4 Python scripts that produced every number and chart, in pipeline order — run them yourself to reproduce or extend the analysis |
| `requirements.txt` | Exact pinned package versions needed to run the scripts in `analysis_scripts/` |

### `eda_charts/` contents
- `viz_class_balance.png` — target class distribution
- `viz_missingness.png` — missing values by feature
- `viz_boxplots.png` — outlier check across all 9 indicators
- `viz_distributions.png` — feature distributions split by potability class
- `viz_correlation.png` — full correlation matrix
- `viz_model_comparison.png` — test AUC for all 4 candidate models vs. the logistic regression baseline
- `viz_roc_final.png` — ROC curve, final ensemble model, AUC 0.874
- `viz_confusion_final.png` — confusion matrix, final ensemble model
- `viz_calibration.png` — calibration curve, raw vs. isotonic-calibrated probabilities
- `viz_shap_summary.png` — SHAP summary plot (per-sample feature impact and direction)
- `viz_feature_importance.png` — mean |SHAP value| per feature (overall importance)

### `analysis_scripts/` contents (run in this order)
1. `step1_eda.py` — initial inspection: shape, missingness, skew, outlier scan
2. `step2_clean_viz.py` — within-class median imputation + core EDA charts
3. `step3_baseline.py` — logistic regression diagnostic baseline (VIF → fit → CI → likelihood-ratio test → Hosmer-Lemeshow → ROC) — establishes whether a linear relationship exists before reaching for anything more complex
4. `step4_models.py` — tunes Random Forest, XGBoost, and LightGBM via cross-validated randomized search; builds a soft-voting ensemble; selects the best model by test AUC; calibrates its probabilities; computes SHAP values; saves the final model and all remaining charts

Each script writes `model_summary.json` incrementally (step3 creates it, step4 extends it), so run them in order.

## Users, inputs, outputs

- **Users**: water utility staff, regulators, NGO/field teams, and researchers who need a fast, defensible read on sample risk plus the statistical backing to justify it.
- **Inputs**: 9 measurable water-quality indicators (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity).
- **Outputs**: a rigorous written analysis of what does and doesn't predict potability, plus a saved, calibrated model that can be loaded and queried directly for new samples.
- **Success criteria**: the analysis is statistically sound (correct test for a binary outcome, diagnostics run and interpreted, not just reported), and the final model is genuinely the best available for this problem (multiple candidates properly tuned and compared, not just one).

## Project flow followed

1. **Understand the problem** — binary outcome (Potable 0/1); the real question is which model class is even appropriate, not just which model fits "best" by one metric.
2. **Clean and inspect data** — checked missingness pattern (found ~MAR, not tied to the outcome), imputed within-class median, checked outliers (all under ~3%, left as-is since tree/boosting models are robust to them).
3. **EDA and visual analysis** — distributions by class, correlation matrix (every feature under 0.04 linear correlation with the outcome), regulatory reference-range check.
4. **Test for a linear relationship first** — outcome is 0/1, so OLS is invalid; logistic regression (the correct linear analogue) was fit and rigorously evaluated: VIF, coefficient significance, likelihood-ratio test, McFadden's pseudo-R², Hosmer-Lemeshow. **All four converge on the same conclusion: no linear signal exists** (LR test p = 0.87, test AUC 0.537).
5. **Move to models that can capture nonlinear interactions** — three tree-based ensembles (Random Forest, XGBoost, LightGBM) were each tuned via cross-validated randomized hyperparameter search, then combined into a soft-voting ensemble. All four comfortably beat the linear baseline; the ensemble wins on both test AUC and cross-validation AUC.
6. **Calibrate and explain the final model** — isotonic calibration improves probability trustworthiness (Brier score 0.139 → 0.134) without hurting discrimination (AUC unchanged); SHAP analysis identifies Sulfate and pH as by far the strongest individual drivers, with Hardness a close third among a cluster of secondary features.
7. **Evaluate and document limitations honestly** — noted the train/cross-validation AUC consistency (0.874 test vs. 0.871 ± 0.013 CV — a good sign, not a red flag), the absence of microbiological data, and the need for local validation before operational use.
8. **Save and document the final model** — the calibrated ensemble is saved to disk so it can be loaded and queried directly, without rerunning the full pipeline (see "Reproducing the analysis" below).

## Key numbers at a glance

| Model | Test AUC | 5-fold CV AUC | Notes |
|---|---|---|---|
| Logistic regression (all 9 features) | 0.537 | — | Diagnostic baseline only. Likelihood-ratio test p = 0.87 — no linear signal. Never predicts the minority class at the default threshold. |
| Random Forest (tuned) | 0.865 | — | |
| XGBoost (tuned) | 0.873 | — | Strongest individual model |
| LightGBM (tuned) | 0.870 | — | |
| **Ensemble (RF + XGBoost + LightGBM)** | **0.874** | **0.871 ± 0.013** | **Final model** — soft-voting average of the three tuned models |

Full statistical detail, all diagnostic tests, and every figure referenced above are in the Word report.

## Limitations, in brief

- The final model tops out around AUC 0.87 — solid discrimination, but this dataset is physicochemical-only with no microbiological data, so there's a real ceiling on how far any model can go here.
- The model has no visibility into microbiological contamination (bacteria, viruses, parasites), which is a separate and equally critical safety dimension. A "likely potable" prediction says nothing about pathogen presence.
- The model has not been locally validated against a specific region's water sources or regulatory standards — do that before any operational deployment.

Full discussion of these and other limitations is in Section 6 of the Word report.

## Reproducing the analysis

```bash
pip install -r requirements.txt
cd analysis_scripts
python3 step1_eda.py
python3 step2_clean_viz.py
python3 step3_baseline.py
python3 step4_models.py
```

`requirements.txt` pins the exact package versions this analysis was built and verified against. Runtime is roughly 1–2 minutes on a single core; `step4_models.py` does the bulk of the work (three cross-validated hyperparameter searches plus SHAP computation).

`step4_models.py` also saves `final_model_raw.joblib` and `final_model_calibrated.joblib` (the fitted ensemble, before and after probability calibration) if you want to load the model directly in Python rather than rerunning the pipeline:

```python
import joblib
import pandas as pd

model = joblib.load('final_model_calibrated.joblib')
sample = pd.DataFrame([{
    'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
    'Sulfate': 330, 'Conductivity': 420, 'Organic_carbon': 14,
    'Trihalomethanes': 66, 'Turbidity': 4,
}])
probability_potable = model.predict_proba(sample)[:, 1][0]
```
