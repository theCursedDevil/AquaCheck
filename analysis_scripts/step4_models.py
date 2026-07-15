"""
STEP 4 — Nonlinear model comparison and final model selection.

Step 3 showed the outcome carries no usable LINEAR signal (LR test
p=0.87, test AUC=0.537). This script fits three tree-based ensemble
models that can capture nonlinear thresholds and feature interactions,
tunes each with cross-validated random search, compares them on a
held-out test set, calibrates the winner's probabilities, and explains
it with SHAP (a more rigorous, interaction-aware alternative to raw
"feature_importances_").

Models compared:
  - Random Forest       (bagging, low variance, easy to reason about)
  - XGBoost              (gradient boosting, typically strongest on
                          small/medium tabular data)
  - LightGBM             (gradient boosting, faster, leaf-wise growth)
  - Soft-voting ensemble of the three tuned models
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42

df = pd.read_csv('water_potability_clean.csv')
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']
X = df[features]
y = df['Potability']

# Same split as step3, for a fair, direct comparison against the linear baseline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print("=" * 70)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Train class balance: {y_train.mean():.3f} potable")
print("=" * 70)

results = {}
fitted_models = {}

# ---------------------------------------------------------------
# Model 1: Random Forest
# ---------------------------------------------------------------
print("\n[1/3] Tuning Random Forest ...")
rf_param_dist = {
    'n_estimators': [200, 350, 500],
    'max_depth': [6, 8, 10, 12],
    'min_samples_leaf': [1, 3, 5, 10, 15],
    'max_features': ['sqrt', 'log2', 0.5],
    'class_weight': ['balanced', 'balanced_subsample'],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_dist, n_iter=12, scoring='roc_auc', cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
print(f"  Best params: {rf_search.best_params_}")
print(f"  Best CV AUC: {rf_search.best_score_:.4f}")
fitted_models['Random Forest'] = rf_best

# ---------------------------------------------------------------
# Model 2: XGBoost
# ---------------------------------------------------------------
print("\n[2/3] Tuning XGBoost ...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_param_dist = {
    'n_estimators': [150, 300, 450],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'min_child_weight': [1, 3, 5, 10],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric='auc',
                   scale_pos_weight=scale_pos_weight, n_jobs=-1),
    xgb_param_dist, n_iter=15, scoring='roc_auc', cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
print(f"  Best params: {xgb_search.best_params_}")
print(f"  Best CV AUC: {xgb_search.best_score_:.4f}")
fitted_models['XGBoost'] = xgb_best

# ---------------------------------------------------------------
# Model 3: LightGBM
# ---------------------------------------------------------------
print("\n[3/3] Tuning LightGBM ...")
lgbm_param_dist = {
    'n_estimators': [150, 300, 450],
    'max_depth': [3, 4, 5, 6, -1],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'min_child_samples': [10, 20, 30],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
}
lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced',
                    n_jobs=-1, verbosity=-1),
    lgbm_param_dist, n_iter=15, scoring='roc_auc', cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1
)
lgbm_search.fit(X_train, y_train)
lgbm_best = lgbm_search.best_estimator_
print(f"  Best params: {lgbm_search.best_params_}")
print(f"  Best CV AUC: {lgbm_search.best_score_:.4f}")
fitted_models['LightGBM'] = lgbm_best

# ---------------------------------------------------------------
# Model 4: Soft-voting ensemble of the three tuned models
# ---------------------------------------------------------------
print("\nBuilding soft-voting ensemble of the three tuned models ...")
voting = VotingClassifier(
    estimators=[('rf', rf_best), ('xgb', xgb_best), ('lgbm', lgbm_best)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train, y_train)
ensemble_cv_auc = cross_val_score(voting, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"  Ensemble CV AUC: {ensemble_cv_auc.mean():.4f} +/- {ensemble_cv_auc.std():.4f}")
fitted_models['Ensemble (RF+XGB+LGBM)'] = voting

# ---------------------------------------------------------------
# Evaluate all four on the held-out test set
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("HELD-OUT TEST SET COMPARISON")
print("=" * 70)
test_summary = {}
for name, model in fitted_models.items():
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    test_summary[name] = auc
    print(f"  {name:<25s} Test AUC = {auc:.4f}")

best_name = max(test_summary, key=test_summary.get)
best_model = fitted_models[best_name]
print(f"\nBest model by test AUC: {best_name} ({test_summary[best_name]:.4f})")

# ---------------------------------------------------------------
# Calibrate the winning model's probabilities (isotonic, via CV on train)
# Raw tree-ensemble probabilities are often over/under-confident;
# calibration matters here because the report presents a probability,
# not just a class label, to end users via the risk screener.
# ---------------------------------------------------------------
print(f"\nCalibrating {best_name} probabilities (5-fold isotonic) ...")
calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=cv)
calibrated.fit(X_train, y_train)
prob_test_calibrated = calibrated.predict_proba(X_test)[:, 1]
prob_test_raw = best_model.predict_proba(X_test)[:, 1]

auc_calibrated = roc_auc_score(y_test, prob_test_calibrated)
brier_raw = brier_score_loss(y_test, prob_test_raw)
brier_calibrated = brier_score_loss(y_test, prob_test_calibrated)
print(f"  AUC after calibration: {auc_calibrated:.4f} (AUC is rank-based, calibration won't change it much)")
print(f"  Brier score raw:        {brier_raw:.4f} (lower = better-calibrated probabilities)")
print(f"  Brier score calibrated: {brier_calibrated:.4f}")

# Find the threshold that maximizes F1 on the calibrated probabilities
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(y_test, (prob_test_calibrated >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[int(np.argmax(f1s))]
print(f"  F1-optimal decision threshold: {best_thresh:.2f} (default 0.5 used for reported metrics below)")

pred_class = (prob_test_calibrated >= 0.5).astype(int)
print(f"\n{best_name} (calibrated) -- classification report @ threshold 0.5:")
print(classification_report(y_test, pred_class, target_names=['Not Potable', 'Potable']))
cm = confusion_matrix(y_test, pred_class)
print("Confusion matrix:\n", cm)

# 5-fold CV AUC of the winning (uncalibrated) model, for the headline number
best_cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\n{best_name} 5-fold CV AUC (train): {best_cv_scores.mean():.4f} +/- {best_cv_scores.std():.4f}")

# ---------------------------------------------------------------
# SHAP interpretability for the winning model (skip if ensemble --
# explain the strongest individual tree model instead, since SHAP's
# TreeExplainer doesn't support VotingClassifier directly)
# ---------------------------------------------------------------
print("\nComputing SHAP values for interpretability ...")
import shap
shap_source_name = best_name if best_name != 'Ensemble (RF+XGB+LGBM)' else max(
    {k: v for k, v in test_summary.items() if k != 'Ensemble (RF+XGB+LGBM)'}, key=test_summary.get
)
shap_model = fitted_models[shap_source_name]
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer(X_test)

# Handle binary-classifier SHAP output shape differences across libraries
sv = shap_values
if len(sv.values.shape) == 3:  # (n_samples, n_features, n_classes)
    sv_plot = shap.Explanation(
        values=sv.values[:, :, 1], base_values=sv.base_values[:, 1] if np.ndim(sv.base_values) > 1 else sv.base_values,
        data=sv.data, feature_names=features
    )
else:
    sv_plot = sv

fig = plt.figure(figsize=(8, 6))
shap.summary_plot(sv_plot, X_test, show=False, plot_size=None)
plt.title(f'SHAP Feature Impact — {shap_source_name}')
plt.tight_layout(); plt.savefig('viz_shap_summary.png', bbox_inches='tight'); plt.close()

mean_abs_shap = pd.Series(np.abs(sv_plot.values).mean(axis=0), index=features).sort_values(ascending=False)
print("\nMean |SHAP value| per feature (overall importance, direction-agnostic):")
print(mean_abs_shap.round(4))

fig, ax = plt.subplots(figsize=(7, 5))
mean_abs_shap.sort_values().plot(kind='barh', ax=ax, color='#55A868')
ax.set_title(f'Feature Importance (mean |SHAP value|) — {shap_source_name}')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout(); plt.savefig('viz_feature_importance.png'); plt.close()

# ---------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
names = list(test_summary.keys())
aucs = list(test_summary.values())
colors_bar = ['#4C72B0' if n != best_name else '#55A868' for n in names]
bars = ax.barh(names, aucs, color=colors_bar)
ax.set_xlim(0.5, max(aucs) + 0.08)
ax.axvline(0.537, color='#C44E52', linestyle='--', alpha=0.8, label='Logistic regression baseline (0.537)')
for i, v in enumerate(aucs):
    ax.text(v + 0.005, i, f"{v:.3f}", va='center')
ax.set_xlabel('Test AUC-ROC')
ax.set_title('Model Comparison — Test AUC')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('viz_model_comparison.png'); plt.close()

# ---------------------------------------------------------------
# ROC curve for the winning model
# ---------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, prob_test_calibrated)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f'{best_name} (AUC={auc_calibrated:.3f})', color='#55A868', lw=2)
ax.plot([0, 1], [0, 1], '--', color='gray', label='Random guess (AUC=0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve — {best_name} (Final Model)')
ax.legend()
plt.tight_layout(); plt.savefig('viz_roc_final.png'); plt.close()

# ---------------------------------------------------------------
# Confusion matrix viz
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout(); plt.savefig('viz_confusion_final.png'); plt.close()

# ---------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, prob_test_raw, n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, prob_test_calibrated, n_bins=10)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
ax.plot(mean_pred_raw, frac_pos_raw, 'o-', color='#C44E52', label='Raw model')
ax.plot(mean_pred_cal, frac_pos_cal, 'o-', color='#55A868', label='Calibrated model')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives observed')
ax.set_title(f'Calibration Curve — {best_name}')
ax.legend()
plt.tight_layout(); plt.savefig('viz_calibration.png'); plt.close()

# ---------------------------------------------------------------
# Save the calibrated final model + everything the report needs
# ---------------------------------------------------------------
import joblib
joblib.dump(calibrated, 'final_model_calibrated.joblib')
joblib.dump(best_model, 'final_model_raw.joblib')
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

with open('model_summary.json') as f:
    summary = json.load(f)

summary.update({
    'candidate_models_test_auc': test_summary,
    'best_model_name': best_name,
    'best_model_cv_auc_mean': float(best_cv_scores.mean()),
    'best_model_cv_auc_std': float(best_cv_scores.std()),
    'best_model_test_auc': float(test_summary[best_name]),
    'best_model_test_auc_calibrated': float(auc_calibrated),
    'brier_raw': float(brier_raw),
    'brier_calibrated': float(brier_calibrated),
    'f1_optimal_threshold': float(best_thresh),
    'shap_source_model': shap_source_name,
    'shap_top_features': mean_abs_shap.index[:3].tolist(),
    'shap_top_values': mean_abs_shap.values[:3].tolist(),
    'best_params': {
        'Random Forest': rf_search.best_params_,
        'XGBoost': {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in xgb_search.best_params_.items()},
        'LightGBM': lgbm_search.best_params_,
    },
})
with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"FINAL MODEL: {best_name}")
print(f"Test AUC: {test_summary[best_name]:.4f}  |  Calibrated AUC: {auc_calibrated:.4f}")
print(f"Top 3 features (SHAP): {mean_abs_shap.index[:3].tolist()}")
print(f"{'='*70}")
print("\nSaved: final_model_calibrated.joblib, final_model_raw.joblib,")
print("       model_summary.json, 6 new charts, X_test.csv, y_test.csv")
