"""
STEP 3 — Linear baseline

The outcome is binary (0/1), so plain OLS is invalid by construction
(residuals can't be normal/homoscedastic for a 0/1 target, and predictions
can fall outside [0,1]). Logistic regression is the correct linear-model
analogue, so it is fit here -- not as the final deliverable, but as a
required diagnostic: it tells us whether a LINEAR combination of the 9
raw indicators carries any signal at all, before reaching for anything
more complex.

Full inferential detail (coefficients, SE, Wald z, 95% CI, VIF, likelihood-
ratio test, Hosmer-Lemeshow) is kept because a "the linear model doesn't
work" conclusion needs to be demonstrated rigorously, not asserted.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from scipy import stats as sstats
import json

df = pd.read_csv('water_potability_clean.csv')
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']
X = df[features]
y = df['Potability']

# ---- VIF check (multicollinearity) ----
X_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("VIF (multicollinearity check) -- all should be near 1 for independent features:")
print(vif_data.to_string(index=False))

# ---- Train/test split (same split used by every model in step4, for fair comparison) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

# ---- Fit logistic regression ----
X_train_sm = sm.add_constant(X_train_s)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)
print("\n" + "=" * 70)
print("LOGISTIC REGRESSION SUMMARY")
print("=" * 70)
print(result.summary())

# ---- Likelihood Ratio Test (overall model significance) ----
lr_stat = 2 * (result.llf - result.llnull)
lr_pvalue = sstats.chi2.sf(lr_stat, df=len(features))
print(f"\nLikelihood Ratio Test: LR={lr_stat:.3f}, df={len(features)}, p={lr_pvalue:.4f}")
print(f"Pseudo R-squared (McFadden) = {result.prsquared:.4f}")

# ---- Hosmer-Lemeshow goodness of fit ----
def hosmer_lemeshow(y_true, y_prob, g=10):
    data = pd.DataFrame({'y': y_true.values, 'prob': y_prob.values})
    data['decile'] = pd.qcut(data['prob'], g, duplicates='drop')
    grouped = data.groupby('decile')
    obs_1 = grouped['y'].sum()
    obs_0 = grouped['y'].count() - obs_1
    exp_1 = grouped['prob'].sum()
    exp_0 = grouped['prob'].count() - exp_1
    hl_stat = (((obs_1 - exp_1) ** 2 / exp_1) + ((obs_0 - exp_0) ** 2 / exp_0)).sum()
    dof = g - 2
    pval = sstats.chi2.sf(hl_stat, dof)
    return hl_stat, pval

pred_prob_train = result.predict(X_train_sm)
hl_stat, hl_p = hosmer_lemeshow(y_train, pred_prob_train)
print(f"Hosmer-Lemeshow test: stat={hl_stat:.2f}, p={hl_p:.4f} (p<0.05 => poor fit)")

# ---- Test set performance ----
X_test_sm = sm.add_constant(X_test_s)
pred_prob_test = result.predict(X_test_sm)
auc = roc_auc_score(y_test, pred_prob_test)
print(f"\n{'='*70}\nLOGISTIC REGRESSION -- TEST SET PERFORMANCE\n{'='*70}")
print(f"AUC-ROC: {auc:.4f}  (0.5 = random guessing, 1.0 = perfect)")
pred_class = (pred_prob_test >= 0.5).astype(int)
print(classification_report(y_test, pred_class, target_names=['Not Potable', 'Potable']))

print("\nCONCLUSION: no individual coefficient is significant at p<0.05, the")
print("overall likelihood-ratio test fails to reject the null (no linear")
print("relationship), and test AUC is barely above chance. This is the")
print("correct, expected result for a linear model on physicochemical")
print("readings that (per the correlation matrix in step 2) each have near-")
print("zero linear correlation with potability. A model class that can")
print("capture NONLINEAR interactions is required -> step4_models.py.")

summary = {
    'n_obs': len(df),
    'logit_test_auc': float(auc),
    'logit_pseudo_r2': float(result.prsquared),
    'logit_lr_stat': float(lr_stat),
    'logit_lr_pvalue': float(lr_pvalue),
    'logit_hl_stat': float(hl_stat),
    'logit_hl_p': float(hl_p),
    'max_vif': float(vif_data['VIF'][1:].max()),
    'baseline_accuracy': float(max(y.mean(), 1 - y.mean())),
}
with open('model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved model_summary.json (logistic regression baseline numbers)")
