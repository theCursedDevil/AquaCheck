"""
STEP 2 — Clean data and produce EDA visualizations.
Imputation strategy: within-class median (preserves the class-conditional
distribution instead of pulling missing values toward one global median).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

df = pd.read_csv('water_potability.csv')
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# ---- Cleaning: median imputation WITHIN each Potability class ----
df_clean = df.copy()
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    df_clean[col] = df_clean.groupby('Potability')[col].transform(lambda x: x.fillna(x.median()))

df_clean.to_csv('water_potability_clean.csv', index=False)
print("Missing values after cleaning:", df_clean.isnull().sum().sum())

# ---- Viz 1: class balance ----
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Potability'].value_counts().sort_index()
colors = ['#4C72B0', '#55A868']
ax.bar(['Not Potable (0)', 'Potable (1)'], counts.values, color=colors)
for i, v in enumerate(counts.values):
    ax.text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha='center')
ax.set_title('Target Class Distribution')
ax.set_ylabel('Count')
plt.tight_layout(); plt.savefig('viz_class_balance.png'); plt.close()

# ---- Viz 2: missingness ----
fig, ax = plt.subplots(figsize=(7, 4))
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values()
ax.barh(miss.index, miss.values, color='#C44E52')
for i, v in enumerate(miss.values):
    ax.text(v + 5, i, f"{v} ({v/len(df)*100:.1f}%)", va='center')
ax.set_title('Missing Values by Feature')
ax.set_xlabel('Count missing')
plt.tight_layout(); plt.savefig('viz_missingness.png'); plt.close()

# ---- Viz 3: distributions by class ----
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, col in zip(axes.flat, features):
    sns.kdeplot(data=df_clean, x=col, hue='Potability', ax=ax, fill=True, alpha=0.4,
                palette={0: '#4C72B0', 1: '#55A868'}, common_norm=False, legend=False)
    ax.set_title(col)
fig.suptitle('Feature Distributions by Potability Class', fontsize=15, y=1.01)
handles = [plt.Line2D([0], [0], color='#4C72B0', lw=6), plt.Line2D([0], [0], color='#55A868', lw=6)]
fig.legend(handles, ['Not Potable', 'Potable'], loc='upper right', bbox_to_anchor=(1.0, 1.02))
plt.tight_layout(); plt.savefig('viz_distributions.png', bbox_inches='tight'); plt.close()

# ---- Viz 4: boxplots for outliers ----
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, col in zip(axes.flat, features):
    sns.boxplot(data=df_clean, y=col, ax=ax, color='#8172B2')
    ax.set_title(col)
fig.suptitle('Outlier Check (Boxplots)', fontsize=15, y=1.01)
plt.tight_layout(); plt.savefig('viz_boxplots.png', bbox_inches='tight'); plt.close()

# ---- Viz 5: correlation heatmap ----
fig, ax = plt.subplots(figsize=(9, 7))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            square=True, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix (incl. Potability)')
plt.tight_layout(); plt.savefig('viz_correlation.png'); plt.close()

print("\nCorrelation with Potability:\n", corr['Potability'].sort_values(ascending=False).round(3))

# ---- Viz 6: WHO / EPA reference-range check (informational only) ----
who_ranges = {
    'ph': (6.5, 8.5),
    'Sulfate': (0, 250),
    'Chloramines': (0, 4),
    'Turbidity': (0, 5),
    'Trihalomethanes': (0, 80),
}
print("\n% of samples OUTSIDE WHO/EPA reference range (informational):")
for k, (lo, hi) in who_ranges.items():
    pct = ((df_clean[k] < lo) | (df_clean[k] > hi)).mean() * 100
    print(f"  {k}: {pct:.1f}% outside [{lo},{hi}]")

print("\nStep 2 complete -> water_potability_clean.csv written, 6 charts saved.")
