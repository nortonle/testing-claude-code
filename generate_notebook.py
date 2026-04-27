"""
Generate churn_prediction_analysis.ipynb
Run: python3 generate_notebook.py
"""
import json, os

def code_cell(src):
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md_cell(
"""# Customer Churn Prediction: ML vs Deep Learning Comparative Analysis

**Author**: Duc Le | **Study**: Master's Thesis Research

---

## Overview

This notebook implements a comprehensive comparative study between **4 classical ML models**
and **2 deep learning models** for customer churn prediction.

| Model Family | Models |
|:---|:---|
| Machine Learning | Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), XGBoost (XGB) |
| Deep Learning | ANN (2 hidden layers), DNN (4 hidden layers) |

### Three Experimental Scenarios

| # | Scenario | Features | Hyperparameters |
|:---|:---|:---|:---|
| 1 | Baseline | Original | Default |
| 2 | Feature Engineering | EDA-driven | Default |
| 3 | HPO | EDA-driven | Tuned (Grid / Random / Optuna) |

### Evaluation Metrics
- **Primary champion criterion**: Recall (minimises missed churners)
- Accuracy, Precision, F1-Score, Training Time, Inference Time

### Methodological Guarantees
- 5-fold stratified cross-validation for all model evaluation
- SMOTE applied **inside each fold** (prevent data leakage)
- Target encoding computed **inside each fold** (prevent data leakage)
- 200 K stratified subsample for baseline scenarios
- 150 K stratified subsample for HPO search
"""))

# ── SECTION 1: DATA LOADING ───────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 1 — Data Loading

We load the full 1 M-row customer churn dataset, inspect its shape, dtypes, and
summary statistics. The full dataset is used for EDA; stratified subsamples are
drawn later for model training to keep compute feasible.
"""))

cells.append(code_cell(
"""# ── Environment fix — run ONCE, then Kernel → Restart, then run Cell 3 ────────
import subprocess, sys, os, glob

# 1. numpy 2.x breaks pandas/pyarrow — pin to last 1.x
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2"])

# 2. shap 0.47+ requires numpy>=2 — pin to last numpy-1.x-compatible release
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap==0.46.0"])

# 3. Remaining packages
for pkg in ["scikit-learn>=1.6", "imbalanced-learn>=0.12",
            "optuna>=3.6", "xgboost>=2.0", "scikeras>=0.13", "tensorflow==2.16.2"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg])

# 4. Remove broken TF Metal plugin on macOS Apple Silicon.
#    libmetal_plugin.dylib links against a Linux .so that does not exist on macOS.
#    Removing it lets TF load cleanly on CPU.
for plugin in glob.glob(os.path.join(
        sys.prefix, "lib", "python*",
        "site-packages", "tensorflow-plugins", "*.dylib")):
    try:
        os.remove(plugin)
        print(f"Removed broken plugin: {os.path.basename(plugin)}")
    except Exception as e:
        print(f"Could not remove {plugin}: {e}")

print("\\nDone. Kernel → Restart Kernel, then re-run this cell, then Cell 3.")
"""))

cells.append(code_cell(
"""import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_METAL'] = '0'   # disable Metal GPU plugin on macOS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# sklearn
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    cross_val_score, train_test_split
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.base import clone

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# sklearn TargetEncoder (sklearn >= 1.3, no extra package needed)
from sklearn.preprocessing import TargetEncoder

# XGBoost
from xgboost import XGBClassifier

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# SHAP (requires scikit-learn >= 1.6 — ensured by the install cell above)
import shap

# TensorFlow / Keras — compatible with TF <= 2.15 and TF >= 2.16 (standalone Keras 3)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# scipy distributions for RandomizedSearchCV
from scipy.stats import loguniform, randint

# ── Global configuration ──────────────────────────────────────────────────────
RANDOM_STATE        = 42
BASELINE_SAMPLE_SIZE = 200_000   # rows for Scenario 1 & 2
HPO_SAMPLE_SIZE     = 150_000    # rows for HPO search (Scenario 3)
N_FOLDS             = 5          # stratified k-fold splits
HPO_SCORING         = 'recall'   # metric optimised by all HPO strategies
CARDINALITY_THRESHOLD = 7        # < 7 → OHE; >= 7 → TargetEncoder

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("All libraries loaded.")
print(f"TensorFlow : {tf.__version__}")
"""))

cells.append(code_cell(
"""DATA_PATH = "customer_churn_1M.csv"

df_full = pd.read_csv(DATA_PATH, parse_dates=['signup_date'])

print(f"Shape  : {df_full.shape[0]:,} rows x {df_full.shape[1]} columns")
print(f"Target : 'churn'  (unique values: {sorted(df_full['churn'].unique())})")
print(f"Memory : {df_full.memory_usage(deep=True).sum() / 1e6:.1f} MB")
"""))

cells.append(code_cell(
"""# Dtypes overview
print(df_full.dtypes.to_string())
print()
display(df_full.head(3))
print()
display(df_full.describe(include='all').T.round(2))
"""))

# ── SECTION 2: EDA ────────────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 2 — Exploratory Data Analysis (EDA)

EDA covers:
1. **Missing values** — affected columns and extent
2. **Target distribution** — class imbalance ratio
3. **Numerical features** — distributions split by churn label
4. **Categorical features** — churn rate per category
5. **Correlation analysis** — Pearson correlation heatmap
6. **Feature–target relationships** — boxplots for key predictors

Findings directly motivate the feature engineering choices in Section 3.
"""))

cells.append(code_cell(
"""# ── 2.1 Missing Values ──────────────────────────────────────────────────────
missing_count = df_full.isnull().sum()
missing_pct   = (missing_count / len(df_full) * 100).round(3)
missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %': missing_pct
}).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

print("Columns with missing values:")
display(missing_df)

if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_df) * 0.6)))
    missing_df['Missing %'].plot.barh(ax=ax, color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Value Rate per Feature', fontweight='bold')
    for i, v in enumerate(missing_df['Missing %']):
        ax.text(v + 0.05, i, f'{v:.2f}%', va='center')
    plt.tight_layout()
    plt.show()
"""))

cells.append(code_cell(
"""# ── 2.2 Target Distribution & Class Imbalance ────────────────────────────────
churn_counts = df_full['churn'].value_counts()
churn_pct    = (df_full['churn'].value_counts(normalize=True) * 100).round(2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
labels = ['No Churn (0)', 'Churn (1)']
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(labels, churn_counts.values, color=colors, alpha=0.85,
                   edgecolor='white', width=0.5)
for bar, count, pct in zip(bars, churn_counts.values, churn_pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                 f'{count:,}\\n({pct}%)', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')

# Pie
axes[1].pie(churn_counts.values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0, 0.06),
            textprops={'fontsize': 11})
axes[1].set_title('Churn Proportion', fontsize=13, fontweight='bold')

plt.suptitle('Target Variable — Churn', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"Imbalance ratio (majority:minority) = {churn_counts[0]/churn_counts[1]:.2f}:1")
print("-> SMOTE applied within each CV fold to address this imbalance.")
"""))

cells.append(code_cell(
"""# ── 2.3 Numerical Feature Distributions (by Churn Label) ─────────────────────
num_cols_eda = [
    c for c in df_full.select_dtypes(include=['int64','float64']).columns
    if c not in ['churn','senior_citizen'] and df_full[c].nunique() > 2
]

n = len(num_cols_eda)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(num_cols_eda):
    for label, color in zip([0, 1], ['#2ecc71', '#e74c3c']):
        subset = df_full[df_full['churn'] == label][col].dropna()
        axes[i].hist(subset, bins=40, alpha=0.5, color=color,
                     label=f'Churn={label}', density=True)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Numerical Feature Distributions by Churn', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 2.4 Categorical Features — Churn Rate per Category ───────────────────────
cat_cols_eda = [
    c for c in df_full.select_dtypes(include=['object']).columns
    if c not in ['customer_id']
]

print("Cardinality of categorical features:")
for col in cat_cols_eda:
    print(f"  {col:25s}: {df_full[col].nunique()} unique -> {df_full[col].unique().tolist()}")

n = len(cat_cols_eda)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(cat_cols_eda):
    churn_rate = df_full.groupby(col)['churn'].mean().sort_values(ascending=False)
    bars = axes[i].bar(churn_rate.index.astype(str), churn_rate.values,
                       color='#3498db', alpha=0.8, edgecolor='white')
    axes[i].set_title(f'Churn Rate by {col}', fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Churn Rate')
    axes[i].set_ylim(0, min(churn_rate.max() * 1.35, 1.0))
    axes[i].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, churn_rate.values):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003, f'{val:.3f}',
                     ha='center', fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Churn Rate by Categorical Feature', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 2.5 Correlation Heatmap (numeric features) ───────────────────────────────
corr_sample = df_full.select_dtypes(include=['int64','float64']).sample(
    n=50_000, random_state=RANDOM_STATE
)
corr_matrix = corr_sample.corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # hide upper triangle
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.4, ax=ax, annot_kws={'size': 8}
)
ax.set_title('Pearson Correlation Matrix (numeric features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nTop correlations with 'churn':")
target_corr = corr_matrix['churn'].drop('churn').abs().sort_values(ascending=False)
display(target_corr.head(10).to_frame('|correlation with churn|').round(4))
"""))

cells.append(code_cell(
"""# ── 2.6 Key Numerical Features vs Churn (Boxplots) ───────────────────────────
key_num = ['tenure','monthlycharges','customer_satisfaction',
           'num_complaints','num_service_calls','days_since_last_interaction']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(key_num):
    data_no  = df_full[df_full['churn'] == 0][col].dropna()
    data_yes = df_full[df_full['churn'] == 1][col].dropna()
    axes[i].boxplot(
        [data_no.sample(min(5000, len(data_no)),   random_state=RANDOM_STATE),
         data_yes.sample(min(5000, len(data_yes)), random_state=RANDOM_STATE)],
        labels=['No Churn', 'Churn'], patch_artist=True,
        boxprops=dict(facecolor='#aed6f1', color='#2980b9'),
        medianprops=dict(color='#e74c3c', linewidth=2)
    )
    axes[i].set_title(col, fontsize=11, fontweight='bold')

plt.suptitle('Key Numerical Features vs Churn', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

print("\\nChurn rate by contract type (highest predictor):")
display(
    df_full.groupby('contract')['churn']
           .agg(['mean','count'])
           .rename(columns={'mean':'churn_rate','count':'n'})
           .sort_values('churn_rate', ascending=False)
           .round(4)
)
"""))

cells.append(md_cell(
"""### EDA Key Findings

| Finding | Detail | Action |
|:---|:---|:---|
| **Class imbalance** | Churn ~15–30% | SMOTE within each CV fold |
| **Missing values** | `credit_score` ~10% missing | Median imputation |
| **Tenure** | Long-tenure customers rarely churn | `satisfaction_tenure_score` interaction |
| **Contract type** | Month-to-month highest churn rate | `is_month_to_month` binary flag |
| **Monthly charges** | Churners pay more per service | `charges_per_service` ratio |
| **Satisfaction + complaints** | Strong negative / positive correlation with churn | `engagement_risk` composite |
| **signup_date** | Potential seasonal variation | Year / month / quarter features |
"""))

# ── SECTION 3: FEATURE ENGINEERING ───────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 3 — Feature Engineering

Engineered features are derived **solely from EDA findings** on the 1 M-row dataset
(original Appendix B and Section 7.3 of the proposal are superseded).

| New Feature | Formula | Motivation |
|:---|:---|:---|
| `account_age_days` | `reference_date - signup_date` | Account longevity proxy |
| `signup_year/month/quarter` | Datetime decomposition | Seasonal churn patterns |
| `charges_per_service` | `monthlycharges / (num_services + 1)` | Perceived value per service |
| `total_to_monthly_ratio` | `totalcharges / (monthlycharges + 1)` | Independent tenure proxy |
| `charge_income_ratio` | `monthlycharges / (annual_income/12 + 1)` | Financial burden |
| `satisfaction_tenure_score` | `satisfaction * log1p(tenure)` | Long-term satisfaction anchor |
| `engagement_risk` | `complaints + service_calls + late_payments` | Composite friction score |
| `is_month_to_month` | `contract == 'month-to-month'` | Highest-risk contract flag |
"""))

cells.append(code_cell(
"""# ── 3.1 Temporal Features from signup_date ───────────────────────────────────
REFERENCE_DATE = df_full['signup_date'].max()   # latest date as reference point

df_full['account_age_days'] = (REFERENCE_DATE - df_full['signup_date']).dt.days
df_full['signup_year']      = df_full['signup_date'].dt.year
df_full['signup_month']     = df_full['signup_date'].dt.month
df_full['signup_quarter']   = df_full['signup_date'].dt.quarter

# Drop columns with no predictive value in the model
df_full.drop(columns=['signup_date', 'customer_id'], inplace=True)

print(f"Reference date   : {REFERENCE_DATE.date()}")
print(f"account_age_days : {df_full['account_age_days'].min()} to {df_full['account_age_days'].max()} days")
"""))

cells.append(code_cell(
"""# ── 3.2 Ratio & Interaction Features ─────────────────────────────────────────

# How much does the customer pay per service? High ratio = poor value perception
df_full['charges_per_service'] = df_full['monthlycharges'] / (df_full['num_services'] + 1)

# totalcharges / monthlycharges approximates tenure independently
df_full['total_to_monthly_ratio'] = df_full['totalcharges'] / (df_full['monthlycharges'] + 1)

# Monthly charges as fraction of monthly income — financial burden
df_full['charge_income_ratio'] = (
    df_full['monthlycharges'] / (df_full['annual_income'] / 12 + 1)
)

# Satisfied long-term customers are anchored; low values indicate risk
df_full['satisfaction_tenure_score'] = (
    df_full['customer_satisfaction'] * np.log1p(df_full['tenure'])
)

print("Ratio/interaction features created.")
display(df_full[['charges_per_service','total_to_monthly_ratio',
                 'charge_income_ratio','satisfaction_tenure_score']].describe().round(3))
"""))

cells.append(code_cell(
"""# ── 3.3 Composite & Binary Features ──────────────────────────────────────────

# Composite friction: more contact events -> higher churn risk
df_full['engagement_risk'] = (
    df_full['num_complaints'] +
    df_full['num_service_calls'] +
    df_full['late_payments']
)

# Binary flag for month-to-month (highest churn risk contract — confirmed by EDA)
df_full['is_month_to_month'] = (df_full['contract'] == 'month-to-month').astype(int)

# ── Validation ────────────────────────────────────────────────────────────────
engineered = ['account_age_days','signup_year','signup_month','signup_quarter',
              'charges_per_service','total_to_monthly_ratio','charge_income_ratio',
              'satisfaction_tenure_score','engagement_risk','is_month_to_month']

print(f"Total columns after FE : {df_full.shape[1]}")
print(f"\\nEngineered feature null counts:")
print(df_full[engineered].isnull().sum().to_string())

print("\\nCorrelation of engineered features with churn:")
fe_corr = df_full[engineered + ['churn']].corr()['churn'].drop('churn')
display(fe_corr.sort_values(key=abs, ascending=False).to_frame('corr').round(4))
"""))

# ── SECTION 4: PREPROCESSING ──────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 4 — Preprocessing Pipeline

### Design

| Step | Numeric | Cat (cardinality < 7) | Cat (cardinality >= 7) |
|:---|:---|:---|:---|
| **Imputation** | Median | Mode | Mode |
| **Encoding** | — | OneHotEncoder (drop first) | TargetEncoder |
| **Scaling** | StandardScaler | — | — |

### Leak-proof Pipeline Architecture

```
imblearn.Pipeline
  ├─ ColumnTransformer  ← fitted on training fold only
  │     ├─ numeric  : SimpleImputer(median) → StandardScaler
  │     ├─ low-card : SimpleImputer(mode)   → OneHotEncoder(drop=first)
  │     └─ high-card: SimpleImputer(mode)   → TargetEncoder
  ├─ SMOTE             ← applied on training fold only (after preprocessing)
  └─ Classifier
```

Both target encoding and SMOTE are guaranteed to see **only training-fold labels**.
"""))

cells.append(code_cell(
"""# ── 4.1 Column Group Definitions ─────────────────────────────────────────────
TARGET = 'churn'
ALL_FEATURES = [c for c in df_full.columns if c != TARGET]

# Separate numeric vs categorical columns
numeric_cols_raw = (df_full[ALL_FEATURES]
                    .select_dtypes(include=['int64','float64'])
                    .columns.tolist())
cat_cols_raw     = (df_full[ALL_FEATURES]
                    .select_dtypes(include=['object'])
                    .columns.tolist())

# Split categorical by cardinality
low_card_cols  = [c for c in cat_cols_raw if df_full[c].nunique() <  CARDINALITY_THRESHOLD]
high_card_cols = [c for c in cat_cols_raw if df_full[c].nunique() >= CARDINALITY_THRESHOLD]

print(f"Numeric features       : {len(numeric_cols_raw)}")
print(f"Low-cardinality (OHE)  : {low_card_cols}")
print(f"High-cardinality (TE)  : {high_card_cols}")
"""))

cells.append(code_cell(
"""# ── 4.2 Pipeline Builder ──────────────────────────────────────────────────────

def build_pipeline(classifier, num_cols, low_card, high_card):
    \"\"\"
    Build an imblearn.Pipeline:
        ColumnTransformer (impute + encode + scale)
        -> SMOTE
        -> classifier

    All transformers are fitted inside each CV fold only — no leakage.
    The TargetEncoder in ColumnTransformer receives y from the training fold
    because imblearn.Pipeline passes y through fit_transform correctly.
    \"\"\"
    # Numeric: median imputation -> standardisation
    numeric_pipe = SkPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    transformers = [('num', numeric_pipe, num_cols)]

    # Low-cardinality categorical: mode imputation -> one-hot (drop first level)
    if low_card:
        ohe_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(drop='first', sparse_output=False,
                                      handle_unknown='ignore')),
        ])
        transformers.append(('ohe', ohe_pipe, low_card))

    # High-cardinality categorical: mode imputation -> target encoding
    # TargetEncoder.fit receives y from the training fold via ColumnTransformer
    if high_card:
        te_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('te',      TargetEncoder(smooth='auto', target_type='binary')),
        ])
        transformers.append(('te', te_pipe, high_card))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote',        SMOTE(random_state=RANDOM_STATE)),
        ('classifier',   classifier),
    ])


def build_preprocessor_only(num_cols, low_card, high_card):
    \"\"\"Return the bare ColumnTransformer (no SMOTE, no classifier) for DL manual CV.\"\"\"
    numeric_pipe = SkPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    transformers = [('num', numeric_pipe, num_cols)]
    if low_card:
        ohe_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(drop='first', sparse_output=False,
                                      handle_unknown='ignore')),
        ])
        transformers.append(('ohe', ohe_pipe, low_card))
    if high_card:
        te_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('te',      TargetEncoder(smooth='auto', target_type='binary')),
        ])
        transformers.append(('te', te_pipe, high_card))
    return ColumnTransformer(transformers=transformers, remainder='drop')
"""))

cells.append(code_cell(
"""# ── 4.3 Cross-Validation Helper & Results Storage ────────────────────────────

def evaluate_cv(pipeline, X, y, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    \"\"\"
    Stratified k-fold CV.
    Returns mean accuracy, precision, recall, f1, train_time, infer_time.
    SMOTE is applied inside the pipeline on training folds only.
    \"\"\"
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metrics = defaultdict(list)

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        t0 = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        metrics['train_time'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_val)
        metrics['infer_time'].append(time.perf_counter() - t0)

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def evaluate_dl_cv(model_fn, X, y, num_cols, low_card, high_card,
                   epochs=20, batch_size=512,
                   n_folds=N_FOLDS, random_state=RANDOM_STATE):
    \"\"\"
    DL-specific CV: preprocessor fitted per fold, class_weight computed per fold.
    No SMOTE — cost-sensitive learning is the appropriate strategy for DL
    (SMOTE on OHE-encoded data creates fractional synthetic values that do not
    exist in real validation data, causing a systematic CV-test recall gap).
    \"\"\"
    import gc
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metrics = defaultdict(list)
    y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        prep = build_preprocessor_only(num_cols, low_card, high_card)
        X_tr_p  = prep.fit_transform(X_tr, y_tr)
        X_val_p = prep.transform(X_val)

        n_neg, n_pos, n_total = int((y_tr==0).sum()), int((y_tr==1).sum()), len(y_tr)
        cw = {0: n_total / (2 * n_neg), 1: n_total / (2 * n_pos)}

        tf.random.set_seed(random_state + fold_i)
        model = model_fn(X_tr_p.shape[1])

        t0 = time.perf_counter()
        model.fit(X_tr_p, y_tr, epochs=epochs, batch_size=batch_size,
                  class_weight=cw,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                           restore_best_weights=True, verbose=0)],
                  validation_split=0.1, verbose=0)
        metrics['train_time'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_prob = model.predict(X_val_p, verbose=0).flatten()
        metrics['infer_time'].append(time.perf_counter() - t0)

        y_pred = (y_prob > 0.5).astype(int)
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))

        del model; gc.collect()

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def results_to_df(results_dict, scenario_label):
    \"\"\"Convert {model_name: metrics_dict} to a tidy DataFrame row per model.\"\"\"
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            'Scenario':      scenario_label,
            'Model':         model_name,
            'Accuracy':      round(m['accuracy'],    4),
            'Precision':     round(m['precision'],   4),
            'Recall':        round(m['recall'],      4),
            'F1':            round(m['f1'],          4),
            'Train_Time_s':  round(m['train_time'],  2),
            'Infer_Time_s':  round(m['infer_time'],  4),
        })
    return pd.DataFrame(rows)


# Global containers
ALL_RESULTS = []   # list of DataFrames, one per scenario / HPO method
BEST_HPO    = {}   # {model: {method, recall, metrics, best_params}}

print("Pipeline builder and CV helper ready.")
"""))

# ── SECTION 5: SCENARIO 1 ────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 5 — Scenario 1: Baseline (Default Hyperparameters, Original Features)

All six models trained with **default hyperparameters** on the **original feature set**
(no engineered features; only `customer_id` and raw `signup_date` are dropped).

- Sample: 200 K stratified rows
- CV: 5-fold stratified, SMOTE inside each fold
- Preprocessing: median/mode imputation → OHE / TargetEncoding → StandardScaler
"""))

cells.append(code_cell(
"""# ── 5.1 Subsample — original feature set (no FE columns) ────────────────────
# Re-read to get the raw column set before feature engineering
df_s1_raw = pd.read_csv(DATA_PATH, parse_dates=['signup_date'])
df_s1_raw.drop(columns=['customer_id', 'signup_date'], inplace=True)

_, df_s1 = train_test_split(
    df_s1_raw, test_size=BASELINE_SAMPLE_SIZE / len(df_s1_raw),
    stratify=df_s1_raw['churn'], random_state=RANDOM_STATE
)
df_s1 = df_s1.reset_index(drop=True)

X_s1 = df_s1.drop(columns=[TARGET])
y_s1 = df_s1[TARGET]

# Column groups for original feature set
num_s1       = X_s1.select_dtypes(include=['int64','float64']).columns.tolist()
cat_s1       = X_s1.select_dtypes(include=['object']).columns.tolist()
low_card_s1  = [c for c in cat_s1 if X_s1[c].nunique() <  CARDINALITY_THRESHOLD]
high_card_s1 = [c for c in cat_s1 if X_s1[c].nunique() >= CARDINALITY_THRESHOLD]

print(f"Scenario 1 sample : {df_s1.shape}  |  churn rate : {y_s1.mean():.4f}")
print(f"Features          : {X_s1.shape[1]}")
"""))

cells.append(code_cell(
"""# ── 5.2 Deep Learning Model Builders ─────────────────────────────────────────

def build_ann(meta=None, n_units_1=64, n_units_2=32,
              dropout_rate=0.3, learning_rate=0.001):
    \"\"\"ANN: 2 hidden layers (shallow network).\"\"\"
    n_features = meta['n_features_in_']
    model = Sequential([
        Dense(n_units_1, activation='relu', input_shape=(n_features,)),
        Dropout(dropout_rate),
        Dense(n_units_2, activation='relu'),
        Dropout(dropout_rate * 0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy')
    return model


def build_dnn(meta=None, n_layers=4, base_units=128,
              dropout_rate=0.3, learning_rate=0.001):
    \"\"\"DNN: 4 hidden layers with halving units (deep network).\"\"\"
    n_features = meta['n_features_in_']
    layer_list = [Dense(base_units, activation='relu', input_shape=(n_features,))]
    for i in range(1, n_layers):
        layer_list.append(Dropout(dropout_rate))
        units = max(base_units // (2 ** i), 16)
        layer_list.append(Dense(units, activation='relu'))
    layer_list.append(Dropout(dropout_rate * 0.5))
    layer_list.append(Dense(1, activation='sigmoid'))
    model = Sequential(layer_list)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy')
    return model


ANN_DEFAULT = KerasClassifier(model=build_ann, epochs=20, batch_size=512,
                               verbose=0, random_state=RANDOM_STATE)
DNN_DEFAULT = KerasClassifier(model=build_dnn, epochs=20, batch_size=512,
                               verbose=0, random_state=RANDOM_STATE)

# Raw builders for evaluate_dl_cv (accept n_features directly, no meta dict)
def build_ann_raw(n_features):
    model = Sequential([
        Dense(64,  activation='relu', input_shape=(n_features,)),
        Dropout(0.3),
        Dense(32,  activation='relu'),
        Dropout(0.15),
        Dense(1,   activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def build_dnn_raw(n_features):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        Dropout(0.3),
        Dense(64,  activation='relu'),
        Dropout(0.3),
        Dense(32,  activation='relu'),
        Dropout(0.3),
        Dense(16,  activation='relu'),
        Dropout(0.15),
        Dense(1,   activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

DL_BUILDERS = {'ANN': build_ann_raw, 'DNN': build_dnn_raw}

BASELINE_ML = {
    'LR':  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'DT':  DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RF':  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=100, eval_metric='logloss',
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
}
BASELINE_DL = {'ANN': ANN_DEFAULT, 'DNN': DNN_DEFAULT}

MODEL_NAMES = list(BASELINE_ML.keys()) + list(BASELINE_DL.keys())

print("Baseline model definitions ready.")
"""))

cells.append(code_cell(
"""# ── 5.3 Run 5-Fold CV — Scenario 1 ─────────────────────────────────────────
s1_results = {}
print("Scenario 1 — Baseline (default params, original features)")
print(f"Sample: {len(X_s1):,} rows  |  Folds: {N_FOLDS}\\n")

for name, clf in BASELINE_ML.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_s1, low_card_s1, high_card_s1)
    t0 = time.time()
    s1_results[name] = evaluate_cv(pipe, X_s1, y_s1)
    print(f"done ({time.time()-t0:.0f}s)  Recall={s1_results[name]['recall']:.4f}")

for name, fn in DL_BUILDERS.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    t0 = time.time()
    s1_results[name] = evaluate_dl_cv(fn, X_s1, y_s1, num_s1, low_card_s1, high_card_s1)
    print(f"done ({time.time()-t0:.0f}s)  Recall={s1_results[name]['recall']:.4f}")

df_s1_res = results_to_df(s1_results, 'Scenario 1')
ALL_RESULTS.append(df_s1_res)

print("\\n── Scenario 1 Results ──────────────────────────────────")
display(df_s1_res.set_index('Model').drop(columns='Scenario'))
"""))

# ── SECTION 6: SCENARIO 2 ────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 6 — Scenario 2: Baseline + Feature Engineering

Same default hyperparameters as Scenario 1, but using the **engineered feature set**
from Section 3. This isolates the contribution of feature engineering.

- Sample: 200 K stratified rows (with engineered features)
"""))

cells.append(code_cell(
"""# ── 6.1 Subsample with engineered features ────────────────────────────────────
_, df_s2 = train_test_split(
    df_full, test_size=BASELINE_SAMPLE_SIZE / len(df_full),
    stratify=df_full[TARGET], random_state=RANDOM_STATE
)
df_s2 = df_s2.reset_index(drop=True)

X_s2 = df_s2.drop(columns=[TARGET])
y_s2 = df_s2[TARGET]

# Column groups after FE (more numeric features due to engineered columns)
num_s2       = X_s2.select_dtypes(include=['int64','float64']).columns.tolist()
cat_s2       = X_s2.select_dtypes(include=['object']).columns.tolist()
low_card_s2  = [c for c in cat_s2 if X_s2[c].nunique() <  CARDINALITY_THRESHOLD]
high_card_s2 = [c for c in cat_s2 if X_s2[c].nunique() >= CARDINALITY_THRESHOLD]

print(f"Scenario 2 sample : {df_s2.shape}  |  churn rate : {y_s2.mean():.4f}")
print(f"Features (with FE): {X_s2.shape[1]}  (vs {X_s1.shape[1]} in Scenario 1)")
"""))

cells.append(code_cell(
"""# ── 6.2 Run 5-Fold CV — Scenario 2 ──────────────────────────────────────────
s2_results = {}
print("Scenario 2 — Baseline + Feature Engineering")
print(f"Sample: {len(X_s2):,} rows  |  Folds: {N_FOLDS}\\n")

for name, clf in BASELINE_ML.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_s2, low_card_s2, high_card_s2)
    t0 = time.time()
    s2_results[name] = evaluate_cv(pipe, X_s2, y_s2)
    print(f"done ({time.time()-t0:.0f}s)  Recall={s2_results[name]['recall']:.4f}")

for name, fn in DL_BUILDERS.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    t0 = time.time()
    s2_results[name] = evaluate_dl_cv(fn, X_s2, y_s2, num_s2, low_card_s2, high_card_s2)
    print(f"done ({time.time()-t0:.0f}s)  Recall={s2_results[name]['recall']:.4f}")

df_s2_res = results_to_df(s2_results, 'Scenario 2')
ALL_RESULTS.append(df_s2_res)

print("\\n── Scenario 2 Results ──────────────────────────────────")
display(df_s2_res.set_index('Model').drop(columns='Scenario'))
"""))

# ── SCENARIO 2b: CLASS-WEIGHT ONLY (no SMOTE) — supplementary comparison ─────
cells.append(md_cell(
"""---
## Scenario 2b — Class-Weight Only (Supplementary Comparison)

This extra scenario replaces SMOTE with **cost-sensitive learning** (`class_weight` /
`scale_pos_weight`) using the **original dataset imbalance ratio** as the weight.
It uses the same engineered feature set and sample as Scenario 2.

**Why include this?**
SMOTE interpolates between minority samples in the *preprocessed* (post-OHE) feature
space. For binary one-hot features this creates fractional synthetic values
(e.g. 0.3, 0.7) that do not exist in real validation data. Tree models like RF split
on these artefacts and learn decision boundaries that do not generalise — yielding
near-zero recall on the real test distribution. LR is immune because its linear
boundary generalises regardless.

Cost-sensitive learning avoids synthetic samples entirely: it penalises
misclassification of minority samples by a factor proportional to the imbalance
ratio, which works identically for all model families.

> Breiman (2001) and Chen et al. (2004, XGBoost precursor) both recommend
> cost-sensitive approaches over resampling for tree ensembles on tabular data.
"""))

cells.append(code_cell(
"""# ── 6b.1 Build class-weight pipeline (standard sklearn Pipeline, no SMOTE) ────
_n_pos = int((y_s2 == 1).sum())
_n_neg = int((y_s2 == 0).sum())
_pos_w = _n_neg / _n_pos
_cw    = {0: 1, 1: int(_pos_w)}
print(f"Imbalance ratio (neg/pos): {_pos_w:.1f}   class_weight = {_cw}")

def build_cw_pipeline(classifier, num_cols, low_card, high_card):
    \"\"\"Same preprocessing as build_pipeline but WITHOUT SMOTE.\"\"\"
    numeric_pipe = SkPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    transformers = [('num', numeric_pipe, num_cols)]
    if low_card:
        ohe_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe',     OneHotEncoder(drop='first', sparse_output=False,
                                      handle_unknown='ignore')),
        ])
        transformers.append(('ohe', ohe_pipe, low_card))
    if high_card:
        te_pipe = SkPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('te',      TargetEncoder(smooth='auto', target_type='binary')),
        ])
        transformers.append(('te', te_pipe, high_card))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return SkPipeline([('preprocessor', preprocessor), ('classifier', classifier)])

CW_ML = {
    'LR':  LogisticRegression(max_iter=1000, class_weight=_cw,
                               random_state=RANDOM_STATE, n_jobs=-1),
    'DT':  DecisionTreeClassifier(class_weight=_cw, random_state=RANDOM_STATE),
    'RF':  RandomForestClassifier(n_estimators=100, class_weight=_cw,
                                   random_state=RANDOM_STATE, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=100, scale_pos_weight=_pos_w,
                         eval_metric='logloss',
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
}

def build_keras_cw(meta=None, n_units_1=64, n_units_2=32,
                   dropout_rate=0.3, learning_rate=0.001):
    n_f = meta['n_features_in_']
    m = Sequential([Dense(n_units_1,'relu',input_shape=(n_f,)),Dropout(dropout_rate),
                     Dense(n_units_2,'relu'),Dropout(dropout_rate*0.5),Dense(1,'sigmoid')])
    m.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    return m

CW_DL = {
    'ANN': KerasClassifier(model=build_keras_cw, epochs=20, batch_size=512,
                            class_weight=_cw, verbose=0, random_state=RANDOM_STATE),
    'DNN': KerasClassifier(model=build_dnn, epochs=20, batch_size=512,
                            class_weight=_cw, verbose=0, random_state=RANDOM_STATE),
}
print("Class-weight pipeline ready.")
"""))

cells.append(code_cell(
"""# ── 6b.2 Run CV — Scenario 2b ─────────────────────────────────────────────
s2b_results = {}
print("Scenario 2b — Class-Weight Only (no SMOTE), engineered features")
print(f"Sample: {len(X_s2):,} rows  |  Folds: {N_FOLDS}\\n")

for name, clf in {**CW_ML, **CW_DL}.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_cw_pipeline(clone(clf), num_s2, low_card_s2, high_card_s2)
    t0 = time.time()
    s2b_results[name] = evaluate_cv(pipe, X_s2, y_s2)
    print(f"done ({time.time()-t0:.0f}s)  Recall={s2b_results[name]['recall']:.4f}")

df_s2b_res = results_to_df(s2b_results, 'Scenario 2b (CW)')
ALL_RESULTS.append(df_s2b_res)

print("\\n── Scenario 2 (SMOTE) vs Scenario 2b (Class-Weight) ─────────────────")
comparison = pd.concat([
    df_s2_res.set_index('Model')[['Recall','F1']].rename(columns=lambda c: f'SMOTE_{c}'),
    df_s2b_res.set_index('Model')[['Recall','F1']].rename(columns=lambda c: f'CW_{c}'),
], axis=1)
display(comparison.round(4))
"""))

# ── SECTION 7: HPO ───────────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 7 — Scenario 3: Hyperparameter Optimisation (HPO)

All six models (including ANN and DNN) are tuned using three strategies on
the **engineered feature set** and the **150 K HPO subsample**.

| Strategy | Library | Nature |
|:---|:---|:---|
| **GridSearchCV** | sklearn | Exhaustive grid |
| **RandomizedSearchCV** | sklearn | Random sampling |
| **Optuna** | optuna | Bayesian (TPE) |

After each HPO search, the best parameters are re-evaluated with **5-fold CV**
(3-fold used inside Optuna/GridSearchCV for DL models to manage Keras cost).

> **Runtime note**: GridSearchCV on RF and XGB can take several hours.
> Reduce grid sizes or set `n_jobs` appropriately for your hardware.
"""))

cells.append(code_cell(
"""# ── HPO Subsample: 150 K stratified rows ─────────────────────────────────────
_, df_hpo = train_test_split(
    df_full, test_size=HPO_SAMPLE_SIZE / len(df_full),
    stratify=df_full[TARGET], random_state=RANDOM_STATE
)
df_hpo = df_hpo.reset_index(drop=True)

X_hpo = df_hpo.drop(columns=[TARGET])
y_hpo = df_hpo[TARGET]

# Reuse column groups from Scenario 2 (same engineered feature set)
num_hpo       = num_s2.copy()
low_card_hpo  = low_card_s2.copy()
high_card_hpo = high_card_s2.copy()

print(f"HPO sample shape : {df_hpo.shape}  |  churn rate : {y_hpo.mean():.4f}")

# Result containers
hpo_grid   = {}
hpo_rand   = {}
hpo_optuna = {}

# EarlyStopping for DL HPO (faster convergence per trial)
ES_HPO = EarlyStopping(patience=3, restore_best_weights=True, verbose=0)

CV_ML  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
CV_DL  = StratifiedKFold(n_splits=3,       shuffle=True, random_state=RANDOM_STATE)
"""))

# ── GridSearchCV ──────────────────────────────────────────────────────────────
cells.append(md_cell("### 7a — GridSearchCV\n"))

cells.append(code_cell(
"""# ── 7a.1 Parameter Grids (compact for feasibility) ───────────────────────────
GRID_ML = {
    'LR': {
        'classifier__C':       [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver':  ['liblinear'],
    },
    'DT': {
        'classifier__max_depth':         [5, 10, 15, None],
        'classifier__min_samples_split':  [2, 20, 50],
        'classifier__criterion':          ['gini', 'entropy'],
    },
    'RF': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth':    [5, 10, None],
        'classifier__max_features': ['sqrt', 'log2'],
    },
    'XGB': {
        'classifier__n_estimators':  [100, 200],
        'classifier__max_depth':     [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample':     [0.8, 1.0],
    },
}

GRID_DL = {
    'ANN': {
        'classifier__model__n_units_1':    [64, 128],
        'classifier__model__dropout_rate': [0.2, 0.3],
        'classifier__model__learning_rate':[0.001, 0.01],
    },
    'DNN': {
        'classifier__model__n_layers':     [3, 4],
        'classifier__model__base_units':   [64, 128],
        'classifier__model__learning_rate':[0.001, 0.01],
    },
}

# DL model wrappers for HPO (fewer epochs + EarlyStopping)
def build_ann_hpo(meta=None, n_units_1=64, n_units_2=32,
                   dropout_rate=0.3, learning_rate=0.001):
    return build_ann(meta=meta, n_units_1=n_units_1, n_units_2=n_units_2,
                     dropout_rate=dropout_rate, learning_rate=learning_rate)

def build_dnn_hpo(meta=None, n_layers=4, base_units=128,
                   dropout_rate=0.3, learning_rate=0.001):
    return build_dnn(meta=meta, n_layers=n_layers, base_units=base_units,
                     dropout_rate=dropout_rate, learning_rate=learning_rate)

DL_HPO_CLF = {
    'ANN': KerasClassifier(model=build_ann_hpo, epochs=15, batch_size=512,
                            verbose=0, validation_split=0.1,
                            callbacks=[ES_HPO], random_state=RANDOM_STATE),
    'DNN': KerasClassifier(model=build_dnn_hpo, epochs=15, batch_size=512,
                            verbose=0, validation_split=0.1,
                            callbacks=[ES_HPO], random_state=RANDOM_STATE),
}
"""))

cells.append(code_cell(
"""# ── 7a.2 GridSearchCV — ML Models ────────────────────────────────────────────
print("GridSearchCV — ML models\\n")
for name, clf in BASELINE_ML.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_hpo, low_card_hpo, high_card_hpo)
    gs   = GridSearchCV(pipe, GRID_ML[name], scoring=HPO_SCORING,
                        cv=CV_ML, n_jobs=-1, refit=True, verbose=0)
    t0 = time.time()
    gs.fit(X_hpo, y_hpo)
    elapsed = time.time() - t0

    # Re-evaluate best pipeline with full 5-fold CV for fair reporting
    hpo_grid[name] = evaluate_cv(gs.best_estimator_, X_hpo, y_hpo)
    hpo_grid[name]['best_params'] = gs.best_params_
    print(f"done ({elapsed:.0f}s) | CV recall={gs.best_score_:.4f} "
          f"-> 5-fold recall={hpo_grid[name]['recall']:.4f}")
"""))

cells.append(code_cell(
"""# ── 7a.3 GridSearchCV — DL Models ────────────────────────────────────────────
print("GridSearchCV — DL models (3-fold CV, Keras cost constraint)\\n")
for name, clf in DL_HPO_CLF.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_hpo, low_card_hpo, high_card_hpo)
    gs   = GridSearchCV(pipe, GRID_DL[name], scoring=HPO_SCORING,
                        cv=CV_DL, n_jobs=1, refit=True, verbose=0)
    t0 = time.time()
    gs.fit(X_hpo, y_hpo)
    elapsed = time.time() - t0

    hpo_grid[name] = evaluate_cv(gs.best_estimator_, X_hpo, y_hpo)
    hpo_grid[name]['best_params'] = gs.best_params_
    print(f"done ({elapsed:.0f}s) | CV recall={gs.best_score_:.4f} "
          f"-> 5-fold recall={hpo_grid[name]['recall']:.4f}")

df_grid = results_to_df(hpo_grid, 'GridSearchCV')
print("\\n── GridSearchCV Results ─────────────────────────────────")
display(df_grid.set_index('Model').drop(columns='Scenario'))
"""))

# ── RandomizedSearchCV ────────────────────────────────────────────────────────
cells.append(md_cell("### 7b — RandomizedSearchCV\n"))

cells.append(code_cell(
"""# ── 7b.1 Distribution Specs ──────────────────────────────────────────────────
N_ITER_ML = 30
N_ITER_DL = 15

RAND_ML = {
    'LR': {
        'classifier__C':       loguniform(1e-3, 1e2),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver':  ['liblinear'],
    },
    'DT': {
        'classifier__max_depth':         [*range(3, 20)] + [None],
        'classifier__min_samples_split':  randint(2, 100),
        'classifier__min_samples_leaf':   randint(1, 50),
        'classifier__criterion':          ['gini', 'entropy'],
    },
    'RF': {
        'classifier__n_estimators':      randint(50, 300),
        'classifier__max_depth':         [*range(3, 20)] + [None],
        'classifier__max_features':      ['sqrt', 'log2'],
        'classifier__min_samples_split':  randint(2, 30),
    },
    'XGB': {
        'classifier__n_estimators':      randint(50, 300),
        'classifier__max_depth':         randint(3, 10),
        'classifier__learning_rate':     loguniform(1e-3, 0.5),
        'classifier__subsample':         [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree':  [0.7, 0.8, 0.9, 1.0],
    },
}

RAND_DL = {
    'ANN': {
        'classifier__model__n_units_1':    [32, 64, 128, 256],
        'classifier__model__n_units_2':    [16, 32, 64],
        'classifier__model__dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'classifier__model__learning_rate':loguniform(1e-4, 1e-1),
    },
    'DNN': {
        'classifier__model__n_layers':     [3, 4, 5],
        'classifier__model__base_units':   [64, 128, 256],
        'classifier__model__dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'classifier__model__learning_rate':loguniform(1e-4, 1e-1),
    },
}
"""))

cells.append(code_cell(
"""# ── 7b.2 RandomizedSearchCV — ML Models ──────────────────────────────────────
print(f"RandomizedSearchCV — ML models ({N_ITER_ML} iters each)\\n")
for name, clf in BASELINE_ML.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_hpo, low_card_hpo, high_card_hpo)
    rs   = RandomizedSearchCV(pipe, RAND_ML[name], n_iter=N_ITER_ML,
                               scoring=HPO_SCORING, cv=CV_ML, n_jobs=-1,
                               refit=True, random_state=RANDOM_STATE, verbose=0)
    t0 = time.time()
    rs.fit(X_hpo, y_hpo)
    elapsed = time.time() - t0

    hpo_rand[name] = evaluate_cv(rs.best_estimator_, X_hpo, y_hpo)
    hpo_rand[name]['best_params'] = rs.best_params_
    print(f"done ({elapsed:.0f}s) | CV recall={rs.best_score_:.4f} "
          f"-> 5-fold recall={hpo_rand[name]['recall']:.4f}")
"""))

cells.append(code_cell(
"""# ── 7b.3 RandomizedSearchCV — DL Models ──────────────────────────────────────
print(f"RandomizedSearchCV — DL models ({N_ITER_DL} iters, 3-fold)\\n")
for name, clf in DL_HPO_CLF.items():
    print(f"  [{name}] ...", end=' ', flush=True)
    pipe = build_pipeline(clone(clf), num_hpo, low_card_hpo, high_card_hpo)
    rs   = RandomizedSearchCV(pipe, RAND_DL[name], n_iter=N_ITER_DL,
                               scoring=HPO_SCORING, cv=CV_DL, n_jobs=1,
                               refit=True, random_state=RANDOM_STATE, verbose=0)
    t0 = time.time()
    rs.fit(X_hpo, y_hpo)
    elapsed = time.time() - t0

    hpo_rand[name] = evaluate_cv(rs.best_estimator_, X_hpo, y_hpo)
    hpo_rand[name]['best_params'] = rs.best_params_
    print(f"done ({elapsed:.0f}s) | CV recall={rs.best_score_:.4f} "
          f"-> 5-fold recall={hpo_rand[name]['recall']:.4f}")

df_rand = results_to_df(hpo_rand, 'RandomizedSearchCV')
print("\\n── RandomizedSearchCV Results ───────────────────────────")
display(df_rand.set_index('Model').drop(columns='Scenario'))
"""))

# ── Optuna ────────────────────────────────────────────────────────────────────
cells.append(md_cell(
"""### 7c — Optuna (Bayesian Optimisation, TPE Sampler)

- ML models: **50 trials**, 5-fold CV per trial
- DL models: **30 trials**, 3-fold CV per trial (Keras cost)
- After search: best params evaluated with full 5-fold CV
"""))

cells.append(code_cell(
"""# ── 7c.1 Optuna Objective Functions — ML Models ───────────────────────────────

def make_ml_objective(model_name, X, y, num_cols, low_card, high_card, cv_splitter):
    \"\"\"Return an Optuna objective that optimises recall for the given ML model.\"\"\"
    def objective(trial):
        if model_name == 'LR':
            clf = LogisticRegression(
                C=trial.suggest_float('C', 1e-3, 1e2, log=True),
                penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                solver='liblinear', max_iter=1000,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        elif model_name == 'DT':
            clf = DecisionTreeClassifier(
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 100),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 50),
                criterion=trial.suggest_categorical('criterion', ['gini','entropy']),
                random_state=RANDOM_STATE,
            )
        elif model_name == 'RF':
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                max_features=trial.suggest_categorical('max_features', ['sqrt','log2']),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 30),
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        elif model_name == 'XGB':
            clf = XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric='logloss', verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe   = build_pipeline(clf, num_cols, low_card, high_card)
        scores = cross_val_score(pipe, X, y, cv=cv_splitter,
                                  scoring=HPO_SCORING, n_jobs=-1)
        return scores.mean()
    return objective


N_OPTUNA_ML = 50
print(f"Optuna — ML models ({N_OPTUNA_ML} trials, 5-fold)\\n")

for name in BASELINE_ML:
    print(f"  [{name}] ...", end=' ', flush=True)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    obj_fn = make_ml_objective(name, X_hpo, y_hpo,
                                num_hpo, low_card_hpo, high_card_hpo, CV_ML)
    t0 = time.time()
    study.optimize(obj_fn, n_trials=N_OPTUNA_ML, show_progress_bar=False)
    elapsed = time.time() - t0

    bp = study.best_params
    if name == 'LR':
        best_clf = LogisticRegression(C=bp['C'], penalty=bp['penalty'],
                                       solver='liblinear', max_iter=1000,
                                       random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'DT':
        best_clf = DecisionTreeClassifier(
            max_depth=bp['max_depth'],
            min_samples_split=bp['min_samples_split'],
            min_samples_leaf=bp['min_samples_leaf'],
            criterion=bp['criterion'], random_state=RANDOM_STATE)
    elif name == 'RF':
        best_clf = RandomForestClassifier(
            n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],
            max_features=bp['max_features'], min_samples_split=bp['min_samples_split'],
            random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'XGB':
        best_clf = XGBClassifier(
            n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],
            learning_rate=bp['learning_rate'], subsample=bp['subsample'],
            colsample_bytree=bp['colsample_bytree'],
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='logloss', verbosity=0)

    best_pipe = build_pipeline(best_clf, num_hpo, low_card_hpo, high_card_hpo)
    hpo_optuna[name] = evaluate_cv(best_pipe, X_hpo, y_hpo)
    hpo_optuna[name]['best_params'] = bp
    print(f"done ({elapsed:.0f}s) | best trial={study.best_value:.4f} "
          f"-> 5-fold recall={hpo_optuna[name]['recall']:.4f}")
"""))

cells.append(code_cell(
"""# ── 7c.2 Optuna Objective Functions — DL Models ───────────────────────────────
N_OPTUNA_DL = 30
print(f"Optuna — DL models ({N_OPTUNA_DL} trials, 3-fold)\\n")

def make_dl_objective(arch, X, y, num_cols, low_card, high_card, cv_splitter):
    \"\"\"Return Optuna objective for ANN or DNN.\"\"\"
    def objective(trial):
        if arch == 'ANN':
            u1  = trial.suggest_categorical('n_units_1', [32, 64, 128, 256])
            u2  = trial.suggest_categorical('n_units_2', [16, 32, 64])
            dr  = trial.suggest_float('dropout_rate', 0.1, 0.5)
            lr  = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            def _build(meta=None):
                return build_ann(meta=meta, n_units_1=u1, n_units_2=u2,
                                  dropout_rate=dr, learning_rate=lr)
        else:  # DNN
            nl  = trial.suggest_int('n_layers', 3, 5)
            bu  = trial.suggest_categorical('base_units', [64, 128, 256])
            dr  = trial.suggest_float('dropout_rate', 0.1, 0.5)
            lr  = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            def _build(meta=None):
                return build_dnn(meta=meta, n_layers=nl, base_units=bu,
                                  dropout_rate=dr, learning_rate=lr)

        clf  = KerasClassifier(model=_build, epochs=15, batch_size=512, verbose=0,
                                validation_split=0.1,
                                callbacks=[EarlyStopping(patience=3, verbose=0)],
                                random_state=RANDOM_STATE)
        pipe = build_pipeline(clf, num_cols, low_card, high_card)
        scores = cross_val_score(pipe, X, y, cv=cv_splitter,
                                  scoring=HPO_SCORING, n_jobs=1)
        return scores.mean()
    return objective


for arch in ['ANN', 'DNN']:
    print(f"  [{arch}] ...", end=' ', flush=True)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    obj_fn = make_dl_objective(arch, X_hpo, y_hpo,
                                num_hpo, low_card_hpo, high_card_hpo, CV_DL)
    t0 = time.time()
    study.optimize(obj_fn, n_trials=N_OPTUNA_DL, show_progress_bar=False)
    elapsed = time.time() - t0

    bp = study.best_params
    if arch == 'ANN':
        def _best_ann(meta=None, _bp=bp):
            return build_ann(meta=meta, n_units_1=_bp['n_units_1'],
                              n_units_2=_bp['n_units_2'],
                              dropout_rate=_bp['dropout_rate'],
                              learning_rate=_bp['learning_rate'])
        best_clf = KerasClassifier(model=_best_ann, epochs=20, batch_size=512,
                                    verbose=0, random_state=RANDOM_STATE)
    else:
        def _best_dnn(meta=None, _bp=bp):
            return build_dnn(meta=meta, n_layers=_bp['n_layers'],
                              base_units=_bp['base_units'],
                              dropout_rate=_bp['dropout_rate'],
                              learning_rate=_bp['learning_rate'])
        best_clf = KerasClassifier(model=_best_dnn, epochs=20, batch_size=512,
                                    verbose=0, random_state=RANDOM_STATE)

    best_pipe = build_pipeline(best_clf, num_hpo, low_card_hpo, high_card_hpo)
    hpo_optuna[arch] = evaluate_cv(best_pipe, X_hpo, y_hpo)
    hpo_optuna[arch]['best_params'] = bp
    print(f"done ({elapsed:.0f}s) | best trial={study.best_value:.4f} "
          f"-> 5-fold recall={hpo_optuna[arch]['recall']:.4f}")
"""))

cells.append(code_cell(
"""# ── 7d Best HPO Method per Model & Aggregate ─────────────────────────────────
print("Best HPO method per model (by Recall):\\n")

hpo_method_rows = []
for model in MODEL_NAMES:
    row = {'Model': model}
    best_recall = -1
    best_method = None

    for method_name, method_dict in [('GridSearch',  hpo_grid),
                                      ('RandomSearch', hpo_rand),
                                      ('Optuna',       hpo_optuna)]:
        if model in method_dict:
            r = method_dict[model]['recall']
            row[method_name] = round(r, 4)
            if r > best_recall:
                best_recall = r
                best_method = method_name
                BEST_HPO[model] = {
                    'method':      method_name,
                    'recall':      r,
                    'metrics':     method_dict[model],
                    'best_params': method_dict[model].get('best_params', {}),
                }
        else:
            row[method_name] = float('nan')

    row['Best_Method'] = best_method
    row['Best_Recall'] = round(best_recall, 4)
    hpo_method_rows.append(row)

df_hpo_compare = pd.DataFrame(hpo_method_rows).set_index('Model')
display(df_hpo_compare)

# Append HPO results to global store
for label, method_dict in [('Scenario 3 — GridSearchCV',   hpo_grid),
                             ('Scenario 3 — RandomSearch',   hpo_rand),
                             ('Scenario 3 — Optuna',         hpo_optuna)]:
    ALL_RESULTS.append(results_to_df(method_dict, label))
"""))

# ── SECTION 8: COMPARISON ─────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 8 — Comprehensive Model Comparison

Aggregates results from all scenarios and all HPO strategies into a single
comparison view. Champion model is the one with highest Recall in its best HPO configuration.
"""))

cells.append(code_cell(
"""# ── 8.1 Master Comparison Table ───────────────────────────────────────────────
df_all = pd.concat(ALL_RESULTS, ignore_index=True)

# Recall pivot: rows = models, columns = scenarios
recall_pivot = df_all.pivot_table(index='Model', columns='Scenario', values='Recall')
print("Recall across all scenarios:")
display(recall_pivot.round(4))
"""))

cells.append(code_cell(
"""# ── 8.2 Full Metrics Table ────────────────────────────────────────────────────
print("Complete metrics for all scenarios and models:")
display(df_all.round(4))
"""))

cells.append(code_cell(
"""# ── 8.3 Visualisations ────────────────────────────────────────────────────────
SCENARIOS_ORDERED = [
    'Scenario 1', 'Scenario 2',
    'Scenario 3 — GridSearchCV',
    'Scenario 3 — RandomSearch',
    'Scenario 3 — Optuna',
]
METRICS_PLOT = ['Recall', 'F1', 'Accuracy', 'Precision']
COLORS = sns.color_palette('husl', len(MODEL_NAMES))
BAR_W  = 0.13
X_POS  = np.arange(len(SCENARIOS_ORDERED))

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for ax_i, metric in enumerate(METRICS_PLOT):
    ax = axes[ax_i]
    for m_i, model in enumerate(MODEL_NAMES):
        vals = []
        for sc in SCENARIOS_ORDERED:
            sub = df_all[(df_all['Model'] == model) & (df_all['Scenario'] == sc)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else np.nan)
        offset = (m_i - len(MODEL_NAMES) / 2) * BAR_W + BAR_W / 2
        ax.bar(X_POS + offset, vals, width=BAR_W, label=model,
               color=COLORS[m_i], alpha=0.85, edgecolor='white')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_xticks(X_POS)
    ax.set_xticklabels([s.replace(' — ', '\\n') for s in SCENARIOS_ORDERED],
                        fontsize=8, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)

plt.suptitle('Model Performance Across All Scenarios', fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 8.4 Timing Comparison (Scenario 2 baseline) ──────────────────────────────
df_s2_disp = df_all[df_all['Scenario'] == 'Scenario 2']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax_i, col in enumerate(['Train_Time_s', 'Infer_Time_s']):
    bars = axes[ax_i].bar(df_s2_disp['Model'], df_s2_disp[col],
                           color=COLORS, alpha=0.85, edgecolor='white')
    axes[ax_i].set_title(col.replace('_', ' ') + ' (Scenario 2)',
                          fontsize=12, fontweight='bold')
    axes[ax_i].set_ylabel('Seconds')
    for bar, val in zip(bars, df_s2_disp[col]):
        axes[ax_i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() * 1.02,
                         f'{val:.2f}s', ha='center', fontsize=9)

plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 8.5 HPO Method Comparison Heatmap ────────────────────────────────────────
hpo_rows = df_all[df_all['Scenario'].str.startswith('Scenario 3')]
hpo_recall_pivot = hpo_rows.pivot_table(index='Model', columns='Scenario', values='Recall')

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(hpo_recall_pivot, annot=True, fmt='.4f', cmap='YlGn',
            linewidths=0.5, ax=ax, annot_kws={'size': 11})
ax.set_title('HPO Method Comparison — Recall per Model', fontsize=13, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.show()
"""))

# ── SECTION 9: SHAP ───────────────────────────────────────────────────────────
cells.append(md_cell(
"""---
## Section 9 — SHAP Analysis on Champion Model

**Champion model**: highest mean Recall across all HPO configurations.

SHAP (SHapley Additive exPlanations) provides theoretically grounded,
model-agnostic feature attribution.

| Plot | Purpose |
|:---|:---|
| **Beeswarm summary** | Global importance + direction of effect per feature |
| **Bar chart** | Mean \\|SHAP\\| ranking |
| **Dependence plots** | Marginal effect of top-3 features |
| **Waterfall** | Local explanation for a single high-risk prediction |
"""))

cells.append(code_cell(
"""# ── 9.1 Identify Champion Model ───────────────────────────────────────────────
champ_rows = []
for model, info in BEST_HPO.items():
    champ_rows.append({
        'Model':   model,
        'Method':  info['method'],
        'Recall':  info['recall'],
        'F1':      info['metrics']['f1'],
        'Accuracy':info['metrics']['accuracy'],
    })

df_champ = pd.DataFrame(champ_rows).sort_values('Recall', ascending=False)
display(df_champ.set_index('Model'))

CHAMPION_MODEL  = df_champ.iloc[0]['Model']
CHAMPION_METHOD = df_champ.iloc[0]['Method']
CHAMPION_RECALL = df_champ.iloc[0]['Recall']
CHAMPION_PARAMS = BEST_HPO[CHAMPION_MODEL]['best_params']

print(f"\\nChampion : {CHAMPION_MODEL}  tuned by {CHAMPION_METHOD}")
print(f"Recall   : {CHAMPION_RECALL:.4f}")
"""))

cells.append(code_cell(
"""# ── 9.2 Rebuild Champion Pipeline for SHAP ────────────────────────────────────
# Train/test split from the HPO subsample for SHAP computation
X_tr_shap, X_te_shap, y_tr_shap, y_te_shap = train_test_split(
    X_hpo, y_hpo, test_size=0.2, stratify=y_hpo, random_state=RANDOM_STATE
)

bp = CHAMPION_PARAMS

if CHAMPION_MODEL == 'LR':
    champ_clf = LogisticRegression(C=bp.get('C', 1.0),
                                    penalty=bp.get('penalty', 'l2'),
                                    solver='liblinear', max_iter=1000,
                                    random_state=RANDOM_STATE, n_jobs=-1)
elif CHAMPION_MODEL == 'DT':
    champ_clf = DecisionTreeClassifier(
        max_depth=bp.get('max_depth', None),
        min_samples_split=bp.get('min_samples_split', 2),
        criterion=bp.get('criterion', 'gini'),
        random_state=RANDOM_STATE)
elif CHAMPION_MODEL == 'RF':
    champ_clf = RandomForestClassifier(
        n_estimators=bp.get('n_estimators', 100),
        max_depth=bp.get('max_depth', None),
        max_features=bp.get('max_features', 'sqrt'),
        random_state=RANDOM_STATE, n_jobs=-1)
elif CHAMPION_MODEL == 'XGB':
    champ_clf = XGBClassifier(
        n_estimators=bp.get('n_estimators', 100),
        max_depth=bp.get('max_depth', 6),
        learning_rate=bp.get('learning_rate', 0.1),
        subsample=bp.get('subsample', 1.0),
        colsample_bytree=bp.get('colsample_bytree', 1.0),
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='logloss', verbosity=0)
elif CHAMPION_MODEL == 'ANN':
    def _champ_ann(meta=None, _bp=bp):
        return build_ann(meta=meta,
                          n_units_1=_bp.get('n_units_1', 64),
                          n_units_2=_bp.get('n_units_2', 32),
                          dropout_rate=_bp.get('dropout_rate', 0.3),
                          learning_rate=_bp.get('learning_rate', 0.001))
    champ_clf = KerasClassifier(model=_champ_ann, epochs=20, batch_size=512,
                                 verbose=0, random_state=RANDOM_STATE)
elif CHAMPION_MODEL == 'DNN':
    def _champ_dnn(meta=None, _bp=bp):
        return build_dnn(meta=meta,
                          n_layers=_bp.get('n_layers', 4),
                          base_units=_bp.get('base_units', 128),
                          dropout_rate=_bp.get('dropout_rate', 0.3),
                          learning_rate=_bp.get('learning_rate', 0.001))
    champ_clf = KerasClassifier(model=_champ_dnn, epochs=20, batch_size=512,
                                 verbose=0, random_state=RANDOM_STATE)

champ_pipe = build_pipeline(champ_clf, num_hpo, low_card_hpo, high_card_hpo)
champ_pipe.fit(X_tr_shap, y_tr_shap)

y_pred_shap = champ_pipe.predict(X_te_shap)
print(f"Champion holdout  Recall : {recall_score(y_te_shap, y_pred_shap):.4f}")
print(f"Champion holdout  F1     : {f1_score(y_te_shap, y_pred_shap):.4f}")
print()
print(classification_report(y_te_shap, y_pred_shap,
                             target_names=['No Churn', 'Churn']))
"""))

cells.append(code_cell(
"""# ── 9.3 Prepare SHAP Inputs ───────────────────────────────────────────────────
# Extract fitted preprocessor and transform the test set to numeric form
preproc_fitted = champ_pipe.named_steps['preprocessor']
X_te_proc      = preproc_fitted.transform(X_te_shap)

# Recover feature names from ColumnTransformer
try:
    feat_names = preproc_fitted.get_feature_names_out()
    feat_names = [str(f).replace('num__','').replace('ohe__','').replace('te__','')
                  for f in feat_names]
except Exception:
    feat_names = [f'feature_{i}' for i in range(X_te_proc.shape[1])]

# Sample for SHAP (speed)
SHAP_N = min(2000, len(X_te_proc))
shap_idx  = np.random.choice(len(X_te_proc), SHAP_N, replace=False)
X_shap    = X_te_proc[shap_idx]

inner_model = champ_pipe.named_steps['classifier']

# Select SHAP explainer based on champion model family
if CHAMPION_MODEL in ('RF', 'DT', 'XGB'):
    explainer   = shap.TreeExplainer(inner_model)
    shap_values = explainer.shap_values(X_shap)
    # Handle both list format [neg, pos] and 3-D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    expected_val = (explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value)

elif CHAMPION_MODEL == 'LR':
    explainer   = shap.LinearExplainer(inner_model, X_shap)
    shap_values = explainer.shap_values(X_shap)
    expected_val = explainer.expected_value

else:  # ANN / DNN — KernelExplainer on a small background
    background   = shap.kmeans(X_shap, 50)
    explainer    = shap.KernelExplainer(
        lambda x: inner_model.predict_proba(x)[:, 1],
        background
    )
    SHAP_N       = min(200, SHAP_N)    # KernelExplainer is slow; keep sample small
    X_shap       = X_shap[:SHAP_N]
    shap_values  = explainer.shap_values(X_shap)
    expected_val = explainer.expected_value

print(f"Explainer type : {type(explainer).__name__}")
print(f"SHAP values shape : {np.array(shap_values).shape}")
"""))

cells.append(code_cell(
"""# ── 9.4 SHAP Summary Plots ────────────────────────────────────────────────────

# Beeswarm (dot) — shows direction of effect
shap.summary_plot(shap_values, X_shap, feature_names=feat_names,
                   plot_type='dot', max_display=20, show=False)
plt.title(f'SHAP Summary — {CHAMPION_MODEL} (champion by Recall)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Bar — mean |SHAP| global importance
shap.summary_plot(shap_values, X_shap, feature_names=feat_names,
                   plot_type='bar', max_display=20, show=False)
plt.title(f'Mean |SHAP| Feature Importance — {CHAMPION_MODEL}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 9.5 SHAP Dependence Plots — Top 3 Features ───────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top3_idx      = np.argsort(mean_abs_shap)[::-1][:3]
top3_names    = [feat_names[i] for i in top3_idx]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_i, (fi, fn) in enumerate(zip(top3_idx, top3_names)):
    shap.dependence_plot(fi, shap_values, X_shap,
                          feature_names=feat_names,
                          ax=axes[ax_i], show=False)
    axes[ax_i].set_title(f'Dependence: {fn}', fontsize=11, fontweight='bold')

plt.suptitle(f'SHAP Dependence Plots — Top 3 Features ({CHAMPION_MODEL})',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(code_cell(
"""# ── 9.6 SHAP Waterfall — Highest-Risk Prediction ─────────────────────────────
pred_proba  = inner_model.predict_proba(X_shap)[:, 1]
high_risk_i = int(np.argmax(pred_proba))

shap.waterfall_plot(
    shap.Explanation(
        values       = shap_values[high_risk_i],
        base_values  = expected_val,
        data         = X_shap[high_risk_i],
        feature_names= feat_names,
    ),
    max_display=15,
    show=False,
)
plt.title(f'SHAP Waterfall — Highest Risk Prediction ({CHAMPION_MODEL})',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ── 9.7 Top Feature Importance Table ─────────────────────────────────────────
importance_df = (
    pd.DataFrame({'Feature': feat_names,
                  'Mean |SHAP|': mean_abs_shap})
    .sort_values('Mean |SHAP|', ascending=False)
    .head(20)
    .reset_index(drop=True)
)
print(f"\\nTop 20 SHAP Feature Importances — {CHAMPION_MODEL} ({CHAMPION_METHOD})")
display(importance_df.round(5))

print("\\n" + "="*60)
print(f"Study complete.")
print(f"Champion model  : {CHAMPION_MODEL}")
print(f"HPO method      : {CHAMPION_METHOD}")
print(f"Best Recall     : {CHAMPION_RECALL:.4f}")
print("="*60)
"""))

# ── ASSEMBLE & WRITE ──────────────────────────────────────────────────────────
notebook = {
    "cells":    cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language":     "python",
            "name":         "python3"
        },
        "language_info": {
            "name":    "python",
            "version": "3.10.0"
        }
    },
    "nbformat":       4,
    "nbformat_minor": 5,
}

OUT = "churn_prediction_analysis.ipynb"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"Notebook written -> {OUT}")
print(f"  Cells : {len(cells)}")
