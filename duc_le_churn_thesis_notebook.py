# %% [markdown]
# # Advanced Predictive Modelling for Customer Churn in Banking
# 
# This notebook implements the empirical part of Duc Le's master thesis proposal using the updated
# `customer_churn_1M.csv` dataset.
# 
# The research goal is to determine whether machine learning models
# (`Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`) and deep learning models
# (`ANN`, `DNN`) differ meaningfully in predictive performance, and whether any performance gain
# is worth the execution-time trade-off.
# 
# Core methodological safeguards:
# 
# - The old 5,000-row proposal dataset is not used.
# - Feature engineering is based on EDA from the updated 1M-row dataset.
# - The final holdout test set is not used for preprocessing, encoding, resampling, tuning, or model selection.
# - Target encoding is fitted inside training folds only.
# - SMOTE is applied inside training folds only through `imblearn.pipeline.Pipeline`.
# - Class imbalance is handled with class-weighted/cost-sensitive learning, and SMOTE is available as a leakage-safe sensitivity analysis.
# - Warnings and noisy TensorFlow logs are hidden to keep the notebook readable.

# %% [markdown]
# ## 1. Environment Setup
# 
# This cell checks the current Python environment and installs only pinned, mutually compatible
# package versions if something is missing. The pins are based on the local environment checked
# before generating this notebook: Python 3.12.7, pandas 2.2.2, NumPy 1.26.4, scikit-learn 1.8.0,
# imbalanced-learn 0.14.1, XGBoost 3.2.0, TensorFlow 2.16.2, Keras 3.14.0, Optuna 4.8.0,
# SHAP 0.46.0.
# 
# If all packages are already present with the expected versions, no installation is performed.

# %%
# Environment and package installation script.
# This cell is intentionally first so later cells run with a known-compatible stack.

import importlib.metadata as importlib_metadata
import os
import subprocess
import sys
import warnings

# Hide warnings and verbose framework logs across the notebook.
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")

# Exact versions verified in the current Anaconda Python 3.12 environment.
PINNED_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scikit-learn": "1.8.0",
    "imbalanced-learn": "0.14.1",
    "xgboost": "3.2.0",
    "tensorflow": "2.16.2",
    "keras": "3.14.0",
    "optuna": "4.8.0",
    "shap": "0.46.0",
    "seaborn": "0.13.2",
    "matplotlib": "3.9.2",
    "scipy": "1.13.1",
    "statsmodels": "0.14.2",
    "nbformat": "5.10.4",
    "ipykernel": "6.28.0",
}

INSTALL_IF_NEEDED = True


def get_installed_version(package_name: str) -> str | None:
    '''Return installed version or None if the package is missing.'''
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


package_status = []
packages_to_install = []

for package_name, expected_version in PINNED_PACKAGES.items():
    installed_version = get_installed_version(package_name)
    package_status.append(
        {
            "package": package_name,
            "expected": expected_version,
            "installed": installed_version or "MISSING",
            "ok": installed_version == expected_version,
        }
    )
    if installed_version != expected_version:
        packages_to_install.append(f"{package_name}=={expected_version}")

if packages_to_install and INSTALL_IF_NEEDED:
    print("Installing compatible package versions:")
    for package_spec in packages_to_install:
        print(f"  - {package_spec}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages_to_install]
    )
    print("Install complete. Restart the kernel before continuing if imports fail.")
else:
    print("All required packages are already installed with compatible versions.")

# %% [markdown]
# Interpretation:
# 
# The table created in this cell verifies reproducibility. If packages were installed, restart the
# kernel once before continuing so Python imports the newly installed versions.

# %% [markdown]
# ## 2. Imports and Global Configuration
# 
# This section imports the full modelling stack and defines reproducibility and runtime controls.
# Full-data EDA always uses all 1,000,000 rows. Model training can be run on the full data by setting
# `MODEL_SAMPLE_SIZE = None`; the default sample keeps development runs practical on a laptop.

# %%
# Core runtime controls.
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

tf.get_logger().setLevel("ERROR")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reproducibility.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Set this to None for final full-data thesis runs.
MODEL_SAMPLE_SIZE = 200_000
HPO_SAMPLE_SIZE = 80_000
SMOTE_SAMPLE_SIZE = 80_000

# Keep these modest for a first complete pass; increase for final experiments.
CV_SPLITS = 3
DL_EPOCHS = 12
DL_BATCH_SIZE = 2048

RUN_HPO = True
RUN_SMOTE_SENSITIVITY = True
RUN_DEEP_LEARNING = True
RUN_SHAP = True

# A small smoke-test mode can be enabled before running the notebook:
# os.environ["CHURN_SMOKE_TEST"] = "1"
SMOKE_TEST = os.environ.get("CHURN_SMOKE_TEST") == "1"
if SMOKE_TEST:
    MODEL_SAMPLE_SIZE = 5_000
    HPO_SAMPLE_SIZE = 2_000
    SMOTE_SAMPLE_SIZE = 2_000
    CV_SPLITS = 2
    DL_EPOCHS = 2
    RUN_HPO = False
    RUN_SMOTE_SENSITIVITY = True
    RUN_DEEP_LEARNING = True
    RUN_SHAP = False

DATA_PATH = Path("customer_churn_1M.csv")
TARGET = "churn"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook")

print(f"TensorFlow version: {tf.__version__}")
print(f"Notebook data path exists: {DATA_PATH.exists()}")

# %% [markdown]
# Interpretation:
# 
# The configuration separates thesis logic from runtime budget. EDA is not sampled. Only modelling
# sample sizes are configurable so the same notebook can be used both for quick validation and final
# full-data runs.

# %% [markdown]
# ## 3. Data Loading and Schema Check
# 
# This section loads the updated CSV, checks shape, column types, duplicated rows, unique customer
# IDs, and the target variable. This confirms that the proposal appendix is outdated and that the
# notebook should adapt to the actual 1M-row schema.

# %%
# Load the full updated dataset for EDA.
df = pd.read_csv(DATA_PATH)

# Parse dates after loading so invalid date values can be counted explicitly.
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")
print(f"Duplicated full rows: {df.duplicated().sum():,}")
print(f"Unique customer_id values: {df['customer_id'].nunique():,}")
print(f"Invalid signup_date values: {df['signup_date'].isna().sum():,}")

display(df.head())

schema = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=True),
        "missing": df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100).round(3),
    }
).sort_values(["missing_pct", "n_unique"], ascending=[False, False])

display(schema)

# %% [markdown]
# Interpretation:
# 
# The dataset has 1,000,000 customer-level observations and 32 columns. `customer_id` is unique and
# must not be used as a predictive feature. `signup_date` is unique at timestamp level, so the raw
# timestamp is too high-cardinality for direct modelling; it will be converted into cohort features.
# Missingness is modest and concentrated in income, satisfaction, complaints, internet usage, and
# credit score, so median imputation plus missingness indicators is appropriate.

# %% [markdown]
# ## 4. Exploratory Data Analysis
# 
# EDA focuses on data quality, class imbalance, categorical churn rates, numerical churn signals,
# and outliers. These findings determine the feature engineering in the next section.

# %%
# Target distribution and imbalance ratio.
target_counts = df[TARGET].value_counts().sort_index()
target_pct = df[TARGET].value_counts(normalize=True).sort_index().mul(100)
target_summary = pd.DataFrame({"count": target_counts, "percent": target_pct.round(3)})
display(target_summary)

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=target_summary.index.astype(str), y=target_summary["percent"], ax=ax)
ax.set_title("Churn Class Distribution")
ax.set_xlabel("Churn label")
ax.set_ylabel("Percent of customers")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%")
plt.show()

# %% [markdown]
# Interpretation:
# 
# The churn class is imbalanced at roughly 10% churn and 90% non-churn. Accuracy alone would be
# misleading, so recall, F1, AUC, average precision, and balanced accuracy are reported. Imbalance is
# handled inside the training workflow, never by resampling the whole dataset before splitting.

# %%
# Missing-value overview.
missing = (
    pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100),
        }
    )
    .query("missing_count > 0")
    .sort_values("missing_pct", ascending=False)
)
display(missing.round(3))

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=missing.reset_index(), x="missing_pct", y="index", ax=ax)
ax.set_title("Missing Values by Column")
ax.set_xlabel("Missing percent")
ax.set_ylabel("")
plt.show()

# %% [markdown]
# Interpretation:
# 
# Missing values are not widespread enough to justify dropping rows. Because the variables are
# financial and behavioural, median imputation is safer than mean imputation for skewed numeric
# features. Missingness indicators are added for columns where being missing may itself be
# informative.

# %%
# Categorical churn rates.
categorical_cols = [
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method",
    "paperless_billing",
]

cat_rate_tables = {}
for col in categorical_cols:
    table = (
        df.groupby(col, dropna=False)[TARGET]
        .agg(count="size", churn_rate="mean")
        .assign(churn_rate_pct=lambda x: x["churn_rate"].mul(100))
        .sort_values("churn_rate_pct", ascending=False)
    )
    cat_rate_tables[col] = table
    print(f"\n{col}")
    display(table[["count", "churn_rate_pct"]].round(3))

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
for ax, col in zip(axes, categorical_cols):
    plot_df = cat_rate_tables[col].reset_index()
    sns.barplot(data=plot_df, x=col, y="churn_rate_pct", ax=ax)
    ax.set_title(f"Churn Rate by {col}")
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.show()

# %% [markdown]
# Interpretation:
# 
# Contract type is the strongest categorical signal: month-to-month customers churn far more often
# than one-year or two-year customers. Demographic categories have much smaller churn-rate
# differences, so they are retained but not expected to dominate the models.

# %%
# Numerical summary and correlation with churn.
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET).tolist()

numeric_summary = df[numeric_cols].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T
display(numeric_summary.round(3))

target_corr = (
    df[numeric_cols]
    .corrwith(df[TARGET])
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_frame("corr_with_churn")
)
display(target_corr.round(4))

fig, ax = plt.subplots(figsize=(8, 9))
sns.barplot(
    data=target_corr.reset_index(),
    y="index",
    x="corr_with_churn",
    ax=ax,
)
ax.set_title("Linear Correlation with Churn")
ax.set_xlabel("Pearson correlation")
ax.set_ylabel("")
plt.show()

# %% [markdown]
# Interpretation:
# 
# Linear correlations are moderate, which supports testing non-linear models. Satisfaction is
# negatively associated with churn, while complaints, service calls, and late payments are positively
# associated. These patterns motivate engineered features that combine dissatisfaction, support
# contact intensity, payment behaviour, and account tenure.

# %%
# Binned churn rates for selected high-signal variables.
bin_specs = {
    "customer_satisfaction": [0, 2, 4, 6, 8, 10],
    "num_complaints": [-1, 0, 1, 2, 8],
    "num_service_calls": [-1, 0, 1, 3, 6, 13],
    "late_payments": [-1, 0, 1, 2, 6],
    "tenure": [0, 3, 6, 12, 24, 48, 73],
    "days_since_last_interaction": [0, 7, 30, 60, 90, 180, 366],
}

binned_tables = []
for col, bins in bin_specs.items():
    tmp = df[[col, TARGET]].copy()
    tmp[f"{col}_bin"] = pd.cut(tmp[col], bins=bins, include_lowest=True)
    table = (
        tmp.groupby(f"{col}_bin", observed=False)[TARGET]
        .agg(count="size", churn_rate="mean")
        .reset_index()
    )
    table["feature"] = col
    table["bin"] = table[f"{col}_bin"].astype(str)
    table["churn_rate_pct"] = table["churn_rate"].mul(100)
    binned_tables.append(table[["feature", "bin", "count", "churn_rate_pct"]])

binned_df = pd.concat(binned_tables, ignore_index=True)
display(binned_df.round(3))

fig, axes = plt.subplots(2, 3, figsize=(17, 8))
axes = axes.ravel()
for ax, col in zip(axes, bin_specs):
    plot_df = binned_df[binned_df["feature"] == col]
    sns.barplot(data=plot_df, x="bin", y="churn_rate_pct", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Interpretation:
# 
# The binned plots show monotonic churn-risk patterns for dissatisfaction, complaints, service calls,
# late payments, and tenure. These are good candidates for interaction features such as complaint
# pressure, service-contact intensity, and late-payment rate per tenure month.

# %% [markdown]
# ## 5. Leakage-Safe Methodology
# 
# The main leakage risks in this study are:
# 
# - Learning imputers, scalers, encoders, or synthetic samples from the full dataset before splitting.
# - Target encoding high-cardinality categories outside the cross-validation fold.
# - Applying SMOTE before train-test split or before cross-validation.
# - Using unique IDs or raw timestamps as model inputs.
# 
# The notebook prevents these issues by building all preprocessing steps inside scikit-learn or
# imbalanced-learn pipelines. `TargetEncoder` uses internal cross-fitting during `fit_transform`, and
# SMOTE is placed inside an `imblearn.pipeline.Pipeline`, so it runs only on the training portion of
# each fold. These choices follow scikit-learn's target encoding guidance and imbalanced-learn's
# data leakage recommendations.

# %%
# Stratified final holdout split.
# The test set is kept untouched until final model evaluation.

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Train rows: {len(X_train_full):,}")
print(f"Test rows : {len(X_test):,}")
print("Train churn rate:", round(y_train_full.mean() * 100, 3), "%")
print("Test churn rate :", round(y_test.mean() * 100, 3), "%")

# %% [markdown]
# Interpretation:
# 
# The final test set keeps the natural churn distribution. It is not balanced, encoded, imputed, or
# used for model selection. This gives a realistic estimate of deployment performance.

# %% [markdown]
# ## 6. Feature Engineering
# 
# Feature engineering is based on the observed 1M-row data rather than the outdated proposal
# appendix. The raw ID is dropped. The raw timestamp is transformed into interpretable cohort
# variables. Behavioural and financial ratios are added because EDA showed churn is linked to
# contract type, dissatisfaction, complaints, service calls, late payments, and tenure.

# %%
class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    '''Create EDA-driven features without using the target variable.'''

    def fit(self, X, y=None):
        # No statistics are learned here, so there is no target or fold leakage.
        return self

    def transform(self, X):
        X = X.copy()

        # Convert signup timestamp into cohort features; drop the raw high-cardinality timestamp later.
        signup = pd.to_datetime(X["signup_date"], errors="coerce")
        X["signup_year"] = signup.dt.year.astype("Int64").astype(str).replace("<NA>", "missing")
        X["signup_month"] = signup.dt.month.astype("Int64").astype(str).replace("<NA>", "missing")
        X["signup_quarter"] = signup.dt.quarter.astype("Int64").astype(str).replace("<NA>", "missing")
        X["signup_year_month"] = signup.dt.to_period("M").astype(str).replace("NaT", "missing")

        # Safe denominators prevent divide-by-zero errors and leave NaNs for the imputer.
        tenure_safe = X["tenure"].replace(0, np.nan)
        income_safe = X["annual_income"].replace(0, np.nan)

        # Financial burden and charge consistency features.
        X["charges_per_tenure"] = X["totalcharges"] / tenure_safe
        X["monthly_charge_gap"] = X["monthlycharges"] - X["charges_per_tenure"]
        X["annualized_charge_income_ratio"] = (X["monthlycharges"] * 12) / income_safe
        X["total_charge_income_ratio"] = X["totalcharges"] / income_safe

        # Complaint and support intensity features.
        X["complaint_rate_per_call"] = X["num_complaints"] / (X["num_service_calls"] + 1)
        X["support_contact_intensity"] = X["num_service_calls"] / tenure_safe
        X["late_payment_rate"] = X["late_payments"] / tenure_safe
        X["recency_tenure_ratio"] = X["days_since_last_interaction"] / (tenure_safe * 30)

        # Service bundle features.
        service_cols = [
            "has_phone_service",
            "has_internet_service",
            "has_online_security",
            "has_online_backup",
            "has_device_protection",
            "has_tech_support",
            "has_streaming_tv",
            "has_streaming_movies",
        ]
        X["enabled_service_count"] = X[service_cols].sum(axis=1)
        X["protection_support_count"] = X[
            ["has_online_security", "has_online_backup", "has_device_protection", "has_tech_support"]
        ].sum(axis=1)
        X["streaming_count"] = X[["has_streaming_tv", "has_streaming_movies"]].sum(axis=1)
        X["service_adoption_ratio"] = X["num_services"] / 6

        # Dissatisfaction interaction features.
        X["satisfaction_risk"] = 10 - X["customer_satisfaction"]
        X["complaint_satisfaction_pressure"] = X["num_complaints"] * X["satisfaction_risk"]
        X["low_satisfaction_flag"] = (X["customer_satisfaction"] <= 4).astype(float)
        X["complaint_flag"] = (X["num_complaints"] > 0).astype(float)
        X["low_satisfaction_with_complaint"] = (
            X["low_satisfaction_flag"] * X["complaint_flag"]
        )

        # Missingness indicators preserve potential information from missing values.
        for col in [
            "annual_income",
            "customer_satisfaction",
            "num_complaints",
            "avg_monthly_gb",
            "credit_score",
        ]:
            X[f"{col}_was_missing"] = X[col].isna().astype(int)

        # Drop direct identifiers and raw timestamp.
        return X.drop(columns=["customer_id", "signup_date"], errors="ignore")

# %% [markdown]
# Interpretation:
# 
# This transformer uses only information available in each row. It does not compute target rates,
# global means, or dataset-level statistics. That keeps engineered features safe for cross-validation
# and final holdout testing.

# %%
# Feature groups after feature engineering.
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "dependents",
    "tenure",
    "senior_citizen",
    "monthlycharges",
    "totalcharges",
    "num_services",
    "has_phone_service",
    "has_internet_service",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
    "customer_satisfaction",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "avg_monthly_gb",
    "days_since_last_interaction",
    "credit_score",
    "charges_per_tenure",
    "monthly_charge_gap",
    "annualized_charge_income_ratio",
    "total_charge_income_ratio",
    "complaint_rate_per_call",
    "support_contact_intensity",
    "late_payment_rate",
    "recency_tenure_ratio",
    "enabled_service_count",
    "protection_support_count",
    "streaming_count",
    "service_adoption_ratio",
    "satisfaction_risk",
    "complaint_satisfaction_pressure",
    "low_satisfaction_flag",
    "complaint_flag",
    "low_satisfaction_with_complaint",
    "annual_income_was_missing",
    "customer_satisfaction_was_missing",
    "num_complaints_was_missing",
    "avg_monthly_gb_was_missing",
    "credit_score_was_missing",
]

LOW_CARD_CATEGORICAL_FEATURES = [
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method",
    "paperless_billing",
    "signup_year",
    "signup_month",
    "signup_quarter",
]

# Target-encoded feature. The raw timestamp is not encoded; the monthly cohort is.
HIGH_CARD_CATEGORICAL_FEATURES = ["signup_year_month"]


def make_column_transformer():
    '''Create the preprocessing column transformer used inside every model pipeline.'''
    numeric_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    low_card_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_card_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encoder",
                TargetEncoder(
                    target_type="binary",
                    smooth="auto",
                    cv=CV_SPLITS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("low_card", low_card_pipe, LOW_CARD_CATEGORICAL_FEATURES),
            ("target_encoded", high_card_pipe, HIGH_CARD_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_preprocess_steps():
    '''Return top-level steps so imblearn Pipeline does not receive a nested Pipeline.'''
    return [
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocess", make_column_transformer()),
    ]


preprocess_preview = SkPipeline(make_preprocess_steps())
sample_preview = preprocess_preview.fit_transform(X_train_full.head(2_000), y_train_full.head(2_000))
print("Preview transformed shape:", sample_preview.shape)

# %% [markdown]
# Interpretation:
# 
# The preview confirms that the custom feature engineering and column transformer produce a numeric
# matrix suitable for all ML and DL models. Target encoding is part of the transformer and will be
# re-fitted within each fold.

# %% [markdown]
# ## 7. Modelling Samples
# 
# The full training set is available for final thesis runs. For iterative notebook development, a
# stratified modelling sample reduces runtime while preserving the churn ratio.

# %%
def stratified_sample(X_data, y_data, sample_size, random_state=RANDOM_STATE):
    '''Return a stratified sample, or the original data if sample_size is None.'''
    if sample_size is None or sample_size >= len(X_data):
        return X_data.copy(), y_data.copy()

    _, X_sample, _, y_sample = train_test_split(
        X_data,
        y_data,
        test_size=sample_size,
        stratify=y_data,
        random_state=random_state,
    )
    return X_sample.copy(), y_sample.copy()


X_model, y_model = stratified_sample(X_train_full, y_train_full, MODEL_SAMPLE_SIZE)
X_hpo, y_hpo = stratified_sample(X_train_full, y_train_full, HPO_SAMPLE_SIZE)
X_smote, y_smote = stratified_sample(X_train_full, y_train_full, SMOTE_SAMPLE_SIZE)

print(f"Model sample rows: {len(X_model):,} | churn rate: {y_model.mean() * 100:.3f}%")
print(f"HPO sample rows  : {len(X_hpo):,} | churn rate: {y_hpo.mean() * 100:.3f}%")
print(f"SMOTE rows       : {len(X_smote):,} | churn rate: {y_smote.mean() * 100:.3f}%")

# %% [markdown]
# Interpretation:
# 
# All modelling subsets are drawn only from the training partition and preserve the churn ratio.
# The final test set remains untouched.

# %% [markdown]
# ## 8. Evaluation Utilities
# 
# This section defines reusable evaluation functions. Each fold fits a fresh pipeline so preprocessing,
# target encoding, and optional SMOTE are learned only from the fold's training data.

# %%
def calculate_scale_pos_weight(y_data):
    '''Return negative-to-positive class ratio for XGBoost.'''
    positives = int((y_data == 1).sum())
    negatives = int((y_data == 0).sum())
    return negatives / max(positives, 1)


def build_ml_models(y_reference):
    '''Create the four ML models with class imbalance handling.'''
    xgb_weight = calculate_scale_pos_weight(y_reference)

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=xgb_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def make_ml_pipeline(model, use_smote=False):
    '''Build one complete pipeline with optional leakage-safe SMOTE.'''
    steps = make_preprocess_steps()
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=0.35,
                    k_neighbors=5,
                    random_state=RANDOM_STATE,
                ),
            )
        )
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def predict_positive_probability(fitted_model, X_data):
    '''Return positive-class probabilities for estimators with either predict_proba or decision_function.'''
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X_data)[:, 1]
    scores = fitted_model.decision_function(X_data)
    return 1 / (1 + np.exp(-scores))


def metric_dict(y_true, y_pred, y_proba):
    '''Compute churn metrics with zero-division protection.'''
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }


def evaluate_pipeline_cv(name, pipeline, X_data, y_data, cv_splits=CV_SPLITS):
    '''Evaluate a sklearn/imblearn pipeline with timed stratified CV.'''
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_data, y_data), start=1):
        X_train_fold = X_data.iloc[train_idx]
        X_valid_fold = X_data.iloc[valid_idx]
        y_train_fold = y_data.iloc[train_idx]
        y_valid_fold = y_data.iloc[valid_idx]

        fitted = clone(pipeline)

        fit_start = time.perf_counter()
        fitted.fit(X_train_fold, y_train_fold)
        fit_seconds = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_proba = predict_positive_probability(fitted, X_valid_fold)
        y_pred = (y_proba >= 0.50).astype(int)
        predict_seconds = time.perf_counter() - pred_start

        row = {
            "model": name,
            "fold": fold,
            "train_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_1000": predict_seconds / len(X_valid_fold) * 1000 * 1000,
        }
        row.update(metric_dict(y_valid_fold, y_pred, y_proba))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_cv_results(cv_results):
    '''Summarize fold-level metrics as mean and standard deviation.'''
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "avg_precision",
        "train_seconds",
        "predict_ms_per_1000",
    ]
    summary = cv_results.groupby("model")[metric_cols].agg(["mean", "std"])
    return summary.sort_values(("recall", "mean"), ascending=False)

# %% [markdown]
# Interpretation:
# 
# The evaluation utility returns fold-level metrics, not just averages. This is important because
# statistical comparison later uses fold-level distributions.

# %% [markdown]
# ## 9. Machine Learning Benchmark
# 
# The primary ML benchmark uses class-weighted or cost-sensitive learning for all four models.
# This is the fairest default for 1M rows because it handles imbalance without changing the natural
# feature distribution. A separate SMOTE sensitivity experiment follows.

# %%
ml_models = build_ml_models(y_model)
ml_cv_results = []

for model_name, estimator in ml_models.items():
    print(f"Evaluating {model_name}...")
    pipeline = make_ml_pipeline(estimator, use_smote=False)
    result = evaluate_pipeline_cv(model_name, pipeline, X_model, y_model)
    ml_cv_results.append(result)

ml_cv_results = pd.concat(ml_cv_results, ignore_index=True)
display(ml_cv_results.round(4))

ml_summary = summarize_cv_results(ml_cv_results)
display(ml_summary.round(4))

# %% [markdown]
# Interpretation:
# 
# Compare recall and F1 first because churn is imbalanced and missed churners are costly. Then check
# ROC-AUC, average precision, and execution time to decide whether a more complex model is worth
# its runtime cost.

# %%
# Visual comparison of predictive performance and runtime.
plot_df = (
    ml_cv_results.groupby("model")
    .agg(
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        roc_auc=("roc_auc", "mean"),
        train_seconds=("train_seconds", "mean"),
        predict_ms_per_1000=("predict_ms_per_1000", "mean"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=plot_df, x="model", y="recall", ax=axes[0])
axes[0].set_title("Mean CV Recall")
axes[0].tick_params(axis="x", rotation=30)

sns.barplot(data=plot_df, x="model", y="f1", ax=axes[1])
axes[1].set_title("Mean CV F1")
axes[1].tick_params(axis="x", rotation=30)

sns.scatterplot(
    data=plot_df,
    x="train_seconds",
    y="recall",
    hue="model",
    s=140,
    ax=axes[2],
)
axes[2].set_title("Recall vs Training Time")
axes[2].set_xlabel("Mean training seconds per fold")
axes[2].set_ylabel("Mean recall")
plt.tight_layout()
plt.show()

# %% [markdown]
# Interpretation:
# 
# The scatter plot directly addresses the thesis trade-off question: a model is attractive only if
# its recall/F1 improvement justifies its additional training and prediction time.

# %% [markdown]
# ## 10. Leakage-Safe SMOTE Sensitivity Analysis
# 
# This section tests whether SMOTE improves churn detection. SMOTE is inside the pipeline, so it is
# fit only on the training part of each CV fold. The validation fold keeps the original imbalanced
# distribution.

# %%
if RUN_SMOTE_SENSITIVITY:
    smote_results = []
    smote_models = build_ml_models(y_smote)

    for model_name, estimator in smote_models.items():
        print(f"Evaluating SMOTE + {model_name}...")
        pipeline = make_ml_pipeline(estimator, use_smote=True)
        result = evaluate_pipeline_cv(f"SMOTE + {model_name}", pipeline, X_smote, y_smote)
        smote_results.append(result)

    smote_cv_results = pd.concat(smote_results, ignore_index=True)
    display(smote_cv_results.round(4))
    display(summarize_cv_results(smote_cv_results).round(4))
else:
    smote_cv_results = pd.DataFrame()
    print("SMOTE sensitivity analysis skipped.")

# %% [markdown]
# Interpretation:
# 
# If SMOTE raises recall but sharply reduces precision or increases runtime, it may be useful only
# when the business cost of missing churners is much higher than the cost of contacting non-churners.

# %% [markdown]
# ## 11. Hyperparameter Optimisation for ML Models
# 
# This section implements three tuning strategies from the proposal: grid search, random search,
# and Optuna. Searches optimise recall because the thesis prioritises detection of true churners.
# The search spaces are intentionally budgeted; expand them for final dissertation experiments.

# %%
def get_param_spaces(y_reference):
    '''Return compact grids and distributions for all ML models.'''
    xgb_weight = calculate_scale_pos_weight(y_reference)
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "model__C": [0.1, 1.0, 5.0],
                "model__solver": ["lbfgs"],
            },
            "random": {
                "model__C": uniform(0.05, 6.0),
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "model__max_depth": [6, 10, 14],
                "model__min_samples_leaf": [50, 100, 250],
            },
            "random": {
                "model__max_depth": randint(4, 18),
                "model__min_samples_leaf": randint(30, 300),
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [10, 16],
                "model__min_samples_leaf": [40, 120],
            },
            "random": {
                "model__n_estimators": randint(100, 260),
                "model__max_depth": randint(8, 20),
                "model__min_samples_leaf": randint(30, 180),
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=xgb_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "grid": {
                "model__n_estimators": [180, 300],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.04, 0.08],
                "model__subsample": [0.85],
                "model__colsample_bytree": [0.85],
            },
            "random": {
                "model__n_estimators": randint(160, 360),
                "model__max_depth": randint(3, 7),
                "model__learning_rate": uniform(0.03, 0.08),
                "model__subsample": uniform(0.75, 0.2),
                "model__colsample_bytree": uniform(0.75, 0.2),
                "model__min_child_weight": randint(2, 10),
            },
        },
    }


def run_grid_and_random_searches(X_data, y_data):
    '''Run budgeted GridSearchCV and RandomizedSearchCV for all ML models.'''
    spaces = get_param_spaces(y_data)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_estimators = {}

    for model_name, spec in spaces.items():
        print(f"Grid search: {model_name}")
        base_pipe = make_ml_pipeline(spec["estimator"], use_smote=False)
        grid = GridSearchCV(
            estimator=base_pipe,
            param_grid=spec["grid"],
            scoring="recall",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        start = time.perf_counter()
        grid.fit(X_data, y_data)
        rows.append(
            {
                "model": model_name,
                "search": "GridSearchCV",
                "best_recall": grid.best_score_,
                "seconds": time.perf_counter() - start,
                "best_params": grid.best_params_,
            }
        )
        best_estimators[(model_name, "GridSearchCV")] = grid.best_estimator_

        print(f"Random search: {model_name}")
        random_search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=spec["random"],
            n_iter=8,
            scoring="recall",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        start = time.perf_counter()
        random_search.fit(X_data, y_data)
        rows.append(
            {
                "model": model_name,
                "search": "RandomizedSearchCV",
                "best_recall": random_search.best_score_,
                "seconds": time.perf_counter() - start,
                "best_params": random_search.best_params_,
            }
        )
        best_estimators[(model_name, "RandomizedSearchCV")] = random_search.best_estimator_

    return pd.DataFrame(rows), best_estimators

# %% [markdown]
# Interpretation:
# 
# Grid search is exhaustive over a small grid. Random search samples a wider range. Both are wrapped
# around the full leakage-safe pipeline, so each parameter trial includes fold-local preprocessing.

# %%
def make_optuna_estimator(model_name, trial, y_reference):
    '''Build an estimator with Optuna-suggested hyperparameters.'''
    xgb_weight = calculate_scale_pos_weight(y_reference)

    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 0.05, 8.0, log=True),
            max_iter=1_000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 4, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 30, 300),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 280),
            max_depth=trial.suggest_int("max_depth", 8, 22),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 30, 180),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 160, 380),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12),
            subsample=trial.suggest_float("subsample", 0.75, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.75, 0.95),
            min_child_weight=trial.suggest_int("min_child_weight", 2, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=xgb_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_optuna_searches(X_data, y_data, n_trials=12):
    '''Run recall-optimised Optuna studies for all ML models.'''
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_estimators = {}

    for model_name in model_names:
        print(f"Optuna search: {model_name}")

        def objective(trial):
            estimator = make_optuna_estimator(model_name, trial, y_data)
            pipeline = make_ml_pipeline(estimator, use_smote=False)
            fold_scores = []

            for train_idx, valid_idx in cv.split(X_data, y_data):
                X_train_fold = X_data.iloc[train_idx]
                X_valid_fold = X_data.iloc[valid_idx]
                y_train_fold = y_data.iloc[train_idx]
                y_valid_fold = y_data.iloc[valid_idx]

                fitted = clone(pipeline)
                fitted.fit(X_train_fold, y_train_fold)
                y_proba = predict_positive_probability(fitted, X_valid_fold)
                y_pred = (y_proba >= 0.50).astype(int)
                fold_scores.append(recall_score(y_valid_fold, y_pred, zero_division=0))

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="maximize")
        start = time.perf_counter()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.perf_counter() - start

        best_estimator = make_optuna_estimator(
            model_name,
            optuna.trial.FixedTrial(study.best_params),
            y_data,
        )
        best_pipeline = make_ml_pipeline(best_estimator, use_smote=False)
        best_pipeline.fit(X_data, y_data)

        rows.append(
            {
                "model": model_name,
                "search": "Optuna",
                "best_recall": study.best_value,
                "seconds": elapsed,
                "best_params": study.best_params,
            }
        )
        best_estimators[(model_name, "Optuna")] = best_pipeline

    return pd.DataFrame(rows), best_estimators

# %% [markdown]
# Interpretation:
# 
# Optuna adaptively explores the search space, so it can often find competitive settings with fewer
# trials than grid search. The objective still refits the entire preprocessing pipeline inside each
# fold, preserving the same leakage controls.

# %%
if RUN_HPO:
    grid_random_results, grid_random_best = run_grid_and_random_searches(X_hpo, y_hpo)
    optuna_results, optuna_best = run_optuna_searches(X_hpo, y_hpo, n_trials=12)
    hpo_results = pd.concat([grid_random_results, optuna_results], ignore_index=True)
    hpo_best_estimators = {**grid_random_best, **optuna_best}

    display(hpo_results.sort_values("best_recall", ascending=False))
else:
    hpo_results = pd.DataFrame()
    hpo_best_estimators = {}
    print("HPO skipped by configuration.")

# %% [markdown]
# Interpretation:
# 
# Use this table to compare not only best recall but also search time. A slower search is justified
# only if it materially improves recall/F1 or produces a model with better deployment trade-offs.

# %% [markdown]
# ## 12. Deep Learning Models: ANN and DNN
# 
# Deep learning models use the same leakage-safe feature engineering and preprocessing. The
# preprocessor is fitted only on the training fold, then transformed arrays are passed to Keras.
# Class imbalance is handled with class weights. ANN is the shallower baseline; DNN adds more
# layers, batch normalisation, and dropout.

# %%
def make_class_weights(y_data):
    '''Compute binary class weights for Keras training.'''
    counts = y_data.value_counts().to_dict()
    total = len(y_data)
    return {
        0: total / (2 * counts.get(0, 1)),
        1: total / (2 * counts.get(1, 1)),
    }


def build_ann(input_dim):
    '''Build a compact ANN baseline.'''
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_dnn(input_dim):
    '''Build a deeper DNN for higher-order feature interactions.'''
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def evaluate_keras_cv(model_name, builder, X_data, y_data, cv_splits=CV_SPLITS):
    '''Evaluate ANN/DNN with fold-local preprocessing and timed Keras training.'''
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_data, y_data), start=1):
        print(f"{model_name} fold {fold}/{cv_splits}")
        X_train_fold = X_data.iloc[train_idx]
        X_valid_fold = X_data.iloc[valid_idx]
        y_train_fold = y_data.iloc[train_idx]
        y_valid_fold = y_data.iloc[valid_idx]

        preprocessor = SkPipeline(make_preprocess_steps())

        prep_start = time.perf_counter()
        X_train_arr = preprocessor.fit_transform(X_train_fold, y_train_fold).astype("float32")
        X_valid_arr = preprocessor.transform(X_valid_fold).astype("float32")
        prep_seconds = time.perf_counter() - prep_start

        model = builder(X_train_arr.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ]

        fit_start = time.perf_counter()
        model.fit(
            X_train_arr,
            y_train_fold.to_numpy(),
            validation_data=(X_valid_arr, y_valid_fold.to_numpy()),
            epochs=DL_EPOCHS,
            batch_size=DL_BATCH_SIZE,
            class_weight=make_class_weights(y_train_fold),
            callbacks=callbacks,
            verbose=0,
        )
        fit_seconds = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_proba = model.predict(X_valid_arr, batch_size=DL_BATCH_SIZE, verbose=0).ravel()
        predict_seconds = time.perf_counter() - pred_start
        y_pred = (y_proba >= 0.50).astype(int)

        row = {
            "model": model_name,
            "fold": fold,
            "preprocess_seconds": prep_seconds,
            "train_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_1000": predict_seconds / len(X_valid_fold) * 1000 * 1000,
        }
        row.update(metric_dict(y_valid_fold, y_pred, y_proba))
        rows.append(row)

        tf.keras.backend.clear_session()

    return pd.DataFrame(rows)

# %% [markdown]
# Interpretation:
# 
# The Keras evaluator mirrors the ML evaluator: fold-local preprocessing, class-weighted training,
# and timed inference. This keeps the ML vs DL comparison fair.

# %%
if RUN_DEEP_LEARNING:
    ann_results = evaluate_keras_cv("ANN", build_ann, X_model, y_model)
    dnn_results = evaluate_keras_cv("DNN", build_dnn, X_model, y_model)
    dl_cv_results = pd.concat([ann_results, dnn_results], ignore_index=True)
    display(dl_cv_results.round(4))
    display(summarize_cv_results(dl_cv_results).round(4))
else:
    dl_cv_results = pd.DataFrame()
    print("Deep learning skipped by configuration.")

# %% [markdown]
# Interpretation:
# 
# Compare ANN and DNN against the ML models on recall, F1, AUC, and runtime. A DNN is only
# preferable if it improves detection enough to justify slower training and reduced interpretability.

# %% [markdown]
# ## 13. Combined CV Comparison and Statistical Testing
# 
# This section combines fold-level ML, SMOTE sensitivity, and DL results. Paired tests are used only
# where models share the same folds; otherwise, the table is treated as descriptive evidence.

# %%
all_cv_parts = [ml_cv_results]
if "smote_cv_results" in globals() and not smote_cv_results.empty:
    all_cv_parts.append(smote_cv_results)
if "dl_cv_results" in globals() and not dl_cv_results.empty:
    all_cv_parts.append(dl_cv_results)

all_cv_results = pd.concat(all_cv_parts, ignore_index=True)
display(summarize_cv_results(all_cv_results).round(4))

tradeoff_df = (
    all_cv_results.groupby("model")
    .agg(
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        roc_auc=("roc_auc", "mean"),
        avg_precision=("avg_precision", "mean"),
        train_seconds=("train_seconds", "mean"),
        predict_ms_per_1000=("predict_ms_per_1000", "mean"),
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=tradeoff_df,
    x="predict_ms_per_1000",
    y="recall",
    size="train_seconds",
    hue="model",
    sizes=(80, 500),
    ax=ax,
)
ax.set_title("Recall vs Prediction Latency")
ax.set_xlabel("Prediction milliseconds per 1,000 customers")
ax.set_ylabel("Mean CV recall")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# Interpretation:
# 
# The best model for a bank is not automatically the highest-recall model. A useful model should
# also score customers quickly enough for the retention workflow and remain interpretable enough
# for business use.

# %%
def pairwise_wilcoxon(results_df, metric="recall"):
    '''Run pairwise Wilcoxon tests for models with the same fold count.'''
    rows = []
    pivot = results_df.pivot_table(index="fold", columns="model", values=metric)
    models = list(pivot.columns)

    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            pair = pivot[[model_a, model_b]].dropna()
            if len(pair) < 2:
                continue
            try:
                stat, p_value = stats.wilcoxon(pair[model_a], pair[model_b])
            except ValueError:
                stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_a": pair[model_a].mean(),
                    "mean_b": pair[model_b].mean(),
                    "mean_diff_a_minus_b": pair[model_a].mean() - pair[model_b].mean(),
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


recall_tests = pairwise_wilcoxon(all_cv_results, metric="recall")
f1_tests = pairwise_wilcoxon(all_cv_results, metric="f1")

print("Pairwise Wilcoxon tests for recall")
display(recall_tests.round(5))

print("Pairwise Wilcoxon tests for F1")
display(f1_tests.round(5))

# %% [markdown]
# Interpretation:
# 
# With only a few CV folds, p-values are supportive rather than definitive. Use them as evidence of
# consistency across folds, not as the only model-selection criterion.

# %% [markdown]
# ## 14. Final Holdout Evaluation
# 
# This section fits selected final models on the full training partition and evaluates once on the
# untouched holdout test set. If HPO was run, the best HPO model by recall is included.

# %%
final_candidates = {}

# In smoke-test mode, fit final models on the modelling sample to validate code quickly.
X_final_train = X_model if SMOKE_TEST else X_train_full
y_final_train = y_model if SMOKE_TEST else y_train_full

# Add the best base ML model by mean CV recall.
base_best_name = (
    ml_cv_results.groupby("model")["recall"].mean().sort_values(ascending=False).index[0]
)
final_candidates[f"Base {base_best_name}"] = make_ml_pipeline(
    build_ml_models(y_final_train)[base_best_name],
    use_smote=False,
)

# Add the best HPO model if available.
if "hpo_results" in globals() and not hpo_results.empty:
    best_hpo_row = hpo_results.sort_values("best_recall", ascending=False).iloc[0]
    best_key = (best_hpo_row["model"], best_hpo_row["search"])
    final_candidates[f"HPO {best_hpo_row['search']} {best_hpo_row['model']}"] = hpo_best_estimators[
        best_key
    ]

holdout_rows = []
fitted_final_models = {}

for model_name, pipeline in final_candidates.items():
    print(f"Fitting final model: {model_name}")
    fitted = clone(pipeline)

    fit_start = time.perf_counter()
    fitted.fit(X_final_train, y_final_train)
    fit_seconds = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    y_proba = predict_positive_probability(fitted, X_test)
    y_pred = (y_proba >= 0.50).astype(int)
    pred_seconds = time.perf_counter() - pred_start

    row = {
        "model": model_name,
        "train_seconds": fit_seconds,
        "predict_seconds": pred_seconds,
        "predict_ms_per_1000": pred_seconds / len(X_test) * 1000 * 1000,
    }
    row.update(metric_dict(y_test, y_pred, y_proba))
    holdout_rows.append(row)
    fitted_final_models[model_name] = fitted

holdout_results = pd.DataFrame(holdout_rows).sort_values("recall", ascending=False)
display(holdout_results.round(4))

# %% [markdown]
# Interpretation:
# 
# This is the most important performance table because the test set has never influenced model
# training or selection. Use it to support the final thesis conclusion about performance versus
# execution-time trade-off.

# %%
# Confusion matrix and classification report for the best holdout model.
best_holdout_name = holdout_results.iloc[0]["model"]
best_holdout_model = fitted_final_models[best_holdout_name]
best_proba = predict_positive_probability(best_holdout_model, X_test)
best_pred = (best_proba >= 0.50).astype(int)

cm = confusion_matrix(y_test, best_pred)
cm_df = pd.DataFrame(cm, index=["Actual non-churn", "Actual churn"], columns=["Pred non-churn", "Pred churn"])
display(cm_df)

print(classification_report(y_test, best_pred, digits=4, zero_division=0))

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt=",d", cmap="Blues", ax=ax)
ax.set_title(f"Holdout Confusion Matrix: {best_holdout_name}")
plt.show()

# %% [markdown]
# Interpretation:
# 
# False negatives are actual churners the model missed. In banking retention, this count is often
# more costly than false positives, so the confusion matrix should be interpreted alongside recall
# and business intervention cost.

# %% [markdown]
# ## 15. Threshold Tuning for Business Trade-Offs
# 
# The default threshold of 0.50 may not be optimal for churn intervention. This section shows how
# precision and recall change across thresholds, allowing the bank to choose a threshold based on
# campaign budget and missed-churn cost.

# %%
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, best_proba)
threshold_df = pd.DataFrame(
    {
        "threshold": np.r_[thresholds, 1.0],
        "precision": precision_vals,
        "recall": recall_vals,
    }
)
threshold_df["f1"] = (
    2
    * threshold_df["precision"]
    * threshold_df["recall"]
    / (threshold_df["precision"] + threshold_df["recall"]).replace(0, np.nan)
)

display(threshold_df.iloc[:: max(len(threshold_df) // 20, 1)].round(4))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
ax.set_title(f"Threshold Trade-Off: {best_holdout_name}")
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Score")
ax.legend()
plt.show()

# %% [markdown]
# Interpretation:
# 
# Lower thresholds usually increase recall but reduce precision. If the bank can afford more
# retention offers, a lower threshold may be justified; if intervention budget is tight, a higher
# threshold may be preferable.

# %% [markdown]
# ## 16. Interpretability with SHAP
# 
# SHAP is applied to the best tree-based final model when available. The explanation uses transformed
# features from the fitted pipeline, preserving the same preprocessing used during training.

# %%
def get_transformed_matrix_and_names(fitted_pipeline, X_data):
    '''Transform X through feature engineering and preprocessing, returning matrix and names.'''
    engineered = fitted_pipeline.named_steps["feature_engineering"].transform(X_data)
    matrix = fitted_pipeline.named_steps["preprocess"].transform(engineered)
    try:
        names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(matrix.shape[1])])
    return matrix, names


if RUN_SHAP:
    tree_model_name = None
    for candidate_name in fitted_final_models:
        estimator = fitted_final_models[candidate_name].named_steps["model"]
        if isinstance(estimator, (RandomForestClassifier, XGBClassifier, DecisionTreeClassifier)):
            tree_model_name = candidate_name
            break

    if tree_model_name is None:
        print("No tree-based final model found for SHAP. Consider permutation importance instead.")
    else:
        print(f"Explaining: {tree_model_name}")
        shap_model = fitted_final_models[tree_model_name]
        estimator = shap_model.named_steps["model"]

        X_explain = X_test.sample(n=min(1_000, len(X_test)), random_state=RANDOM_STATE)
        X_explain_matrix, feature_names = get_transformed_matrix_and_names(shap_model, X_explain)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_explain_matrix)

        # Some estimators return a list for binary classification; use the positive class.
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        shap.summary_plot(
            shap_values_to_plot,
            X_explain_matrix,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Summary: {tree_model_name}")
        plt.tight_layout()
        plt.show()
else:
    print("SHAP skipped by configuration.")

# %% [markdown]
# Interpretation:
# 
# The SHAP summary identifies the strongest global drivers of churn. Use it to check whether the
# model's logic aligns with EDA and banking intuition, especially satisfaction, complaints, contract
# type, payment behaviour, and support/service features.

# %% [markdown]
# ## 17. Save Outputs
# 
# This section saves fold-level results and final holdout results so they can be used in the thesis
# write-up.

# %%
OUTPUT_DIR = Path("thesis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

all_cv_results.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
tradeoff_df.to_csv(OUTPUT_DIR / "runtime_tradeoff_summary.csv", index=False)
holdout_results.to_csv(OUTPUT_DIR / "holdout_results.csv", index=False)

if "hpo_results" in globals() and not hpo_results.empty:
    hpo_results.to_csv(OUTPUT_DIR / "hpo_results.csv", index=False)

print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")

# %% [markdown]
# Interpretation:
# 
# The saved CSV files preserve the numerical evidence for the thesis tables: cross-validation
# metrics, runtime trade-offs, HPO results, and final holdout performance.

# %% [markdown]
# ## 18. Methodology References
# 
# - Proposal: `01_Duc_Le_Research_proposal.pdf`.
# - scikit-learn TargetEncoder documentation:
#   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
#   and internal cross-fitting example:
#   https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
# - imbalanced-learn common pitfalls:
#   https://imbalanced-learn.org/dev/common_pitfalls.html
#   and pipeline documentation:
#   https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html
# - Chawla et al. (2002): original SMOTE method for synthetic minority oversampling.
# - Molnar (2020): interpretable machine learning and SHAP-style model interpretation.
# 
# These references support the notebook's leakage-safe target encoding, fold-local resampling,
# class-imbalance treatment, and interpretability workflow.

