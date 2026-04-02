<div align="center">

# 🏥 Claim Denial Prediction System

### An End-to-End Machine Learning Pipeline for Healthcare Claims Analytics

> **Predict. Classify. Explain.** — A three-stage ML pipeline that flags denied claims before they happen, categorizes denial type, and pinpoints the reasons — reducing revenue leakage and operational overhead.

</div>


## 🚀 Overview

Healthcare claim denials cost providers **billions of dollars annually** in rework, appeals, and lost revenue. This project builds an intelligent prediction system that operates in three stages:

| Stage | Task | Type | Output |
|-------|------|------|--------|
| 🔴 Stage 1 | Will this claim be denied? | Binary Classification | `DenialFlag` (0 or 1) |
| 🟡 Stage 2 | What *type* of denial is it? | Multi-Class Classification | `N / P / Z / F` |
| 🟢 Stage 3 | *Why* will it be denied? | Multi-Label Classification | Up to 26 denial reason codes |

This cascade approach mirrors real-world claim adjudication workflows, making predictions actionable and interpretable.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Raw Records | **1,000,000** rows |
| Features (raw) | **39** columns |
| Records after cleaning | **926,015** rows |
| Final feature count | **23** numeric features |

### Data Domains

```
📁 Claim Data
├── 🏷️  Claim-level details     (Clinic, Service, CPTCode, AmountCharged)
├── 💰  Financial attributes    (CoPay, Deduc, CoIns, TotalPaid)
├── 🏢  Provider & Payer info   (Provider, Payer, BillingProviderNPI)
├── 🩺  Diagnosis codes         (f21diag1)
└── 📅  Service & billing dates (ServiceDt, ClaimBillDate, f11insdob)
```

---

## 🏗️ Pipeline Architecture

```
Raw CSV (1M rows)
      │
      ▼
┌─────────────────────────┐
│   Data Preprocessing     │  Remove leakage, identifiers, parse dates
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Feature Engineering    │  Frequency encoding + date features → 23 features
└───────────┬─────────────┘
            │
      ┌─────┴──────┐
      │            │
      ▼            ▼
 Binary (X)   Multi-class (X)   ← same features, different targets
      │            │
      ▼            ▼
 Random        Random Forest /
 Forest         XGBoost / MLP
      │            │
      └─────┬──────┘
            │
            ▼
    Multi-label (X, y: 26 labels)
            │
            ▼
    Classifier Chains (RF)
            │
            ▼
     Structured Predictions
```

---

## 🧹 Data Preprocessing

### 🔒 Removing Data Leakage
Columns that directly reveal claim outcomes were dropped to prevent target leakage:

```python
leakage_cols = ["TotalPaid", "TotalAdj", "TotalVoid", "Balance",
                "CltResp", "cliANSI1", "cliANSI2", "lastActDt"]
```

### 🪪 Removing Identifiers
```python
id_cols = ["TPCLIID", "LIATPCLIid", "ClaimID", "ClientID"]
```

### 🎯 Target Engineering

**Binary Target** (`DenialFlag`): 73,985 null records dropped before binarizing.

**Multi-Class Target** (`MultiFlag`):

| Label | Meaning | Records | Proportion |
|-------|---------|---------|------------|
| `N` | No Denial | 764,565 | 82.57% |
| `P` | Partial Denial | 97,757 | 10.56% |
| `Z` | Zero Payment | 48,946 | 5.29% |
| `F` | Full Denial | 14,747 | 1.59% |

**Multi-Label Target**: Combined `target1–target4`, encoded with `MultiLabelBinarizer` → **926,015 × 26** binary matrix.

---

## ⚙️ Feature Engineering

### 🔢 Frequency Encoding (over One-Hot Encoding)

Applied to 10 high-cardinality categorical columns:

```
Clinic · Service · CPTCode · Payer · Provider
BillingProviderNPI · ClaimFacilityNPI · eligStatus
tpcliStrModifier · f21diag1
```

| Why NOT One-Hot? | Why Frequency Encoding? |
|-----------------|------------------------|
| Explodes dimensionality | Keeps features compact |
| Creates sparse matrices | Preserves category distribution |
| Inefficient for tree models | Works natively with RF/XGBoost |

### 📅 Date-Derived Features

```python
service_year, service_month      # Seasonal billing patterns
billing_year, billing_month      # Submission timing
patient_age                      # Risk stratification
```

---

## 🧠 Models & Results

### 🔵 Model 1 — Binary Classification (Denial Flag)

**Algorithm:** Random Forest Classifier (`n_estimators=100`)

**Strategy:** Threshold tuning to maximize recall — catching more denied claims is more critical than avoiding false alarms.

| Metric | Value |
|--------|-------|
| Accuracy | **0.97** |
| Precision | 0.88 |
| Recall | **0.92** |
| F1 Score | 0.90 |
| ROC-AUC | **0.986** |

#### Threshold Analysis

| Threshold | Recall | Precision | Trade-off |
|-----------|--------|-----------|-----------|
| 0.3 | **0.92** | 0.88 | ✅ Selected — maximizes recall |
| 0.4 | 0.91 | 0.91 | Balanced |
| 0.5 | 0.89 | 0.94 | Default |
| 0.6 | 0.88 | 0.95 | High precision |

> **Decision:** Threshold **0.3** selected. In claims processing, missing a real denial (false negative) is more costly than a false alarm — so recall is prioritized.

## 📈 Feature Importance — Binary Model
Derived from the Random Forest binary classification model:

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `Payer` | 0.250 | Strongest predictor — payer policies dominate |
| 2 | `AmountCharged` | 0.149 | Higher charges correlate with denial risk |
| 3 | `service_month` | 0.077 | Seasonal patterns in claim outcomes |
| 4 | `Provider` | 0.077 | Provider-level denial history matters |
| 5 | `Service` | 0.066 | Service type eligibility rules |
| 6 | `Clinic` | 0.052 | Clinic-level billing behavior |
| 7 | `DaysBetServiceToBilling` | 0.051 | Delayed billing increases denial risk |
| 8 | `tpcliStrPOS` | 0.046 | Place of service affects coverage |
| 9 | `CPTCode` | 0.045 | Procedure code determines eligibility |
| 10 | `ClaimFacilityNPI` | 0.040 | Facility-level risk signal |

---

### 🟡 Model 2 — Multi-Class Classification (Denial Type)

**Target Classes:** `F` (Full) · `N` (None) · `P` (Partial) · `Z` (Zero Payment)

#### Model Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|-------|----------|-------------------|----------------|------------|
| ✅ **Random Forest** | **0.96** | **0.88** | **0.81** | **0.84** |
| XGBoost | 0.95 | 0.82 | 0.69 | 0.75 |
| MLP (Neural Net) | 0.93 | 0.80 | 0.67 | 0.73 |

#### Per-Class Performance (Random Forest)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| F (Full Denial) | 0.84 | 0.66 | 0.74 | 2,949 |
| N (No Denial) | 0.98 | 0.99 | 0.98 | 152,913 |
| P (Partial) | 0.93 | 0.94 | 0.93 | 19,552 |
| Z (Zero Pay) | 0.78 | 0.65 | 0.71 | 9,789 |

#### Why Random Forest Won

```
✅ Highest macro F1 — critical for imbalanced multi-class problems
✅ Better recall on minority classes (F, Z)
✅ More stable and interpretable than boosting/neural net
✅ Handles imbalance natively via class_weight="balanced"
```
## 📈 Feature Importance

Derived from the Random Forest multi-class model:

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `Payer` | 0.203 | Payer rules drive denials most |
| 2 | `AmountCharged` | 0.173 | Financial thresholds matter |
| 3 | `service_month` | 0.093 | Seasonal claim patterns |
| 4 | `Service` | 0.065 | Service type eligibility |
| 5 | `DaysBetServiceToBilling` | 0.065 | Late billing → higher denial |
| 6 | `Provider` | 0.063 | Provider-level risk |
| 7 | `billing_month` | 0.049 | Submission timing effects |

> ⚠️ Features `f21diag1`, `AuthStatus`, and `patient_age` showed near-zero importance — candidates for removal in future iterations.

---

### 🟢 Model 3 — Multi-Label Classification (Denial Reasons)

**Setup:** 26 possible denial reason labels, any combination can apply to a single claim.

#### Model Comparison

| Model | Micro F1 | Macro F1 | Recall (Micro) | Precision (Micro) |
|-------|----------|----------|----------------|-------------------|
| Binary Relevance | 0.87 | 0.65 | 0.83 | 0.90 |
| ✅ **Classifier Chains** | **0.87** | **0.67** | **0.84** | 0.90 |

#### Why Classifier Chains?

```
✅ Captures label dependencies (e.g., label 23 often co-occurs with label 31)
✅ Better Macro F1 → improved performance on rare denial codes
✅ Higher recall → fewer missed denial reasons
✅ More realistic — denial reasons in practice are often correlated
```
## 📈 Feature Importance — Multi-Label Model
Derived from the Classifier Chains (Random Forest base) multi-label model:

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `AmountCharged` | 0.190 | Billing amount is the top driver of denial reasons |
| 2 | `Payer` | 0.150 | Payer-specific rules determine denial codes |
| 3 | `service_year` | 0.087 | Year-level policy changes affect reason codes |
| 4 | `SameDayCli` | 0.076 | Same-day claims carry distinct denial patterns |
| 5 | `billing_year` | 0.065 | Billing year reflects regulatory environment |
| 6 | `Provider` | 0.046 | Provider behavior influences reason type |
| 7 | `Service` | 0.045 | Service category tied to specific denial codes |
| 8 | `CPTCode` | 0.032 | Procedure-level eligibility mismatches |
| 9 | `tpcliStrModifier` | 0.029 | Billing modifiers affect reason assignment |
| 10 | `Clinic` | 0.026 | Clinic-level patterns in denial reasons |

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

### 2. Place Dataset

```
project/
└── ClaimDenialInputMultiLabel.csv   ← your raw input file
```

### 3. Run the Notebook

```bash
jupyter notebook checkpoint4.ipynb
```

### 4. Saved Artifacts

After training, the following files are saved:

| File | Contents |
|------|----------|
| `binary_model.pkl` | Random Forest — Denial Flag |
| `multiclass_model.pkl` | Random Forest — Denial Type |
| `multilabel_model.pkl` | Classifier Chains — Denial Reasons |
| `label_encoder.pkl` | MultiFlag label encoder |
| `mlb.pkl` | MultiLabelBinarizer (26 labels) |
| `imputer_ml.pkl` | Mean imputer for missing values |
| `freq_maps.pkl` | Frequency encoding maps |

---

## 🗂️ Project Structure

```
claim-denial-prediction/
│
├── 📓 checkpoint4.ipynb     # Full pipeline notebook
├── 📄 README.md                         # This file
│
├── 🗃️ Data
│   └── ClaimDenialInputMultiLabel.csv   # Raw input (1M rows)
│
└── 🤖 Saved Models
    ├── binary_model.pkl
    ├── multiclass_model.pkl
    ├── multilabel_model.pkl
    ├── label_encoder.pkl
    ├── mlb.pkl
    ├── imputer_ml.pkl
    └── freq_maps.pkl
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frequency encoding over one-hot | Avoids dimensionality explosion with 10+ high-cardinality columns |
| Stratified train-test split | Preserves class ratios for imbalanced targets |
| Threshold 0.3 for binary model | Prioritizes recall — missing a denied claim is costlier than a false alarm |
| `class_weight="balanced"` | Compensates for severe class imbalance (F: 1.6% of data) |
| Classifier Chains over Binary Relevance | Captures real-world correlations between denial reason codes |

---
## 🌐 REST API — FastAPI Inference Server

The trained models are served via a FastAPI application (`app.py`), enabling real-time claim denial predictions through a simple HTTP interface.

### Running the Server
```bash
uvicorn app:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — confirms API is running |
| `POST` | `/predict` | Submit a claim and get full predictions |

### `POST /predict` — Request Body

Send a raw claim record as JSON. Dates are parsed and all preprocessing (frequency encoding, feature alignment) happens server-side.
```json
{
  "Clinic": "CLN_13838227",
  "Service": "Methadone Maintenance Week",
  "AmountCharged": 297.61,
  "CPTCode": "H0020",
  "Payer": "PayerName",
  "Provider": "ProviderName",
  "BillingProviderNPI": "1234567890",
  "ClaimFacilityNPI": "0987654321",
  "AuthStatus": 1,
  "eligStatus": "Active",
  "CoPay": 0.0,
  "Deduc": 0.0,
  "CoIns": 0.0,
  "SameDayCli": 0,
  "DaysBetServiceToBilling": 1,
  "tpcliStrModifier": null,
  "tpcliStrPOS": 11,
  "f21diag1": "F1120",
  "ServiceDt": "2025-06-22",
  "ClaimBillDate": "2025-04-06",
  "f11insdob": "1985-03-15"
}
```

### `POST /predict` — Response

All three model predictions are returned in a single response:
```json
{
  "denial_flag": 1,
  "denial_type": "P",
  "denial_reasons": ["13", "23", "31"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `denial_flag` | `int` | `1` = claim likely denied, `0` = likely paid |
| `denial_type` | `str` | `N` / `P` / `Z` / `F` — denial category |
| `denial_reasons` | `list[str]` | One or more denial reason codes (up to 26 possible) |

### Preprocessing Pipeline (Server-Side)

The API automatically handles all transformations before inference:
```
Raw JSON Input
      │
      ├── Date parsing → service_year, service_month, billing_year, billing_month, patient_age
      ├── Frequency encoding → maps categorical values using training distributions
      ├── Column alignment → fills missing fields with 0
      │
      ├──▶ Binary Model   → denial_flag
      ├──▶ Multi-class    → denial_type
      └──▶ Classifier Chains → denial_reasons
```
---

## ⚠️ Limitations & Future Work

- **Low Samples F1 (0.20)** on multi-label model — many claims have no denial reasons labeled, inflating denominator
- **Minority class recall** for `F` and `Z` can be improved with SMOTE or cost-sensitive learning
- **Near-zero importance features** (`f21diag1`, `AuthStatus`, `patient_age`) — worth removing in

---

