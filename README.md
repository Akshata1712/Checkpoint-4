
#  Claim Denial Prediction System

## 🚀 Overview

This project builds an end-to-end machine learning pipeline to predict:

- **Denial Probability** (Binary Classification)
- **Denial Type** (Multi-Class Classification)
- **Denial Reasons** (Multi-Label Classification)

The system processes raw claim data, performs feature engineering, and uses multiple models to generate structured predictions.


## 📊 Dataset

- Initial dataset: **1,000,000 rows × 39 columns**
- Final dataset after cleaning: **926,015 rows**

### Data Includes:
- Claim-level details
- Financial attributes
- Provider & payer information
- Diagnosis codes
- Service & billing dates



##  Data Preprocessing

###  Removing Data Leakage
Removed columns that directly reveal outcomes:
TotalPaid, TotalAdj, Balance, CltResp, cliANSI1, cliANSI2, lastActDt



###  Removing Identifiers
TPCLIID, LIATPCLIid, ClaimID, ClientID


###  Date Processing
Converted:ServiceDt, ClaimBillDate, f11insdob



###  Numeric Conversion
Handled monetary and numeric fields using coercion.




## 🎯 Target Engineering

### Binary Target (DenialFlag)
- Cleaned null values
- Converted to integer (0/1)



### Multi-Class Target (MultiFlag)

| Class | Meaning | Proportion |
|------|--------|-----------|
| N | No denial | 82.56% |
| P | Partial | 10.56% |
| Z | Zero payment | 5.28% |
| F | Full denial | 1.59% |



### Multi-Label Targets
- Combined `target1–target4`
- Applied **MultiLabelBinarizer**
- Total labels: **26**



## ⚙️ Feature Engineering

### 🔹 Frequency Encoding
Applied instead of one-hot encoding:

**Columns:**
- Clinic, Service, CPTCode, Payer, Provider,
BillingProviderNPI, ClaimFacilityNPI, eligStatus,
tpcliStrModifier, f21diag1



### 🔹 Date Features Created
- service_year
- service_month
- billing_year
- billing_month
- patient_age



### 🔹 Final Feature Set
- Total features: **23**
- Fully numeric



## 🔀 Train-Test Split

Used stratified splitting:
```python
train_test_split(..., stratify=y)
```

Why?

- Maintains class distribution
- Prevents bias in imbalanced datasets

## 🧠 MODEL 1: Binary Classification
Model: Random Forest

### Performance
| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.97  |
| Precision | 0.88  |
| Recall    | 0.92  |
| F1 Score  | 0.90  |
| ROC-AUC   | 0.986 |

### Threshold Tuning
| Threshold | Recall   | Precision |
| --------- | -------- | --------- |
| 0.3       | **0.92** | 0.88      |
| 0.4       | 0.91     | 0.91      |
| 0.5       | 0.89     | 0.94      |
| 0.6       | 0.88     | 0.95      |

Final choice: 0.3 (better recall)

## 🧠 MODEL 2: Multi-Class Classification

### Model comparision

| Model         | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| ------------- | -------- | ----------------- | -------------- | ---------- |
| Random Forest | **0.96** | 0.88              | **0.81**       | **0.84**   |
| XGBoost       | 0.95     | 0.82              | 0.69           | 0.75       |
| MLP           | 0.93     | 0.80              | 0.67           | 0.73       |

### 🔍 Interpretation
#### Random Forest
- Best balance across all metrics
- Handles imbalance better
#### XGBoost
- Good accuracy but poor minority recall
#### MLP
- Less stable, weaker generalization

### Final Model: Random Forest

- Highest macro F1 (important for imbalance)
- Better recall for minority classes (F, Z)
- More stable and interpretable

## 🧠 MODEL 3: Multi-Label Classification

### Model Comparision
| Model             | Micro F1 | Macro F1 | Samples F1 | Precision (Micro) | Recall (Micro) |
| ----------------- | -------- | -------- | ---------- | ----------------- | -------------- |
| Binary Relevance  | 0.87     | 0.65     | 0.20       | 0.90              | 0.83           |
| Classifier Chains | **0.87** | **0.67** | 0.20       | 0.90              | **0.84**       |

### 🔍 Interpretation
- Micro F1 same → overall performance similar
- Macro F1 higher in CC → better for rare labels
- Recall higher in CC → better detection

### ✅ Final Model: Classifier Chains

Why selected:

- Captures label dependencies
- Better macro performance
- More realistic modeling of denial reasons

## Feature Engineering

### Date Features
- service_month, billing_month → capture seasonal trends  
- patient_age → important risk factor  
- billing delay → operational inefficiency indicator

### Frequency Encoding
Why NOT one-hot?
- Too many categories → huge dimensionality
- Sparse matrix → inefficient

Why frequency encoding?
- Preserves distribution
- Efficient for large datasets
- Works well with tree models

## Confusion Matrix Insights

- Strong bias toward majority class (N)
- Minority classes (F, Z) often misclassified
Indicates:
- Class imbalance challenge
- Need for better minority handling

## Feature Importance Insights
- Payer
- AmountCharged
- service_month
- DaysBetServiceToBilling
- Provider
  
### Interpretation
- Financial + payer features dominate decisions
- Temporal patterns influence claim outcomes
- Some features had near-zero importance → possible feature reduction
