# 🧠 Sportsense — Athlete Injury Prediction ML

> **Machine Learning pipeline for athlete injury-risk prediction, injury onset estimation, recovery-duration prediction, and explainable predictions using SHAP.**

Sportsense ML analyzes historical athlete **activity, sleep, training, and profile data** to estimate the likelihood of injury within a future risk window.

The system consists of three machine-learning models:

* **Injury Risk Classification**
* **Injury Onset Prediction**
* **Recovery Duration Prediction**

The predictions are supported by **SHAP-based explainability**, allowing the system to identify which features contributed most to an individual prediction.

---

## 🎯 Objectives

The ML system aims to answer three questions:

### 1. Is the athlete at risk?

A classification model predicts the probability of injury within the defined risk window.

### 2. When could the injury occur?

A regression model estimates the number of days until predicted injury onset.

### 3. How long could recovery take?

A regression model estimates the expected recovery duration.

---

# 🔬 ML Pipeline

```text
┌──────────────────────────┐
│       RAW DATASETS       │
├──────────────────────────┤
│ Athlete Metadata         │
│ Daily Activity           │
│ Sleep Data               │
│ Training Sessions        │
│ Injury Labels            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   DATA PREPROCESSING     │
├──────────────────────────┤
│ Data Cleaning            │
│ Missing Values           │
│ Type Conversion           │
│ Duplicate Handling       │
│ Date Alignment           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ TEMPORAL ALIGNMENT       │
├──────────────────────────┤
│ Athlete × Calendar Day   │
│ Historical Data Only     │
│ Leakage-Safe Cutoff      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   FEATURE ENGINEERING     │
├──────────────────────────┤
│ Activity Load             │
│ Training Load             │
│ Rolling Windows           │
│ Sleep Statistics          │
│ Acute/Chronic Ratio       │
│ Athlete Profile Features  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      67 FEATURES         │
└────────────┬─────────────┘
             │
             ▼
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌────────────┐  ┌────────────────────┐
│ Train/Test │  │ Preprocessing      │
│ Split      │  │ Pipeline           │
└─────┬──────┘  └─────────┬──────────┘
      │                    │
      └──────────┬─────────┘
                 ▼
       ┌─────────────────────┐
       │    MODEL TRAINING   │
       └──────────┬──────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
┌────────────┐ ┌──────────┐ ┌─────────────┐
│ XGBoost    │ │ XGBoost  │ │ XGBoost     │
│ Classifier │ │ Regressor│ │ Regressor   │
├────────────┤ ├──────────┤ ├─────────────┤
│ Injury     │ │ Onset    │ │ Recovery    │
│ Risk       │ │ Days     │ │ Duration    │
└─────┬──────┘ └────┬─────┘ └──────┬──────┘
      │             │              │
      └─────────────┼──────────────┘
                    ▼
       ┌────────────────────────┐
       │     MODEL EVALUATION   │
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │    SHAP EXPLAINABILITY │
       ├────────────────────────┤
       │ Global Importance      │
       │ Athlete-Level Factors  │
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │    MODEL BUNDLE        │
       ├────────────────────────┤
       │ Models                 │
       │ Preprocessors          │
       │ Threshold              │
       │ Feature Metadata       │
       │ Model Version          │
       └────────────────────────┘
```

---

# 📂 Dataset

The training pipeline uses five primary datasets.

## Athlete Metadata

Contains athlete-level information:

```text
athlete_id
sport
age
gender
height_cm
weight_kg_baseline
dominant_side
years_playing
position
team_id
prior_season_injury_count
```

## Daily Activity

Contains daily physical activity information including:

```text
steps
distance
calories
active minutes
sedentary minutes
activity intensity
```

## Sleep Data

Contains sleep and recovery information used to derive sleep-related features.

## Training Sessions

Contains athlete training-session information used to calculate workload and training-load features.

## Injury Labels

Contains the supervised learning targets:

```text
injured_in_risk_window
onset_day_offset
recovery_duration
```

---

# 🧹 Data Preprocessing

The raw datasets are cleaned and transformed before model training.

Main preprocessing operations include:

* Missing-value handling
* Data type conversion
* Duplicate removal
* Date normalization
* Athlete-level alignment
* Calendar-day alignment
* Invalid-value handling
* Numerical feature imputation
* Categorical feature encoding

The preprocessing pipelines are saved alongside the trained models so that **training and inference use the same transformations**.

---

# 🔐 Leakage Prevention

Preventing data leakage was a major part of the ML pipeline.

The final feature dataset uses a **temporal cutoff** so that information occurring after the prediction point is not used to generate features.

```text
Historical Athlete Data
        │
        │
        ▼
┌─────────────────┐
│ Prediction      │
│ Cutoff          │
└────────┬────────┘
         │
         ├──────────────► Features
         │               ONLY FROM
         │               PAST DATA
         │
         ▼
   Future Risk Window
         │
         ▼
       Labels
```

This prevents future activity/training information from leaking into the prediction features.

> **Important:** The current cutoff/risk-window definition is an experimental modeling setup because the original labels do not contain explicit real-world risk-window dates. Production training should use explicit temporal labels when available.

---

# ⚙️ Feature Engineering

The final model uses **67 features**.

Major feature groups include:

### Athlete Features

```text
Age
Gender
Height
Weight
Sport
Position
Dominant Side
Years Playing
Previous Injury Count
```

### Activity Features

```text
Activity Load
3-Day Activity Load
7-Day Activity Load
14-Day Activity Load
28-Day Activity Load
Active Minutes
14-Day Active Minutes
Steps
Distance
Calories
```

### Training Features

```text
Training Load
3-Day Training Load
7-Day Training Load
14-Day Training Load
28-Day Training Load
Acute-Chronic Workload Ratio
```

### Sleep Features

```text
Sleep Duration
Sleep Mean
Sleep Standard Deviation
Sleep Consistency
Bed Minutes
```

Rolling features are calculated using **calendar-day aligned athlete data**, rather than simply rolling over available rows.

This prevents incorrect workload calculations when days are missing.

---

# 🤖 Machine Learning Models

Sportsense uses **three independent XGBoost models**.

## 1. Injury Risk Classifier

```text
XGBClassifier
```

### Target

```text
injured_in_risk_window
```

The model produces an injury-risk probability.

The probability is converted into:

```text
LOW
MEDIUM
HIGH
```

using an optimized classification threshold.

---

## 2. Injury Onset Model

```text
XGBRegressor
```

### Target

```text
onset_day_offset
```

Predicts the estimated number of days until injury onset.

Prediction range:

```text
1–30 days
```

The onset model is trained on athletes who are labeled as injured.

---

## 3. Recovery Duration Model

```text
XGBRegressor
```

### Target

```text
recovery_duration
```

Predicts estimated recovery duration.

Prediction range:

```text
5–20 days
```

The recovery model is trained on injured athletes.

---

# 📊 Model Performance

## Injury Risk Classification

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **84.33%** |
| Precision | **92.65%** |
| Recall    | **60.00%** |
| F1 Score  | **72.83%** |
| ROC-AUC   | **78.61%** |

---

## Injury Onset Prediction

| Metric |        Result |
| ------ | ------------: |
| MAE    | **2.57 days** |
| RMSE   | **4.31 days** |
| R²     |      **0.76** |

---

## Recovery Prediction

| Metric |        Result |
| ------ | ------------: |
| MAE    | **2.88 days** |
| RMSE   | **3.44 days** |
| R²     |      **0.24** |

> Recovery prediction currently has limited predictive strength and should be treated as an approximate estimate.

---

# 🔍 Explainable AI — SHAP

Sportsense uses **SHAP (SHapley Additive exPlanations)** to interpret model predictions.

SHAP provides:

### Global Explainability

Identifies which features are generally important across the dataset.

### Athlete-Level Explainability

Identifies which features contributed most to a specific athlete's prediction.

Example:

```text
Risk Increasing Factors
────────────────────────
Height
Bed Minutes
14-Day Active Minutes
Training Load Ratio
Sleep Consistency
```

```text
Risk Reducing Factors
──────────────────────
28-Day Activity Load
3-Day Activity Load
Fairly Active Minutes
Lightly Active Minutes
Activity Load
```

SHAP values represent **model contribution**, not medical causation.

---

# 📦 Model Bundle

The final production artifacts are packaged into:

```text
athlete_injury_prediction_bundle.pkl
```

The bundle contains:

```text
Injury Risk Model
Onset Model
Recovery Model

Injury Preprocessor
Onset Preprocessor
Recovery Preprocessor

Optimal Classification Threshold

Feature Lists
Model Metrics
Prediction Ranges
Model Version
Training Metadata
```

This allows the inference pipeline to load a single versioned model bundle.

---

# 🧪 Inference Pipeline

During prediction:

```text
Athlete Data
     │
     ▼
Feature Engineering
     │
     ▼
67 Features
     │
     ▼
Saved Preprocessors
     │
     ▼
┌───────────────────────┐
│ Injury Risk Model     │
│ Onset Model           │
│ Recovery Model        │
└───────────┬───────────┘
            │
            ▼
      SHAP Explanation
            │
            ▼
┌───────────────────────┐
│ Final Prediction      │
├───────────────────────┤
│ Risk Score            │
│ Risk Level            │
│ Onset Days            │
│ Recovery Days         │
│ Contributing Factors  │
└───────────────────────┘
```

---

# 🗃️ ML Artifacts

```text
models/
│
├── athlete_injury_prediction_bundle.pkl
│
├── injury_risk_model.pkl
├── onset_day_model.pkl
├── recovery_duration_model.pkl
│
├── injury_preprocessor.pkl
├── onset_preprocessor.pkl
├── recovery_preprocessor.pkl
│
├── model_metadata.pkl
└── shap_injury_feature_importance.csv
```

### Production Artifact

The recommended artifact for inference is:

```text
athlete_injury_prediction_bundle.pkl
```

Older experimental artifacts should not be used for production inference.

---

# 🛠️ Tech Stack

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | ML development             |
| Pandas       | Data processing            |
| NumPy        | Numerical computation      |
| Scikit-learn | Preprocessing & evaluation |
| XGBoost      | Machine learning models    |
| SHAP         | Model explainability       |
| Matplotlib   | Visualization              |
| Seaborn      | Exploratory analysis       |
| PyTorch      | ML/GPU environment         |
| Google Colab | Model training             |

---

# 📈 Training Configuration

The final XGBoost models use approximately:

```text
n_estimators = 400
max_depth = 4
learning_rate = 0.035
subsample = 0.85
colsample_bytree = 0.85
random_state = 42
```

The classification model uses:

```text
objective = binary:logistic
```

Regression models use:

```text
objective = reg:squarederror
```

---

# 📁 Training Dataset Size

The project uses approximately:

```text
Athletes              3,000
Daily Activity        180,000 records
Sleep                 180,000 records
Training Sessions     112,108 records
Weight Logs             7,452 records
```

Injury labels:

```text
Non-Injured     1,950
Injured         1,050
```

---

# 🔮 Future Improvements

Potential improvements include:

* Larger real-world athlete datasets
* Sport-specific models
* Better recovery-duration prediction
* Time-series models
* Transformer-based athlete modeling
* Probability calibration
* More precise temporal labels
* Injury-type prediction
* Personalized athlete baselines
* Model drift monitoring
* Automated retraining
* Additional physiological/recovery features

---

# ⚠️ Limitations

This model is intended for **sports analytics and research purposes**.

It does not provide medical diagnosis.

Prediction quality depends on:

* Dataset quality
* Athlete data completeness
* Accuracy of training labels
* Representativeness of the training population
* Temporal consistency of the available data

The recovery-duration model currently has relatively low R² compared with the other components.

---

# 📌 Model Version

```text
Sportsense ML
Version: v1.0
```

---

# ⚖️ Disclaimer

Sportsense generates **AI-based statistical predictions from athlete data**.

The outputs should be interpreted as predictive analytics rather than medical diagnosis or medical advice.

---

## 🧠 Project Summary

```text
RAW ATHLETE DATA
       ↓
PREPROCESSING
       ↓
TEMPORAL ALIGNMENT
       ↓
FEATURE ENGINEERING
       ↓
67 FEATURES
       ↓
XGBOOST
 ┌─────┼─────┐
 ↓     ↓     ↓
RISK  ONSET  RECOVERY
 ↓     ↓     ↓
 └─────┼─────┘
       ↓
      SHAP
       ↓
EXPLAINABLE PREDICTION
       ↓
MODEL BUNDLE
```

**Sportsense ML — Turning athlete data into explainable injury-risk insights.**
