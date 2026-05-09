# 🧠 Parkinson's Disease Severity Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A production-style REST API that predicts Parkinson's Disease motor severity 
(UPDRS score) from biomedical voice measurements, with SHAP-based explainability 
for individual predictions.

> **Author:** Soumita Chowdhury | Senior ML Engineer  
> **Portfolio project** — demonstrates regression, explainability, and 
> clinical tool design

> ⚠️ **Disclaimer:** This tool is for research and portfolio demonstration 
> only. It is not intended for clinical diagnosis. 

---

## Problem Statement

Parkinson's Disease affects motor control progressively over time. The Unified 
Parkinson's Disease Rating Scale (UPDRS) is the standard clinical measure of 
severity — but scoring requires specialist clinical assessment.

This API explores whether biomedical voice measurements can predict motor UPDRS 
scores remotely, enabling potential large-scale screening without specialist 
equipment. It accepts 19 voice and demographic features and returns:

- Predicted motor UPDRS score
- Clinical severity category (Minimal / Mild / Moderate / Severe / Very Severe)
- Severity description
- Abnormal feature detection vs dataset averages

---

## Severity Categories

| Category | Motor UPDRS Range | Description |
|---|---|---|
| Minimal | 0–10 | Very mild or no motor impairment |
| Mild | 11–20 | Mild motor impairment |
| Moderate | 21–32 | Moderate motor impairment |
| Severe | 33–44 | Severe motor impairment |
| Very Severe | 45+ | Very severe motor impairment |

---

## Architecture

Voice Measurements + Demographics
│
▼
[Preprocessing]
│  drop subject ID · StandardScaler normalisation
▼
[RandomForest Regressor]
│  100 estimators · trained on 4,700 recordings
▼
[Prediction + Severity Category]
│  motor UPDRS score · severity band · description
▼
[SHAP Explainability]
individual feature contributions per prediction

---

## Dataset

- **Source:** [Parkinson's Disease Telemonitoring — UCI ML Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- **Size:** 5,875 recordings from 42 patients
- **Features:** 19 (age, sex, test_time + 16 biomedical voice measurements)
- **Target:** motor_UPDRS (continuous, range 5.04–39.51)
- **Split:** 80% train / 20% test (4,700 train / 1,175 test)

### Known Limitation — Data Leakage Risk
The dataset contains ~139 recordings per patient on average. A random 
train/test split may place the same patient's recordings in both sets, 
inflating model performance. A production system would use patient-level 
splitting. This limitation is documented transparently here.

---

## Model Selection

7 models were trained and evaluated. **R² was chosen as primary metric** — 
it directly expresses the proportion of variance explained, which is the 
most interpretable measure for a clinical regression task.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **RandomForest** ✅ | **0.9726** | **1.3219** | **0.6209** |
| XGBoost | 0.9525 | 1.7404 | 1.1551 |
| GradientBoosting | 0.7664 | 3.8610 | 3.0644 |
| SVR | 0.3439 | 6.4712 | 5.0467 |
| LinearRegression | 0.1224 | 7.4843 | 6.3534 |
| Ridge | 0.1221 | 7.4855 | 6.3551 |
| Lasso | 0.1176 | 7.5048 | 6.4134 |

### Key findings
- **Linear models failed** (R²~0.12) — confirms highly non-linear 
  relationships between voice features and UPDRS severity
- **RandomForest dominates** with R² 0.9726 and RMSE of 1.32 UPDRS points
- **SVR moderate** despite RBF kernel — complex interactions require 
  ensemble approaches
- Full EDA and model exploration: `notebooks/model_exploration.ipynb`

---

## SHAP Explainability — Critical Finding

SHAP analysis revealed that **age contributes 5x more than any voice feature** 
to predictions (66% feature importance).

This raises an important clinical question: is the model learning voice-based 
Parkinson's biomarkers, or largely learning age-related decline?

| Feature | Importance | Clinical meaning |
|---|---|---|
| age | 66.0% | Severity increases with age |
| DFA | 7.9% | Fractal scaling — vocal complexity degrades with motor impairment |
| test_time | 7.6% | Disease progression over time |
| sex | 6.5% | Gender differences in vocal characteristics |
| Jitter(Abs) | 2.6% | Absolute frequency variation |

**Production recommendation:** Build age-adjusted models or stratify by age 
group to isolate the true voice signal from age-related decline.

---

## Project Structure

parkinsons-severity-predictor/
├── data/                          # Raw dataset (not committed)
├── notebooks/
│   └── model_exploration.ipynb    # EDA + model comparison + SHAP
├── src/
│   ├── init.py
│   ├── preprocess.py              # Data loading + StandardScaler
│   ├── train.py                   # Model training + comparison
│   └── predict.py                 # Inference + severity categorisation
├── models/                        # Saved artifacts (auto-generated)
├── app.py                         # FastAPI application
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container definition
└── README.md

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/soumita20/parkinsons-severity-predictor.git
cd parkinsons-severity-predictor
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
and place as `data/parkinsons_updrs.data`

### 5. Train the model
```bash
python src/train.py
```

### 6. Start the API
```bash
uvicorn app:app --reload
```

### 7. Test the API
Open `http://127.0.0.1:8000/docs` for interactive Swagger UI

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root — confirms API is running |
| GET | `/health` | Health check — confirms model is loaded |
| POST | `/predict` | Predict motor UPDRS severity from voice features |
| POST | `/abnormal-features` | Identify abnormal voice measurements vs dataset averages |

### Example Request
```json
{
  "age": 72.0,
  "sex": 0,
  "test_time": 120.0,
  "jitter_pct": 0.00662,
  "jitter_abs": 0.000034,
  "jitter_rap": 0.00401,
  "jitter_ppq5": 0.00317,
  "jitter_ddp": 0.01204,
  "shimmer": 0.02565,
  "shimmer_db": 0.230,
  "shimmer_apq3": 0.01438,
  "shimmer_apq5": 0.01309,
  "shimmer_apq11": 0.01662,
  "shimmer_dda": 0.04314,
  "nhr": 0.01429,
  "hnr": 21.640,
  "rpde": 0.41888,
  "dfa": 0.54842,
  "ppe": 0.16006
}
```

### Example Response
```json
{
  "motor_updrs_prediction": 24.5,
  "severity_category": "Moderate",
  "severity_description": "Moderate motor impairment detected. Voice measurements suggest moderate Parkinson's motor symptoms. Clinical evaluation is strongly recommended.",
  "input_features": {...}
}
```

---

## Docker

```bash
# Build image
docker build -t parkinsons-severity-predictor .

# Run container
docker run -p 8000:8000 parkinsons-severity-predictor
```

---

## Roadmap

- [ ] Patient-level train/test split to address data leakage
- [ ] Age-adjusted models to isolate voice signal
- [ ] Confidence intervals for predictions
- [ ] Interactive frontend with sliders for each voice feature
- [ ] Model card with full clinical limitations
- [ ] Multimodal extension — gait + handwriting features

---

## Author

**Soumita Chowdhury**
Senior ML Engineer | NLP · Conversational AI · Azure
[GitHub](https://github.com/soumita20) |
[LinkedIn](https://www.linkedin.com/in/soumita-chowdhury-93934617/)