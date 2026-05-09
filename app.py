from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from src.predict import load_artifacts, predict_severity, FEATURE_AVERAGES, BINARY_FEATURES

# ── Lifespan: load model once at startup ────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML artifacts once when API starts.
    Keeps model in memory for all subsequent requests.
    """
    print("Loading ML artifacts...")
    ml_models["model"], ml_models["scaler"], ml_models["feature_names"] = \
        load_artifacts()
    print("✅ Model loaded successfully.")
    yield
    ml_models.clear()


# ── App initialisation ───────────────────────────────────────────────────────
app = FastAPI(
    title="Parkinson's Disease Severity Predictor",
    description="""
    A REST API that predicts Parkinson's Disease motor severity (UPDRS score)
    from biomedical voice measurements.

    Input voice features → Get predicted motor UPDRS score + severity category.

    Built by Soumita Chowdhury — Senior ML Engineer portfolio project.

    ⚠️ This tool is for research purposes only and is not intended for
    clinical diagnosis. 
    """,
    version="1.0.0",
    lifespan=lifespan
)


# ── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:            float = Field(..., example=65.0,      description="Age in years")
    sex:            float = Field(..., example=0.0,       description="Sex: 0=Male, 1=Female")
    test_time:      float = Field(..., example=92.0,      description="Days since recruitment")
    jitter_pct:     float = Field(..., example=0.00622,   description="MDVP:Jitter(%)")
    jitter_abs:     float = Field(..., example=0.0000441, description="MDVP:Jitter(Abs)")
    jitter_rap:     float = Field(..., example=0.00311,   description="MDVP:RAP")
    jitter_ppq5:    float = Field(..., example=0.00349,   description="MDVP:PPQ5")
    jitter_ddp:     float = Field(..., example=0.00932,   description="Jitter:DDP")
    shimmer:        float = Field(..., example=0.03401,   description="MDVP:Shimmer")
    shimmer_db:     float = Field(..., example=0.31,      description="MDVP:Shimmer(dB)")
    shimmer_apq3:   float = Field(..., example=0.01685,   description="Shimmer:APQ3")
    shimmer_apq5:   float = Field(..., example=0.02072,   description="Shimmer:APQ5")
    shimmer_apq11:  float = Field(..., example=0.02776,   description="Shimmer:APQ11")
    shimmer_dda:    float = Field(..., example=0.05054,   description="Shimmer:DDA")
    nhr:            float = Field(..., example=0.02971,   description="NHR")
    hnr:            float = Field(..., example=21.68,     description="HNR")
    rpde:           float = Field(..., example=0.54136,   description="RPDE")
    dfa:            float = Field(..., example=0.65354,   description="DFA")
    ppe:            float = Field(..., example=0.21954,   description="PPE")


class PredictResponse(BaseModel):
    motor_updrs_prediction: float
    severity_category:      str
    severity_description:   str
    input_features:         dict


class AbnormalFeaturesResponse(BaseModel):
    abnormal_features: dict
    message:           str


# ── Helper ───────────────────────────────────────────────────────────────────
def map_request_to_features(request: PredictRequest) -> dict:
    """
    Map API request field names to dataset feature names.
    Pydantic uses clean names (jitter_pct) but model expects
    original names (Jitter(%)).
    """
    return {
        "age":            request.age,
        "sex":            request.sex,
        "test_time":      request.test_time,
        "Jitter(%)":      request.jitter_pct,
        "Jitter(Abs)":    request.jitter_abs,
        "Jitter:RAP":     request.jitter_rap,
        "Jitter:PPQ5":    request.jitter_ppq5,
        "Jitter:DDP":     request.jitter_ddp,
        "Shimmer":        request.shimmer,
        "Shimmer(dB)":    request.shimmer_db,
        "Shimmer:APQ3":   request.shimmer_apq3,
        "Shimmer:APQ5":   request.shimmer_apq5,
        "Shimmer:APQ11":  request.shimmer_apq11,
        "Shimmer:DDA":    request.shimmer_dda,
        "NHR":            request.nhr,
        "HNR":            request.hnr,
        "RPDE":           request.rpde,
        "DFA":            request.dfa,
        "PPE":            request.ppe
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Parkinson's Disease Severity Predictor API is running.",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health")
def health():
    """Health check endpoint — confirms model is loaded."""
    if not ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status":       "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts biomedical voice measurements and returns:
    - Predicted motor UPDRS score
    - Severity category (Minimal/Mild/Moderate/Severe/Very Severe)
    - Severity description with clinical guidance
    """
    try:
        features = map_request_to_features(request)
        result   = predict_severity(
            features=features,
            model=ml_models["model"],
            scaler=ml_models["scaler"],
            feature_names=ml_models["feature_names"]
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/abnormal-features")
def abnormal_features(request: PredictRequest):
    """
    Compares input voice measurements against dataset averages.
    Returns features that deviate significantly (>50%) from average.
    Helps clinicians understand which voice characteristics are abnormal.
    """
    features   = map_request_to_features(request)
    abnormal   = {}

    for feature, value in features.items():
        if feature in FEATURE_AVERAGES and feature not in BINARY_FEATURES:
            avg      = FEATURE_AVERAGES[feature]
            deviation = abs(value - avg) / avg * 100
            if deviation > 50:
                abnormal[feature] = {
                    "input_value":     round(value, 6),
                    "dataset_average": round(avg, 6),
                    "deviation_pct":   round(deviation, 1)
                }

    message = (
        f"{len(abnormal)} feature(s) deviate significantly from dataset averages."
        if abnormal
        else "All features are within normal range of dataset averages."
    )

    return {
        "abnormal_features": abnormal,
        "message":           message
    }