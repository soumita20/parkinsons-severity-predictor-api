import joblib
import numpy as np
import pandas as pd


def load_artifacts(models_dir="models"):
    """
    Load saved model, scaler, and feature names.
    Called once at API startup — not on every request.
    """
    model         = joblib.load(f"{models_dir}/best_model.pkl")
    scaler        = joblib.load(f"{models_dir}/scaler.pkl")
    feature_names = joblib.load(f"{models_dir}/feature_names.pkl")
    return model, scaler, feature_names


def predict_severity(features: dict, model, scaler, feature_names) -> dict:
    """
    Given a dictionary of voice measurements, return:
    - predicted motor_UPDRS score
    - severity category based on score
    - which features are most abnormal compared to dataset averages

    Args:
        features: dict of {feature_name: value}
        model: trained regression model
        scaler: fitted StandardScaler
        feature_names: list of feature names in correct order

    Returns:
        {
            "motor_updrs_prediction": 24.5,
            "severity_category": "Moderate",
            "severity_description": "...",
            "input_features": {...}
        }
    """
    # Validate all required features are present
    missing = [f for f in feature_names if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Build input array in correct feature order
    input_array = np.array([[features[f] for f in feature_names]])

    # Scale input — must use same scaler fitted on training data
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = float(model.predict(input_scaled)[0])
    prediction = round(prediction, 2)

    # Categorise severity based on UPDRS motor score ranges
    # Clinical reference: MDS-UPDRS motor score interpretation
    severity_category, severity_description = categorise_severity(prediction)

    return {
        "motor_updrs_prediction": prediction,
        "severity_category":      severity_category,
        "severity_description":   severity_description,
        "input_features":         features
    }


def categorise_severity(score: float) -> tuple:
    """
    Categorise motor UPDRS score into clinical severity bands.

    Motor UPDRS ranges (approximate clinical reference):
    0-10:  Minimal — very mild or no motor impairment
    11-20: Mild — mild motor impairment
    21-32: Moderate — moderate motor impairment
    33-44: Severe — severe motor impairment
    45+:   Very Severe — very severe motor impairment
    """
    if score <= 10:
        return (
            "Minimal",
            "Very mild or no motor impairment detected. "
            "Voice measurements suggest minimal Parkinson's motor symptoms."
        )
    elif score <= 20:
        return (
            "Mild",
            "Mild motor impairment detected. "
            "Voice measurements suggest early-stage motor symptoms. "
            "Clinical evaluation is recommended."
        )
    elif score <= 32:
        return (
            "Moderate",
            "Moderate motor impairment detected. "
            "Voice measurements suggest moderate Parkinson's motor symptoms. "
            "Clinical evaluation is strongly recommended."
        )
    elif score <= 44:
        return (
            "Severe",
            "Severe motor impairment detected. "
            "Voice measurements suggest significant Parkinson's motor symptoms. "
            "Immediate clinical evaluation is recommended."
        )
    else:
        return (
            "Very Severe",
            "Very severe motor impairment detected. "
            "Voice measurements suggest advanced Parkinson's motor symptoms. "
            "Immediate clinical evaluation is required."
        )


# Dataset average values for reference
# Used by the API to show which inputs are abnormal
FEATURE_AVERAGES = {
    "age":            64.8,
    "sex":             0.68,
    "test_time":      92.86,
    "Jitter(%)":       0.00622,
    "Jitter(Abs)":     0.0000441,
    "Jitter:RAP":      0.00311,
    "Jitter:PPQ5":     0.00349,
    "Jitter:DDP":      0.00932,
    "Shimmer":         0.03401,
    "Shimmer(dB)":     0.31,
    "Shimmer:APQ3":    0.01685,
    "Shimmer:APQ5":    0.02072,
    "Shimmer:APQ11":   0.02776,
    "Shimmer:DDA":     0.05054,
    "NHR":             0.02971,
    "HNR":            21.68,
    "RPDE":            0.54136,
    "DFA":             0.65354,
    "PPE":             0.21954
}

# Binary features — percentage deviation doesn't apply
BINARY_FEATURES = {'sex'}