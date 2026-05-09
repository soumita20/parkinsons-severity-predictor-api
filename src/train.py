import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "LinearRegression":   LinearRegression(),
    "Ridge":              Ridge(alpha=1.0),
    "Lasso":              Lasso(alpha=0.1),
    "SVR":                SVR(kernel="rbf", C=1.0),
    "RandomForest":       RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":            XGBRegressor(n_estimators=100,
                                       eval_metric="rmse",
                                       random_state=42),
}


def train_and_compare(X_train, X_test, y_train, y_test):
    """
    Train all models, evaluate on test set.
    Returns results dataframe, trained models dict, and best model.
    Primary metric: R² score (higher is better)
    Secondary metric: RMSE (lower is better)
    """
    train_results  = []
    results        = []
    trained_models = {}

    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        train_results.append({
            "Model": name,
            "Training R²":round(r2_train,4),
            "Training RMSE":  round(train_rmse, 4),
            "Training MAE":   round(train_mae, 4),
        })
        
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        results.append({
            "Model": name,
            "Training R²":round(r2_train,4),
            "Test R²":    round(r2, 4),
            "Training RMSE":  round(train_rmse, 4),
            "Test RMSE":  round(rmse, 4),
            "Training MAE":   round(train_mae, 4),
            "Test MAE":   round(mae, 4)
        })

        trained_models[name] = model
        
    results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False)

    print("\n──────────────────────────────────── Model Comparison ─────────────────────────────────────────")
    print(results_df.to_string(index=False))

    # Best model by R²
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n✅ Best model: {best_name}")

    return best_model, best_name, trained_models, results_df


def save_artifacts(model, scaler, feature_names, model_name,
                   models_dir="models"):
    """
    Save best model, scaler, and feature names.
    """
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model,         os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print(f"\n💾 Saved: {model_name} → {models_dir}/")


def get_detailed_report(model, X_test, y_test, feature_names):
    """
    Return predictions and feature importance for best model.
    """
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)

    print(f"── Detailed Report ──────────────────────────────────────────")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # Feature importance for tree-based models
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print(f"\nTop 10 Features:\n{importance_df.head(10).to_string(index=False)}")
    else:
        importance_df = None

    return y_pred, r2, rmse, mae, importance_df


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_clean, split_data

    df = load_and_clean("data/parkinsons_updrs.data")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_data(df)
    best_model, best_name, trained_models, results_df = train_and_compare(
        X_train, X_test, y_train, y_test
    )
    save_artifacts(best_model, scaler, feature_names, best_name)