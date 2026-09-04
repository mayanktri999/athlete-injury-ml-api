
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import os


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Athlete Injury Prediction API",
    description="ML API for athlete injury risk prediction",
    version="1.2"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "athlete_injury_prediction_bundle.pkl"
)

bundle = joblib.load(MODEL_PATH)

injury_model = bundle["injury_model"]
injury_preprocessor = bundle["injury_preprocessor"]

shap_explainer = shap.TreeExplainer(injury_model)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AthleteRequest(BaseModel):
    athlete_id: str
    features: dict


# ============================================================
# FEATURE NAME MAPPING
# ============================================================

FEATURE_NAMES = {

    "activity_load": "Activity Load",
    "activity_load_3d": "3-Day Activity Load",
    "activity_load_7d": "7-Day Activity Load",
    "activity_load_14d": "14-Day Activity Load",
    "activity_load_28d": "28-Day Activity Load",

    "distance_3d": "3-Day Distance",
    "distance_7d": "7-Day Distance",
    "distance_14d": "14-Day Distance",
    "distance_28d": "28-Day Distance",

    "active_minutes_3d": "3-Day Active Minutes",
    "active_minutes_7d": "7-Day Active Minutes",
    "active_minutes_14d": "14-Day Active Minutes",
    "active_minutes_28d": "28-Day Active Minutes",

    "sleep_minutes_3d": "3-Day Sleep",
    "sleep_minutes_7d": "7-Day Sleep",
    "sleep_minutes_14d": "14-Day Sleep",
    "sleep_minutes_28d": "28-Day Sleep",

    "sleep_deficit": "Sleep Deficit",
    "sleep_std_7d": "Sleep Consistency",

    "acute_chronic_ratio": "Training Load Ratio",

    "training_load": "Training Load",
    "training_load_3d": "3-Day Training Load",
    "training_load_7d": "7-Day Training Load",
    "training_load_14d": "14-Day Training Load",
    "training_load_28d": "28-Day Training Load",

    "training_frequency_3d": "3-Day Training Frequency",
    "training_frequency_7d": "7-Day Training Frequency",
    "training_frequency_14d": "14-Day Training Frequency",
    "training_frequency_28d": "28-Day Training Frequency",

    "training_load_change": "Training Load Change",

    "VeryActiveMinutes": "Very Active Minutes",
    "FairlyActiveMinutes": "Fairly Active Minutes",
    "LightlyActiveMinutes": "Lightly Active Minutes",

    "TotalSteps": "Total Steps",
    "TotalDistance": "Total Distance",
    "TrackerDistance": "Tracker Distance",

    "Calories": "Calories",

    "height_cm": "Height",
    "weight_kg_baseline": "Weight",
    "age": "Age",
    "years_playing": "Years Playing",
    "prior_season_injury_count": "Previous Season Injuries"
}


def clean_feature_name(name):

    if name is None:
        return "Unknown Factor"

    name = str(name)

    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    return FEATURE_NAMES.get(
        name,
        name.replace("_", " ").title()
    )


# ============================================================
# SHAP
# ============================================================

def get_shap_explanation(X_processed, top_n=5):

    shap_values = shap_explainer.shap_values(
        X_processed
    )

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    feature_names = list(
        injury_preprocessor.get_feature_names_out()
    )

    if len(feature_names) != len(shap_values):
        raise ValueError(
            f"SHAP mismatch: "
            f"{len(feature_names)} feature names vs "
            f"{len(shap_values)} SHAP values"
        )

    explanation = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_values
    })

    increasing = (
        explanation[
            explanation["impact"] > 0
        ]
        .sort_values(
            "impact",
            ascending=False
        )
        .head(top_n)
    )

    reducing = (
        explanation[
            explanation["impact"] < 0
        ]
        .sort_values(
            "impact",
            ascending=True
        )
        .head(top_n)
    )

    return increasing, reducing


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "Athlete Injury Prediction API",
        "version": "1.2"
    }


@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "model_version": bundle["model_version"]
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: AthleteRequest):

    try:

        row = pd.DataFrame([
            request.features
        ])


        # ====================================================
        # INJURY RISK
        # ====================================================

        injury_features = bundle[
            "injury_input_features"
        ]

        X_injury = row.reindex(
            columns=injury_features,
            fill_value=np.nan
        )

        X_injury_processed = injury_preprocessor.transform(
            X_injury
        )

        probability = float(
            injury_model.predict_proba(
                X_injury_processed
            )[0, 1]
        )

        threshold = float(
            bundle["optimal_threshold"]
        )

        prediction = probability >= threshold


        if probability >= threshold:
            risk_level = "HIGH"

        elif probability >= threshold * 0.60:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"


        # ====================================================
        # SHAP FACTORS
        # ====================================================

        increasing, reducing = get_shap_explanation(
            X_injury_processed,
            top_n=5
        )


        increasing_factors = []

        for _, item in increasing.iterrows():

            increasing_factors.append({
                "name": clean_feature_name(
                    item["feature"]
                ),
                "impact": round(
                    float(item["impact"]),
                    4
                )
            })


        reducing_factors = []

        for _, item in reducing.iterrows():

            reducing_factors.append({
                "name": clean_feature_name(
                    item["feature"]
                ),
                "impact": round(
                    float(item["impact"]),
                    4
                )
            })


        # ====================================================
        # ONSET
        # ====================================================

        regression_features = bundle[
            "regression_input_features"
        ]

        X_onset = row.reindex(
            columns=regression_features,
            fill_value=np.nan
        )

        X_onset_processed = bundle[
            "onset_preprocessor"
        ].transform(X_onset)

        onset = float(
            bundle["onset_model"].predict(
                X_onset_processed
            )[0]
        )

        onset = float(
            np.clip(onset, 1, 30)
        )


        # ====================================================
        # RECOVERY
        # ====================================================

        X_recovery = row.reindex(
            columns=regression_features,
            fill_value=np.nan
        )

        X_recovery_processed = bundle[
            "recovery_preprocessor"
        ].transform(X_recovery)

        recovery = float(
            bundle["recovery_model"].predict(
                X_recovery_processed
            )[0]
        )

        recovery = float(
            np.clip(recovery, 5, 20)
        )


        # ====================================================
        # FLUTTER RESPONSE
        # ====================================================

        return {

            "success": True,

            "data": {

                "athlete_id": request.athlete_id,

                "risk": {

                    "score": round(
                        probability * 100,
                        2
                    ),

                    "level": risk_level,

                    "is_at_risk": bool(
                        prediction
                    )
                },

                "prediction": {

                    "onset_days": round(
                        onset,
                        1
                    ),

                    "recovery_days": round(
                        recovery,
                        1
                    )
                },

                "factors": {

                    "increasing":
                        increasing_factors,

                    "reducing":
                        reducing_factors
                },

                "model": {

                    "version":
                        bundle["model_version"]
                }
            }
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
