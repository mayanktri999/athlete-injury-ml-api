import os
import math
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import (
    FastAPI,
    HTTPException,
    Header
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BUNDLE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "athlete_injury_prediction_bundle.pkl"
)


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

if not os.path.exists(MODEL_BUNDLE_PATH):
    raise FileNotFoundError(
        f"Model bundle not found: {MODEL_BUNDLE_PATH}"
    )

logger.info("Loading model bundle...")

bundle = joblib.load(MODEL_BUNDLE_PATH)

injury_model = bundle["injury_model"]
onset_model = bundle["onset_model"]
recovery_model = bundle["recovery_model"]

injury_preprocessor = bundle["injury_preprocessor"]
onset_preprocessor = bundle["onset_preprocessor"]
recovery_preprocessor = bundle["recovery_preprocessor"]

OPTIMAL_THRESHOLD = bundle["optimal_threshold"]

CUTOFF_DATE = bundle["cutoff_date"]

INJURY_INPUT_FEATURES = bundle["injury_input_features"]
REGRESSION_INPUT_FEATURES = bundle["regression_input_features"]

MODEL_VERSION = bundle["model_version"]

METRICS = bundle["metrics"]

PREDICTION_RANGES = bundle["prediction_ranges"]


logger.info(
    f"Model bundle loaded successfully. Version: {MODEL_VERSION}"
)

logger.info(
    f"Injury features: {len(INJURY_INPUT_FEATURES)}"
)

logger.info(
    f"Regression features: {len(REGRESSION_INPUT_FEATURES)}"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Athlete Injury Prediction API",
    description="ML API for athlete injury risk prediction",
    version=str(MODEL_VERSION)
)

# ============================================================
# API SECURITY
# ============================================================

API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    logger.warning(
        "API_KEY environment variable is not configured."
    )


def verify_api_key(
    x_api_key: str | None
):
    """
    Verify API key sent by the client.
    """

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API security is not configured."
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )
# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_NAMES = {

    # Athlete profile
    "sport": "Sport",
    "age": "Age",
    "gender": "Gender",
    "height_cm": "Height",
    "weight_kg_baseline": "Baseline Weight",
    "dominant_side": "Dominant Side",
    "years_playing": "Years Playing",
    "position": "Position",
    "team_id": "Team",
    "prior_season_injury_count": "Previous Season Injuries",

    # Activity
    "activity_load": "Activity Load",
    "activity_load_3d": "3-Day Activity Load",
    "activity_load_7d": "7-Day Activity Load",
    "activity_load_14d": "14-Day Activity Load",
    "activity_load_28d": "28-Day Activity Load",

    "active_minutes": "Active Minutes",
    "active_minutes_3d": "3-Day Active Minutes",
    "active_minutes_7d": "7-Day Active Minutes",
    "active_minutes_14d": "14-Day Active Minutes",
    "active_minutes_28d": "28-Day Active Minutes",

    "FairlyActiveMinutes": "Fairly Active Minutes",
    "LightlyActiveMinutes": "Lightly Active Minutes",
    "VeryActiveMinutes": "Very Active Minutes",
    "SedentaryMinutes": "Sedentary Minutes",

    # Training
    "training_load": "Training Load",
    "training_load_3d": "3-Day Training Load",
    "training_load_7d": "7-Day Training Load",
    "training_load_14d": "14-Day Training Load",
    "training_load_28d": "28-Day Training Load",

    "acute_chronic_ratio": "Training Load Ratio",

    # Sleep
    "bed_minutes": "Bed Minutes",
    "sleep_minutes": "Sleep Duration",
    "sleep_hours": "Sleep Hours",

    "sleep_mean_7d": "7-Day Average Sleep",
    "sleep_mean_14d": "14-Day Average Sleep",
    "sleep_mean_28d": "28-Day Average Sleep",

    "sleep_std_7d": "Sleep Consistency",
    "sleep_std_14d": "14-Day Sleep Consistency",
    "sleep_std_28d": "28-Day Sleep Consistency",

    # Generic rolling features
    "steps": "Steps",
    "steps_3d": "3-Day Steps",
    "steps_7d": "7-Day Steps",
    "steps_14d": "14-Day Steps",
    "steps_28d": "28-Day Steps",

    "calories": "Calories",
    "calories_3d": "3-Day Calories",
    "calories_7d": "7-Day Calories",
    "calories_14d": "14-Day Calories",
    "calories_28d": "28-Day Calories",

    "distance": "Distance",
    "distance_3d": "3-Day Distance",
    "distance_7d": "7-Day Distance",
    "distance_14d": "14-Day Distance",
    "distance_28d": "28-Day Distance",
}


# ============================================================
# REQUEST MODEL
# ============================================================

class AthleteRequest(BaseModel):

    athlete_id: str = Field(
        ...,
        description="Unique athlete ID"
    )

    features: dict = Field(
        ...,
        description="67 model input features"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_feature_name(name):
    """
    Convert model/preprocessor feature names
    into safe human-readable names.
    """

    if name is None:
        return "Unknown Feature"

    name = str(name)

    # Remove transformer prefixes
    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    # Remove one-hot suffixes
    if "_" in name:
        base = name.split("_")[0]

        if base in FEATURE_NAMES:
            name = base

    return FEATURE_NAMES.get(
        name,
        name.replace("_", " ").title()
    )


def make_json_safe(value):
    """
    Convert NumPy/Python values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return round(value, 4)

    return value


def determine_risk_level(probability):

    if probability >= OPTIMAL_THRESHOLD:
        return "HIGH"

    elif probability >= (
        OPTIMAL_THRESHOLD * 0.60
    ):
        return "MEDIUM"

    return "LOW"


def get_shap_factors(features_df, top_n=5):

    """
    Generate SHAP explanations for the injury model.

    These explain factors contributing to the model prediction.
    They should NOT be interpreted as medical causation.
    """

    try:

        transformed = injury_preprocessor.transform(
            features_df
        )

        feature_names = (
            injury_preprocessor
            .get_feature_names_out()
        )

        feature_names = [
            clean_feature_name(name)
            for name in feature_names
        ]

        explainer = shap.TreeExplainer(
            injury_model
        )

        shap_values = explainer.shap_values(
            transformed
        )

        # SHAP can return different structures
        # depending on model/version.
        if isinstance(shap_values, list):

            shap_values = shap_values[-1]

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 2:

            shap_values = shap_values[0]

        shap_values = shap_values.flatten()

        # Make sure lengths match
        n = min(
            len(feature_names),
            len(shap_values)
        )

        factors = []

        for i in range(n):

            value = float(shap_values[i])

            if not math.isfinite(value):
                continue

            factors.append({
                "name": feature_names[i],
                "impact": round(value, 4)
            })

        # Remove duplicate human-readable names
        # while keeping the strongest impact
        grouped = {}

        for factor in factors:

            name = factor["name"]
            impact = factor["impact"]

            if (
                name not in grouped
                or abs(impact)
                > abs(grouped[name]["impact"])
            ):
                grouped[name] = factor

        factors = list(grouped.values())

        increasing = sorted(
            [
                f for f in factors
                if f["impact"] > 0
            ],
            key=lambda x: x["impact"],
            reverse=True
        )[:top_n]

        reducing = sorted(
            [
                f for f in factors
                if f["impact"] < 0
            ],
            key=lambda x: x["impact"]
        )[:top_n]

        return {
            "increasing": increasing,
            "reducing": reducing
        }

    except Exception as e:

        logger.error(
            f"SHAP explanation failed: {str(e)}"
        )

        return {
            "increasing": [],
            "reducing": []
        }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Athlete Injury Prediction API",
        "status": "running",
        "version": MODEL_VERSION
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": MODEL_VERSION
    }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_version": MODEL_VERSION,
        "cutoff_date": str(CUTOFF_DATE),
        "injury_features": len(
            INJURY_INPUT_FEATURES
        ),
        "regression_features": len(
            REGRESSION_INPUT_FEATURES
        ),
        "optimal_threshold": make_json_safe(
            OPTIMAL_THRESHOLD
        ),
        "metrics": METRICS,
        "prediction_ranges": PREDICTION_RANGES
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: AthleteRequest,
    x_api_key: str | None = Header(default=None)
):

    verify_api_key(x_api_key)

    try:

        logger.info(
            f"Prediction request received for athlete "
            f"{request.athlete_id}"
        )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in INJURY_INPUT_FEATURES
            if feature not in request.features
        ]

        if missing_features:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing required features",
                    "missing_features": missing_features
                }
            )

        # ----------------------------------------------------
        # Build injury dataframe
        # ----------------------------------------------------

        injury_data = {}

        for feature in INJURY_INPUT_FEATURES:

            injury_data[feature] = (
                request.features.get(feature)
            )

        injury_df = pd.DataFrame(
            [injury_data]
        )

        # ----------------------------------------------------
        # Injury prediction
        # ----------------------------------------------------

        injury_transformed = (
            injury_preprocessor.transform(
                injury_df
            )
        )

        injury_probability = float(
            injury_model.predict_proba(
                injury_transformed
            )[0][1]
        )

        injury_probability = max(
            0.0,
            min(
                1.0,
                injury_probability
            )
        )

        risk_score = injury_probability * 100

        risk_level = determine_risk_level(
            injury_probability
        )

        is_at_risk = (
            injury_probability
            >= OPTIMAL_THRESHOLD
        )

        # ----------------------------------------------------
        # Regression dataframe
        # ----------------------------------------------------

        regression_data = {}

        for feature in REGRESSION_INPUT_FEATURES:

            regression_data[feature] = (
                request.features.get(feature)
            )

        regression_df = pd.DataFrame(
            [regression_data]
        )

        # ----------------------------------------------------
        # Onset prediction
        # ----------------------------------------------------

        onset_transformed = (
            onset_preprocessor.transform(
                regression_df
            )
        )

        onset_days = float(
            onset_model.predict(
                onset_transformed
            )[0]
        )

        # ----------------------------------------------------
        # Recovery prediction
        # ----------------------------------------------------

        recovery_transformed = (
            recovery_preprocessor.transform(
                regression_df
            )
        )

        recovery_days = float(
            recovery_model.predict(
                recovery_transformed
            )[0]
        )

        # ----------------------------------------------------
        # Clip predictions to trained ranges
        # ----------------------------------------------------

        onset_days = max(
            1.0,
            min(
                30.0,
                onset_days
            )
        )

        recovery_days = max(
            5.0,
            min(
                20.0,
                recovery_days
            )
        )

        # ----------------------------------------------------
        # SHAP explanations
        # ----------------------------------------------------

        factors = get_shap_factors(
            injury_df
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        response = {
            "success": True,

            "data": {

                "athlete_id": request.athlete_id,

                "risk": {
                    "score": round(
                        risk_score,
                        2
                    ),
                    "level": risk_level,
                    "is_at_risk": is_at_risk
                },

                "prediction": {
                    "onset_days": round(
                        onset_days,
                        1
                    ),
                    "recovery_days": round(
                        recovery_days,
                        1
                    )
                },

                "factors": factors,

                "model": {
                    "version": MODEL_VERSION
                }
            }
        }

        logger.info(
            f"Prediction completed for athlete "
            f"{request.athlete_id}: "
            f"{risk_score:.2f}%"
        )

        return response

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"Prediction failed: {str(e)}"
        )

        logger.error(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        ),
        reload=False
    )