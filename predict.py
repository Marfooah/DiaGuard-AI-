"""Inference helpers for the diabetes classifier."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_CANDIDATES = [
    Path("models/diabetes_ann_model.keras"),
    Path("diabetes_ann_model.keras"),
]
SKLEARN_MODEL_CANDIDATES = [
    Path("models/diabetes_mlp_model.joblib"),
    Path("diabetes_mlp_model.joblib"),
]
PREPROCESSING_CANDIDATES = [
    Path("models/diabetes_preprocessing.joblib"),
    Path("diabetes_preprocessing.joblib"),
]
METRICS_CANDIDATES = [
    Path("models/model_metrics.joblib"),
    Path("model_metrics.joblib"),
]
DATA_CANDIDATES = [
    Path("data/diabetes.csv"),
    Path("diabetes.csv"),
]

FEATURE_META = {
    "Pregnancies": {
        "label": "Pregnancies",
        "unit": "times",
        "min": 0,
        "max": 17,
        "default": 1,
        "help": "Number of times pregnant",
        "group": "History",
    },
    "Glucose": {
        "label": "Glucose Level",
        "unit": "mg/dL",
        "min": 44,
        "max": 199,
        "default": 120,
        "help": "Plasma glucose (2-hr oral glucose tolerance test)",
        "group": "Metabolic",
        "optimal": (70, 99),
        "caution": (100, 125),
    },
    "BloodPressure": {
        "label": "Blood Pressure",
        "unit": "mm Hg",
        "min": 24,
        "max": 122,
        "default": 70,
        "help": "Diastolic blood pressure",
        "group": "Vitals",
        "optimal": (60, 80),
        "caution": (81, 89),
    },
    "SkinThickness": {
        "label": "Skin Thickness",
        "unit": "mm",
        "min": 7,
        "max": 99,
        "default": 20,
        "help": "Triceps skin fold thickness",
        "group": "Vitals",
    },
    "Insulin": {
        "label": "Insulin",
        "unit": "μU/mL",
        "min": 0,
        "max": 846,
        "default": 80,
        "help": "2-hour serum insulin (0 = not measured)",
        "group": "Metabolic",
    },
    "BMI": {
        "label": "Body Mass Index",
        "unit": "kg/m²",
        "min": 18.0,
        "max": 67.1,
        "default": 28.0,
        "help": "Weight (kg) divided by height (m) squared",
        "group": "Metabolic",
        "optimal": (18.5, 24.9),
        "caution": (25.0, 29.9),
    },
    "DiabetesPedigreeFunction": {
        "label": "Family History Score",
        "unit": "index",
        "min": 0.078,
        "max": 2.42,
        "default": 0.5,
        "help": "Diabetes pedigree function — genetic influence",
        "group": "History",
    },
    "Age": {
        "label": "Age",
        "unit": "years",
        "min": 21,
        "max": 81,
        "default": 31,
        "help": "Age in years",
        "group": "History",
    },
}

PRESETS = {
    "Healthy Adult": {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.0,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31,
    },
    "Elevated Risk": {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    },
    "Borderline": {
        "Pregnancies": 3,
        "Glucose": 115,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 42,
    },
}

RISK_STYLES = {
    "low": {"label": "Low Risk", "color": "#34d399", "bg": "rgba(16,185,129,0.08)", "border": "rgba(52,211,153,0.35)"},
    "moderate": {"label": "Moderate Risk", "color": "#fbbf24", "bg": "rgba(251,191,36,0.08)", "border": "rgba(251,191,36,0.35)"},
    "high": {"label": "High Risk", "color": "#a78bfa", "bg": "rgba(167,139,250,0.1)", "border": "rgba(167,139,250,0.4)"},
    "very_high": {"label": "Very High Risk", "color": "#f87171", "bg": "rgba(248,113,113,0.1)", "border": "rgba(248,113,113,0.4)"},
}


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def artifacts_exist() -> bool:
    has_model = (
        _first_existing(MODEL_CANDIDATES) is not None
        or _first_existing(SKLEARN_MODEL_CANDIDATES) is not None
    )
    has_prep = _first_existing(PREPROCESSING_CANDIDATES) is not None
    return has_model and has_prep


def load_artifacts():
    prep_path = _first_existing(PREPROCESSING_CANDIDATES)
    preprocessing = joblib.load(prep_path)
    metrics_path = _first_existing(METRICS_CANDIDATES)
    metrics = joblib.load(metrics_path) if metrics_path else {}
    backend = preprocessing.get("backend", metrics.get("backend", "tensorflow"))
    model_path = _first_existing(MODEL_CANDIDATES)
    sklearn_path = _first_existing(SKLEARN_MODEL_CANDIDATES)

    if backend == "sklearn" or (model_path is None and sklearn_path):
        model = joblib.load(sklearn_path)
        backend = "sklearn"
    else:
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        backend = "tensorflow"

    return model, preprocessing, metrics, backend


@lru_cache(maxsize=1)
def dataset_stats() -> dict:
    data_path = _first_existing(DATA_CANDIDATES)
    if not data_path:
        return {}
    df = pd.read_csv(data_path)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    stats = {}
    for col in FEATURE_META:
        series = df[col].dropna()
        stats[col] = {
            "median": float(series.median()),
            "mean": float(series.mean()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
        }
    stats["total_patients"] = len(df)
    stats["diabetes_rate"] = float(df["Outcome"].mean())
    return stats


def preprocess_input(values: dict, preprocessing: dict) -> np.ndarray:
    feature_cols = preprocessing["feature_cols"]
    zero_cols = preprocessing["zero_cols"]

    row = pd.DataFrame([{col: values[col] for col in feature_cols}], columns=feature_cols)
    row[zero_cols] = row[zero_cols].replace(0, np.nan)
    imputed = preprocessing["imputer"].transform(row)
    scaled = preprocessing["scaler"].transform(imputed)
    return scaled


def _predict_probability(model, scaled: np.ndarray, backend: str) -> float:
    if backend == "sklearn":
        return float(model.predict_proba(scaled)[0][1])
    return float(model.predict(scaled, verbose=0)[0][0])


def risk_style(probability: float) -> dict:
    if probability < 0.35:
        key = "low"
    elif probability < 0.55:
        key = "moderate"
    elif probability < 0.75:
        key = "high"
    else:
        key = "very_high"
    style = RISK_STYLES[key].copy()
    style["key"] = key
    style["message"] = {
        "low": "Clinical indicators are within a favorable range.",
        "moderate": "Some markers are above typical levels — lifestyle review is advised.",
        "high": "Multiple indicators suggest elevated diabetes risk.",
        "very_high": "Strong risk signals detected — professional screening is recommended.",
    }[key]
    return style


def feature_status(feature: str, value: float) -> tuple[str, str]:
    meta = FEATURE_META[feature]
    if "optimal" in meta:
        lo, hi = meta["optimal"]
        if lo <= value <= hi:
            return "Optimal", "#34d399"
        if "caution" in meta:
            clo, chi = meta["caution"]
            if clo <= value <= chi:
                return "Caution", "#fbbf24"
        return "Elevated", "#a78bfa"
    stats = dataset_stats().get(feature, {})
    if stats:
        if value <= stats.get("p25", value):
            return "Below avg", "#34d399"
        if value >= stats.get("p75", value):
            return "Above avg", "#a78bfa"
        return "Typical", "#94a3b8"
    return "—", "#64748b"


def feature_impacts(values: dict, model, preprocessing: dict, backend: str) -> list[dict]:
    baseline = _predict_probability(model, preprocess_input(values, preprocessing), backend)
    impacts = []
    stats = dataset_stats()

    for feature in FEATURE_META:
        modified = values.copy()
        modified[feature] = stats.get(feature, {}).get("median", FEATURE_META[feature]["default"])
        counter_prob = _predict_probability(
            model, preprocess_input(modified, preprocessing), backend
        )
        delta = baseline - counter_prob
        status, color = feature_status(feature, values[feature])
        impacts.append(
            {
                "feature": feature,
                "label": FEATURE_META[feature]["label"],
                "value": values[feature],
                "unit": FEATURE_META[feature]["unit"],
                "impact": delta,
                "status": status,
                "color": color,
            }
        )

    impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return impacts


def health_tips(values: dict) -> list[str]:
    tips = []
    if values["Glucose"] >= 126:
        tips.append("Glucose is in the diabetic range — fasting glucose or HbA1c testing is recommended.")
    elif values["Glucose"] >= 100:
        tips.append("Glucose is elevated — consider dietary changes and regular monitoring.")

    if values["BMI"] >= 30:
        tips.append("BMI indicates obesity — weight management can significantly reduce diabetes risk.")
    elif values["BMI"] >= 25:
        tips.append("BMI is in the overweight range — moderate exercise may help.")

    if values["BloodPressure"] >= 90:
        tips.append("Blood pressure is high — cardiovascular and metabolic health are closely linked.")

    if values["Age"] >= 45:
        tips.append("Age is a known risk factor — annual screening is advisable after 45.")

    if values["DiabetesPedigreeFunction"] >= 0.5:
        tips.append("Family history score is elevated — genetic predisposition increases baseline risk.")

    if not tips:
        tips.append("Maintain balanced nutrition, regular activity, and routine check-ups.")
    return tips[:4]


def predict(values: dict, model=None, preprocessing=None, backend=None) -> dict:
    if model is None or preprocessing is None:
        model, preprocessing, metrics, backend = load_artifacts()
    else:
        metrics_path = _first_existing(METRICS_CANDIDATES)
        metrics = joblib.load(metrics_path) if metrics_path else {}

    scaled = preprocess_input(values, preprocessing)
    probability = _predict_probability(model, scaled, backend)
    threshold = preprocessing.get("threshold", 0.5)
    is_diabetic = probability >= threshold
    style = risk_style(probability)

    return {
        "probability": probability,
        "risk_percent": probability * 100,
        "prediction": "Diabetic" if is_diabetic else "Non-Diabetic",
        "is_diabetic": is_diabetic,
        "threshold": threshold,
        "metrics": metrics,
        "backend": backend,
        "style": style,
        "impacts": feature_impacts(values, model, preprocessing, backend),
        "tips": health_tips(values),
    }
