"""Train and persist the diabetes classifier (mirrors Diabetes_Classifier.ipynb)."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
MODEL_PATH = "models/diabetes_ann_model.keras"
SKLEARN_MODEL_PATH = "models/diabetes_mlp_model.joblib"
PREPROCESSING_PATH = "models/diabetes_preprocessing.joblib"
METRICS_PATH = "models/model_metrics.joblib"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    local_path = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        try:
            import kagglehub

            path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
            df = pd.read_csv(f"{path}/diabetes.csv")
        except Exception:
            url = (
                "https://raw.githubusercontent.com/plotly/datasets/master/"
                "diabetes.csv"
            )
            df = pd.read_csv(url)

    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return x, y


def _build_metrics(y_test, test_probs, test_preds) -> dict:
    return {
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "classification_report": classification_report(y_test, test_preds, output_dict=True),
        "feature_cols": FEATURE_COLS,
        "threshold": 0.5,
    }


def train_with_tensorflow(x_train, x_test, y_train, y_test) -> dict:
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.keras.utils.set_random_seed(SEED)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = Sequential(
        [
            Input(shape=(x_train.shape[1],)),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.15),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.10),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=250,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1,
    )

    test_loss, test_accuracy, test_auc = model.evaluate(x_test, y_test, verbose=0)
    test_probs = model.predict(x_test, verbose=0).ravel()
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = _build_metrics(y_test, test_probs, test_preds)
    metrics["backend"] = "tensorflow"
    metrics["test_loss"] = float(test_loss)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    return metrics


def train_with_sklearn(x_train, x_test, y_train, y_test) -> dict:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=16,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=SEED,
    )
    model.fit(x_train, y_train)

    test_probs = model.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = _build_metrics(y_test, test_probs, test_preds)
    metrics["backend"] = "sklearn"

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, SKLEARN_MODEL_PATH)
    return metrics


def train() -> dict:
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=SEED, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    try:
        metrics = train_with_tensorflow(x_train, x_test, y_train, y_test)
        print("Trained with TensorFlow/Keras")
    except ImportError:
        print("TensorFlow not available — using scikit-learn MLP fallback")
        metrics = train_with_sklearn(x_train, x_test, y_train, y_test)

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "threshold": 0.5,
            "zero_cols": ZERO_COLS,
            "feature_cols": FEATURE_COLS,
            "backend": metrics["backend"],
        },
        PREPROCESSING_PATH,
    )
    joblib.dump(metrics, METRICS_PATH)

    print(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"ROC-AUC Score: {metrics['test_auc']:.4f}")
    print(f"Saved preprocessing: {PREPROCESSING_PATH}")
    return metrics


if __name__ == "__main__":
    train()
