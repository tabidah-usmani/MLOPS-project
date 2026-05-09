import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import joblib
import os

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


def train(data_path: str = "dataset/cleaned.csv", model_type: str = "logistic"):
    """
    Train a single model type and log to MLflow.
    model_type: "logistic" | "random_forest" | "linearsvc"
    Returns (accuracy, f1, pipeline)
    """
    df = pd.read_csv(data_path)
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fake-news-detection")

    with mlflow.start_run(run_name=f"{model_type}_run"):

        # ── Parameters ────────────────────────────
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("sublinear_tf", True)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # ── Improved TF-IDF ───────────────────────
        # sublinear_tf=True  → better for short text (log scaling)
        # min_df=2           → ignore very rare terms
        # max_df=0.95        → ignore terms in almost every doc
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            analyzer='word'
        )

        # ── Model Selection ───────────────────────
        if model_type == "logistic":
            classifier = LogisticRegression(
                max_iter=1000, C=1.0, random_state=42
            )
            mlflow.log_param("C", 1.0)
            mlflow.log_param("max_iter", 1000)

        elif model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            mlflow.log_param("n_estimators", 100)

        elif model_type == "linearsvc":
            # CalibratedClassifierCV wraps LinearSVC so it
            # supports predict_proba (required by Flask API)
            classifier = CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=42)
            )
            mlflow.log_param("max_iter", 2000)
            mlflow.log_param("note", "LinearSVC wrapped with CalibratedClassifierCV for predict_proba support")

        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose logistic | random_forest | linearsvc")

        # ── Build Pipeline ────────────────────────
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])

        # ── Train ─────────────────────────────────
        print(f"Training {model_type} model...")
        pipeline.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # ── Log Metrics ───────────────────────────
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

        # ── Log Model to MLflow Registry ──────────
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name="FakeNewsDetector"
        )

        return accuracy, f1, pipeline


if __name__ == "__main__":

    DATA_PATH = "dataset/cleaned.csv"

    results = {}

    # ── Train All 3 Models ─────────────────────────────────────
    print("=" * 50)
    print("=== Logistic Regression ===")
    acc, f1, pipe = train(data_path=DATA_PATH, model_type="logistic")
    results["logistic"] = {"accuracy": acc, "f1": f1, "pipeline": pipe}

    print("\n" + "=" * 50)
    print("=== Random Forest ===")
    acc, f1, pipe = train(data_path=DATA_PATH, model_type="random_forest")
    results["random_forest"] = {"accuracy": acc, "f1": f1, "pipeline": pipe}

    print("\n" + "=" * 50)
    print("=== LinearSVC ===")
    acc, f1, pipe = train(data_path=DATA_PATH, model_type="linearsvc")
    results["linearsvc"] = {"accuracy": acc, "f1": f1, "pipeline": pipe}

    # ── Compare All Models ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("=== Model Comparison ===")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

    # ── Auto-Select Best Model by F1 ──────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]

    print("\n" + "=" * 50)
    print(f"Best model: {best_name} (F1={best['f1']:.4f})")

    # ── Save Best Model for API ────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["pipeline"], "models/model.pkl")
    print("Best model saved to models/model.pkl")
