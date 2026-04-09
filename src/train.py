import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import joblib
import os

def train(data_path: str = "dataset/cleaned.csv", model_type: str = "logistic"):

    df = pd.read_csv(data_path)
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fake-news-detection")

    with mlflow.start_run(run_name=f"{model_type}_run"):

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Build pipeline
        if model_type == "logistic":
            classifier = LogisticRegression(max_iter=1000, C=1.0)
            mlflow.log_param("C", 1.0)
        else:
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            mlflow.log_param("n_estimators", 100)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', classifier)
        ])

        # Train
        print(f"Training {model_type} model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        # Save model to MLflow
        mlflow.sklearn.log_model(pipeline, "model",
            registered_model_name="FakeNewsDetector")

        # Also save locally for the API
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/model.pkl")
        print("Model saved to models/model.pkl")

        return accuracy, f1

if __name__ == "__main__":
    # Train both models and compare
    print("=== Logistic Regression ===")
    train(model_type="logistic")

    print("\n=== Random Forest ===")
    train(model_type="random_forest")