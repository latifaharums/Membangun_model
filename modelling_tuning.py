import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
from dagshub import dagshub_logger

# Load envi
load_dotenv()
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

# Setup MLflow ke DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(f"https://dagshub.com/{username}/Membangun_model.mlflow")
mlflow.set_experiment("WineQuality_LogisticRegression_Tuning")

def modeling_with_tuning(X_train_path, X_val_path, y_train_path, y_val_path):
    # Load data hasil split
    X_train = pd.read_csv(X_train_path)
    X_val = pd.read_csv(X_val_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_val = pd.read_csv(y_val_path).squeeze()

    # Grid search parameter Logistic Regression
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10],
        "penalty": ["l2"],
        "solver": ["liblinear", "lbfgs"]
    }

    model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    print("Akurasi:", accuracy)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    return best_model, accuracy, report, grid_search.best_params_

if __name__ == "__main__":
    # Path ke dataset hasil preprocessing
    X_train_path = "dataset_preprocessing/X_train.csv"
    X_val_path = "dataset_preprocessing/X_val.csv"
    y_train_path = "dataset_preprocessing/y_train.csv"
    y_val_path = "dataset_preprocessing/y_val.csv"

    with mlflow.start_run(run_name="LogReg_Tuning"):
        model, accuracy, report, best_params = modeling_with_tuning(
            X_train_path, X_val_path, y_train_path, y_val_path
        )

        # Log hyperparameter terbaik
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log metrik evaluasi
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Set tag MLflow
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_type", "LogisticRegression")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="logreg_best_model")

        print("Tuning selesai dan model berhasil dicatat ke MLflow DagsHub")
