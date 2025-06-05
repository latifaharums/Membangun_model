#Library

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def modeling_with_tuning(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Split Train (70%) dan Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Split Temp menjadi Validation (15%) dan Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Gabungin Train + Validation buat tunning
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    # Parameter grid buat Logistic Regression
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"],
        "max_iter": [500, 1000]
    }

    #Membangun model LR
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_trainval, y_trainval)

    best_model = grid_search.best_estimator_

    # Evaluasi di Test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Akurasi:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, accuracy, report, grid_search.best_params_

#MLFlow
if __name__ == "__main__":
    data_path = "processed_winequality-red.csv"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("WineQuality_LogisticRegression_Tuning")

    with mlflow.start_run(run_name="LogReg_Tuning"):
        model, accuracy, report, best_params = modeling_with_tuning(data_path)

        # Log parameter hasil tuning
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log metrik evaluasi
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Logging tags
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_type", "LogisticRegression")

        # Logging model ke MLflow
        mlflow.sklearn.log_model(model, artifact_path="logreg_best_model")

        print("Proses tuning dan logging MLflow selesai.")
