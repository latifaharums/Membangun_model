#Library

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def modeling_with_autolog(data_path):
    # Load dataset red wine 
    df = pd.read_csv(data_path)

    # Memisahkan fitur dan target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Split menjadi Train + Temp (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Split Temp menjadi Validation + Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Model Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Melatih model
    model.fit(X_train, y_train)

    # Evaluasi di Validation
    y_val_pred = model.predict(X_val)
    print("=== VALIDATION EVALUATION ===")
    print("Akurasi:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # Evaluasi di Test
    y_test_pred = model.predict(X_test)
    print("=== TEST EVALUATION ===")
    print("Akurasi:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    return model

#ML Flow
if __name__ == "__main__":
    data_path = "processed_winequality-red.csv"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Wine_Quality_LogReg_Model")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="LogisticRegression_autolog"):
        trained_model = modeling_with_autolog(data_path)
