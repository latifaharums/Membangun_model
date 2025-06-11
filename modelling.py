import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set MLflow tracking lokal
local_mlflow_dir = os.path.abspath("mlruns_local").replace("\\", "/")
mlflow.set_tracking_uri(f"file:///{local_mlflow_dir}")
mlflow.set_experiment("WineQuality_RandomForest_Basic")

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load data
data_path = "processed_winequality-red.csv"
data = pd.read_csv(data_path)
X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="RandomForest_AutoLog"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Akurasi: {acc}")
    print(classification_report(y_val, y_pred, zero_division=0))
