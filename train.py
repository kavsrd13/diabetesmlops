import pandas as pd
import mlflow
import dagshub
import os
import json

# PyCaret imports
from pycaret.classification import setup, compare_models, pull, save_model

# 🔹 MLflow + Dagshub setup
dagshub.init("diabetes-mlops", "krishna.dwivedi1618", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/krishna.dwivedi1618/diabetes-mlops.mlflow")
mlflow.set_experiment("Diabetes-Prediction")

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Setup PyCaret
s = setup(
    data=df,
    target="diabetic",
    session_id=42,
    log_experiment=True,
    experiment_name="Diabetes-Prediction-pycaret",
    log_plots=True,
    log_data=True
)

# Compare multiple models automatically
best_model = compare_models()

# Pull results table for logging
results = pull()
print("📊 Best Model Results:")
print(results)

# 🔹 Save locally
os.makedirs("models", exist_ok=True)
save_model(best_model, "models/best_model")
print("✅ Best model saved at models/best_model.pkl")

# 🔹 Extract best model metrics
# results is a dataframe with all models compared
# we take the first row (best model)
best_model_metrics = results.iloc[0].to_dict()

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

# Save metrics to JSON
with open("reports/train_metrics.json", "w") as f:
    json.dump(best_model_metrics, f, indent=4)

print("✅ Best model and metrics saved successfully")
