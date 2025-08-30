import pandas as pd
import joblib

# Load reference and inference data
train_df = pd.read_csv("data/diabetes.csv")
inference_df = pd.read_csv("data/inference_log.csv")

# Load model
model = joblib.load("models/best_model.pkl")

# Predictions on inference data
inference_df["prediction"] = model.predict(inference_df.drop(columns=["prediction"], errors="ignore"))

# Simple metrics
metrics = {}

for col in ["age", "gender", "blood_pressure", "BMI"]:
    metrics[f"{col}_train_mean"] = train_df[col].mean()
    metrics[f"{col}_inference_mean"] = inference_df[col].mean()
    metrics[f"{col}_mean_diff"] = abs(train_df[col].mean() - inference_df[col].mean())

# Prediction distribution
train_pred_dist = train_df["diabetic"].value_counts(normalize=True).to_dict()
inference_pred_dist = inference_df["prediction"].value_counts(normalize=True).to_dict()

metrics["train_pred_dist"] = train_pred_dist
metrics["inference_pred_dist"] = inference_pred_dist

# Save results
pd.DataFrame([metrics]).to_csv("reports/simple_metrics.csv", index=False)
print("✅ Simple drift metrics saved at reports/simple_metrics.csv")
