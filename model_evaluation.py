
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Create output directory
output_dir = "data/processed"

# Load model comparison and predictions
comparison = pd.read_csv(f"{output_dir}/model_comparison.csv")
predictions = pd.read_csv(f"{output_dir}/model_predictions.csv")
predictions["Date"] = pd.to_datetime(predictions["Date"])

# Load trained model
model = joblib.load(f"{output_dir}/best_demand_model.pkl")

# Calculate residuals
predictions["Residual"] = (
    predictions["Actual Demand"] - predictions["Predicted Demand"]
)

print("\nMODEL EVALUATION")
print("MAE:", round(mean_absolute_error(
    predictions["Actual Demand"],
    predictions["Predicted Demand"]
), 4))

print("RMSE:", round(root_mean_squared_error(
    predictions["Actual Demand"],
    predictions["Predicted Demand"]
), 4))


# 1. Actual vs Predicted Demand
daily_comparison = (
    predictions.groupby("Date")[
        ["Actual Demand", "Predicted Demand"]
    ].sum().reset_index()
)

plt.figure(figsize=(12, 5))
plt.plot(
    daily_comparison["Date"],
    daily_comparison["Actual Demand"],
    label="Actual Demand"
)
plt.plot(
    daily_comparison["Date"],
    daily_comparison["Predicted Demand"],
    label="Predicted Demand"
)
plt.title("Actual vs Predicted Daily Demand")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/actual_vs_predicted.png")
plt.close()


# 2. Model Performance Comparison
plt.figure(figsize=(10, 5))
plt.bar(comparison["Model"], comparison["RMSE"])
plt.title("Model Performance Comparison (RMSE)")
plt.xlabel("Model")
plt.ylabel("RMSE - Lower is Better")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{output_dir}/model_performance.png")
plt.close()


# 3. Residual Error Distribution
plt.figure(figsize=(9, 5))
plt.hist(predictions["Residual"], bins=30, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.title("Residual Error Distribution")
plt.xlabel("Actual Demand - Predicted Demand")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_dir}/residual_distribution.png")
plt.close()


# 4. Linear Regression Coefficients
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": regressor.coef_
})

coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute Coefficient",
    ascending=False
)

coefficients.to_csv(
    f"{output_dir}/feature_coefficients.csv",
    index=False
)

top_features = coefficients.head(15).sort_values(
    "Coefficient"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)
plt.title("Top 15 Linear Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_coefficients.png")
plt.close()


print("\nEVALUATION COMPLETED!")
print("Generated:")
print("- actual_vs_predicted.png")
print("- model_performance.png")
print("- residual_distribution.png")
print("- feature_coefficients.png")
print("- feature_coefficients.csv")
