
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# 1. Load prepared dataset
df = pd.read_csv("data/processed/model_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 2. Define features and target
categorical_features = ["Product", "Variant / Pack Size"]

numeric_features = [
    "Day",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "Demand_Lag_1",
    "Demand_Lag_7",
    "Rolling_Mean_7"
]

features = categorical_features + numeric_features
target = "Quantity Sold"

X = df[features]
y = df[target]

# 3. Chronological train-test split
dates = sorted(df["Date"].unique())
split_index = int(len(dates) * 0.8)
split_date = dates[split_index]

train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("TRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))
print("TRAINING ENDS:", df.loc[train_mask, "Date"].max())
print("TESTING STARTS:", df.loc[test_mask, "Date"].min())


# 4. Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 5. Define regression models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
trained_models = {}

# 6. Train and evaluate models
for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Demand cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))


# 7. Baseline comparison
baseline_predictions = np.maximum(
    X_test["Rolling_Mean_7"].to_numpy(),
    0
)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = root_mean_squared_error(
    y_test,
    baseline_predictions
)

results.append({
    "Model": "7-Day Moving Average Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse
})


# 8. Compare all results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)


# 9. Save the best regression model
regression_results = results_df[
    results_df["Model"] != "7-Day Moving Average Baseline"
]

best_model_name = regression_results.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "data/processed/best_demand_model.pkl"
)

print("\nBEST REGRESSION MODEL:", best_model_name)


# 10. Save test predictions
best_predictions = np.maximum(
    best_model.predict(X_test),
    0
)

prediction_results = df.loc[test_mask, [
    "Date",
    "Product",
    "Variant / Pack Size"
]].copy()

prediction_results["Actual Demand"] = y_test.to_numpy()
prediction_results["Predicted Demand"] = best_predictions

prediction_results.to_csv(
    "data/processed/model_predictions.csv",
    index=False
)

print("\nModel comparison saved.")
print("Best model saved.")
print("Test predictions saved.")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
