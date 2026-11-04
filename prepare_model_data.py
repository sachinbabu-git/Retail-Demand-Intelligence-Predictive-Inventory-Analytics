
import pandas as pd

# Load cleaned sales data
df = pd.read_csv("data/processed/cleaned_sales.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Aggregate daily demand by product variant
daily_sales = (
    df.groupby(["Product", "Variant / Pack Size", "Date"])["Quantity Sold"]
    .sum()
    .reset_index()
)

end_date = df["Date"].max()
prepared_groups = []

# Create a separate timeline for each product variant
for (product, variant), group in daily_sales.groupby(
    ["Product", "Variant / Pack Size"]
):
    group = group.set_index("Date").sort_index()

    date_range = pd.date_range(
        start=group.index.min(),
        end=end_date,
        freq="D"
    )

    group = group.reindex(date_range)

    group["Quantity Sold"] = (
        pd.to_numeric(group["Quantity Sold"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    group.index.name = "Date"
    group["Product"] = product
    group["Variant / Pack Size"] = variant

    prepared_groups.append(group.reset_index())

# Combine all product timelines
daily_sales = pd.concat(prepared_groups, ignore_index=True)

# Date-based features
daily_sales["Day"] = daily_sales["Date"].dt.day
daily_sales["Month"] = daily_sales["Date"].dt.month
daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek
daily_sales["IsWeekend"] = (
    daily_sales["DayOfWeek"] >= 5
).astype(int)

# Historical demand features
daily_sales = daily_sales.sort_values(
    ["Product", "Variant / Pack Size", "Date"]
)

grouped = daily_sales.groupby(
    ["Product", "Variant / Pack Size"]
)["Quantity Sold"]

daily_sales["Demand_Lag_1"] = grouped.shift(1)
daily_sales["Demand_Lag_7"] = grouped.shift(7)

daily_sales["Rolling_Mean_7"] = (
    daily_sales.groupby(
        ["Product", "Variant / Pack Size"]
    )["Quantity Sold"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

# Remove rows without sufficient historical data
model_data = daily_sales.dropna().copy()

# Save forecasting dataset
model_data.to_csv(
    "data/processed/model_data.csv",
    index=False
)

print("\nFORECASTING DATASET PREPARED")
print("Total rows:", len(model_data))
print("Products:", model_data["Product"].nunique())
print(
    "Product variants:",
    model_data[["Product", "Variant / Pack Size"]]
    .drop_duplicates().shape[0]
)
print(
    "Date range:",
    model_data["Date"].min(),
    "to",
    model_data["Date"].max()
)
print("Missing values:", model_data.isnull().sum().sum())
print("\nSample records:")
print(model_data.head().to_string(index=False))
print("\nSaved to data/processed/model_data.csv")
