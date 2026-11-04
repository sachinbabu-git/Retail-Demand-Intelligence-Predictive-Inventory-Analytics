import pandas as pd

# Load original data
df = pd.read_excel(
    "data/raw/sales_data.xlsx",
    sheet_name="POS Transactions"
)

print("Original records:", len(df))

# Remove zero-quantity transactions
df_cleaned = df[df["Quantity Sold"] > 0].copy()

# Verify revenue again
df_cleaned["Expected Revenue"] = (
    df_cleaned["Unit Price (₹)"] *
    df_cleaned["Quantity Sold"]
)

# Check revenue consistency
assert (
    df_cleaned["Revenue (₹)"] ==
    df_cleaned["Expected Revenue"]
).all()

# Remove temporary validation column
df_cleaned = df_cleaned.drop(
    columns=["Expected Revenue"]
)

# Save cleaned data
df_cleaned.to_csv(
    "data/processed/cleaned_sales.csv",
    index=False
)

print("Cleaned records:", len(df_cleaned))
print("Removed records:", len(df) - len(df_cleaned))
print("Data cleaning completed!")