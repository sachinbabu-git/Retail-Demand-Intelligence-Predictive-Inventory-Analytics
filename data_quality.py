import pandas as pd

# Load sales data
df = pd.read_excel(
    "data/raw/sales_data.xlsx",
    sheet_name="POS Transactions"
)

# 1. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 2. Check duplicate rows
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# 3. Check zero quantity
print("\nZERO QUANTITY RECORDS:")
print((df["Quantity Sold"] == 0).sum())

# 4. Check zero revenue
print("\nZERO REVENUE RECORDS:")
print((df["Revenue (₹)"] == 0).sum())

# 5. Check negative values
print("\nNEGATIVE QUANTITIES:")
print((df["Quantity Sold"] < 0).sum())

print("\nNEGATIVE REVENUE:")
print((df["Revenue (₹)"] < 0).sum())
# Investigate zero-value transactions

print("\nZERO QUANTITY TRANSACTIONS:")

zero_quantity = df[df["Quantity Sold"] == 0]

print(zero_quantity.to_string(index=False))


print("\nZERO REVENUE TRANSACTIONS:")

zero_revenue = df[df["Revenue (₹)"] == 0]

print(zero_revenue.to_string(index=False))
# Validate revenue calculations

df["Expected Revenue"] = (
    df["Unit Price (₹)"] * df["Quantity Sold"]
)

df["Revenue Difference"] = (
    df["Revenue (₹)"] - df["Expected Revenue"]
)

incorrect_records = df[
    df["Revenue Difference"] != 0
]

print("\nREVENUE VALIDATION:")
print("Incorrect records:", len(incorrect_records))

if len(incorrect_records) > 0:
    print(incorrect_records.to_string(index=False))
else:
    print("All revenue calculations are correct!")