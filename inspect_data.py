import pandas as pd

file_path = "data/raw/sales_data.xlsx"

df = pd.read_excel(file_path, sheet_name="POS Transactions")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns)

print("\nDATASET INFORMATION:")
df.info()

print("\nSTATISTICAL SUMMARY:")
print(df.describe())