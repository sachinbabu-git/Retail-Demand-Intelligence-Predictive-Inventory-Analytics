
import sqlite3
import pandas as pd

# Connect to SQLite database
connection = sqlite3.connect("data/retail_sales.db")

# 1. Product-wise revenue analysis
product_analysis = pd.read_sql_query("""
    SELECT
        Product,
        SUM("Quantity Sold") AS Total_Quantity,
        SUM("Revenue (₹)") AS Total_Revenue
    FROM sales
    GROUP BY Product
    ORDER BY Total_Revenue DESC;
""", connection)

print("\nPRODUCT-WISE SALES ANALYSIS")
print(product_analysis.head(10).to_string(index=False))


# 2. Monthly revenue analysis
monthly_analysis = pd.read_sql_query("""
    SELECT
        strftime('%Y-%m', Date) AS Month,
        SUM("Revenue (₹)") AS Monthly_Revenue
    FROM sales
    GROUP BY Month
    ORDER BY Month;
""", connection)

print("\nMONTHLY REVENUE ANALYSIS")
print(monthly_analysis.to_string(index=False))


# 3. Overall sales summary
sales_summary = pd.read_sql_query("""
    SELECT
        COUNT(*) AS Total_Transactions,
        SUM("Quantity Sold") AS Total_Units_Sold,
        SUM("Revenue (₹)") AS Total_Revenue,
        AVG("Revenue (₹)") AS Average_Transaction_Value
    FROM sales;
""", connection)

print("\nOVERALL SALES SUMMARY")
print(sales_summary.to_string(index=False))


# Export analysis results
product_analysis.to_csv(
    "data/processed/sql_product_analysis.csv", index=False
)

monthly_analysis.to_csv(
    "data/processed/sql_monthly_analysis.csv", index=False
)

sales_summary.to_csv(
    "data/processed/sql_sales_summary.csv", index=False
)

connection.close()

print("\nSQL AND PANDAS INTEGRATION COMPLETED!")
