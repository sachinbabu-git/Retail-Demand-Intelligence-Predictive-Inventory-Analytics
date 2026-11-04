
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD CLEANED SALES DATA
# ==========================================

df = pd.read_csv("data/processed/cleaned_sales.csv")

print("Dataset loaded successfully!")
print("Total transactions:", len(df))


# ==========================================
# 2. PRODUCT REVENUE ANALYSIS
# ==========================================

product_revenue = (
    df.groupby("Product")["Revenue (₹)"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCTS BY REVENUE:")
print(product_revenue.head(10))


# ==========================================
# 3. PRODUCT QUANTITY ANALYSIS
# ==========================================

product_quantity = (
    df.groupby("Product")["Quantity Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCTS BY QUANTITY:")
print(product_quantity.head(10))


# ==========================================
# 4. TOP 10 PRODUCTS REVENUE CHART
# ==========================================

top_10_revenue = product_revenue.head(10)

plt.figure(figsize=(10, 6))

top_10_revenue.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (₹)")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig(
    "data/processed/top_products_revenue.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRevenue chart saved successfully!")


# ==========================================
# 5. MONTHLY REVENUE ANALYSIS
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.to_period("M")

monthly_revenue = (
    df.groupby("Month")["Revenue (₹)"]
    .sum()
)

print("\nMONTHLY REVENUE:")
print(monthly_revenue)
# Month-over-month revenue growth

monthly_growth = monthly_revenue.pct_change() * 100

print("\nMONTH-OVER-MONTH GROWTH:")
print(monthly_growth.round(2))


# ==========================================
# 6. MONTHLY REVENUE CHART
# ==========================================

plt.figure(figsize=(10, 5))

monthly_revenue.plot(kind="bar")

plt.title("Monthly Revenue Performance")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/processed/monthly_revenue.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Monthly revenue chart saved successfully!")

# ==========================================
# 7. DAILY REVENUE ANALYSIS
# ==========================================

daily_revenue = (
    df.groupby("Date")["Revenue (₹)"]
    .sum()
)

print("\nDAILY REVENUE SUMMARY:")
print(daily_revenue.describe())

# Identify highest revenue days
print("\nTOP 10 DAYS BY REVENUE:")
print(
    daily_revenue
    .sort_values(ascending=False)
    .head(10)
)

# Create daily revenue chart
plt.figure(figsize=(12, 5))

daily_revenue.plot(kind="line")

plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue (₹)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/processed/daily_revenue.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDaily revenue chart saved successfully!")

# ==========================================
# 8. WEEKDAY VS WEEKEND ANALYSIS
# ==========================================

# Create a daily revenue DataFrame
daily_sales = daily_revenue.reset_index()

# Identify weekday or weekend
daily_sales["Day Type"] = daily_sales["Date"].dt.dayofweek.apply(
    lambda day: "Weekend" if day >= 5 else "Weekday"
)

# Calculate average daily revenue
day_type_analysis = (
    daily_sales.groupby("Day Type")["Revenue (₹)"]
    .agg(["mean", "sum", "count"])
)

print("\nWEEKDAY VS WEEKEND ANALYSIS:")
print(day_type_analysis.round(2))

# Visualize average daily revenue
plt.figure(figsize=(8, 5))

day_type_analysis["mean"].plot(kind="bar")

plt.title("Average Daily Revenue: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Revenue (₹)")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "data/processed/weekday_weekend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nWeekday vs Weekend chart saved successfully!")

print("\nEDA completed successfully!")
