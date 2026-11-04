
import pandas as pd
import sqlite3

# STEP 1: Load cleaned sales data
df = pd.read_csv("data/processed/cleaned_sales.csv")

print("CSV loaded:", len(df), "records")

# STEP 2: Connect to SQLite database
connection = sqlite3.connect("data/retail_sales.db")

# STEP 3: Transfer data into SQL table
df.to_sql(
    "sales",
    connection,
    if_exists="replace",
    index=False
)

print("Data successfully transferred to SQL!")

# STEP 4: Verify database records
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM sales")

record_count = cursor.fetchone()[0]

print("Records inserted:", record_count)

# STEP 5: Retrieve first five records
cursor.execute("SELECT * FROM sales LIMIT 5")

print("\nFIRST FIVE RECORDS:")

for record in cursor.fetchall():
    print(record)

# STEP 6: Close database connection
connection.close()

print("\nDatabase created successfully!")
