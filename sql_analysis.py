
import sqlite3

# Connect to database
connection = sqlite3.connect("data/retail_sales.db")

# Create cursor
cursor = connection.cursor()

# QUERY 1: Display all columns, first 5 records
print("\nFIRST 5 RECORDS:")

cursor.execute("""
SELECT *
FROM sales
LIMIT 5;
""")

for row in cursor.fetchall():
    print(row)


# QUERY 2: Display only product and revenue
print("\nPRODUCT AND REVENUE:")

cursor.execute("""
SELECT Product, "Revenue (₹)"
FROM sales
LIMIT 10;
""")

for row in cursor.fetchall():
    print(row)


# QUERY 3: Count total transactions
print("\nTOTAL TRANSACTIONS:")

cursor.execute("""
SELECT COUNT(*)
FROM sales;
""")

print(cursor.fetchone()[0])


# Close connection
connection.close()
