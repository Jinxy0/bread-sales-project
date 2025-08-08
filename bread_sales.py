import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# --- Step 1: Create synthetic bread sales dataset ---
num_rows = 1000  # Minimum rows

# Date range for sales
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_rows)]

# Bread types
bread_types = ["Sourdough", "Baguette", "Whole Wheat", "Rye", "Multigrain"]

# Store locations
stores = ["Northside", "East End", "Downtown", "Westside", "Uptown"]

# Generate data
data = {
    "Date": dates,
    "BreadType": [random.choice(bread_types) for _ in range(num_rows)],
    "Store": [random.choice(stores) for _ in range(num_rows)],
    "QuantitySold": np.random.randint(5, 50, num_rows),
    "UnitPrice": np.random.uniform(1.5, 5.0, num_rows).round(2),
}

# Total sales
data["TotalSale"] = (data["QuantitySold"] * data["UnitPrice"]).round(2)

# Discounts (0% to 20%)
data["Discount"] = np.random.choice([0, 0.05, 0.1, 0.15, 0.2], num_rows)

# Final price after discount
data["FinalSale"] = (data["TotalSale"] * (1 - data["Discount"])).round(2)

# Random customer satisfaction score (1 to 5)
data["CustomerRating"] = np.random.randint(1, 6, num_rows)

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("bread_sales_dataset.csv", index=False)
print("Dataset saved as 'bread_sales_dataset.csv'")

# --- Step 2: Create charts ---
plt.figure(figsize=(16, 12))

# 1. Sales trend over time
plt.subplot(3, 3, 1)
df.groupby("Date")["FinalSale"].sum().plot()
plt.title("Total Sales Over Time")
plt.ylabel("Sales ($)")

# 2. Average sale by bread type
plt.subplot(3, 3, 2)
df.groupby("BreadType")["FinalSale"].mean().plot(kind="bar")
plt.title("Average Sales by Bread Type")
plt.ylabel("Avg Sale ($)")

# 3. Total sales by store
plt.subplot(3, 3, 3)
df.groupby("Store")["FinalSale"].sum().plot(kind="bar", color="orange")
plt.title("Total Sales by Store")
plt.ylabel("Sales ($)")

# 4. Quantity sold distribution
plt.subplot(3, 3, 4)
df["QuantitySold"].plot(kind="hist", bins=20, color="green", alpha=0.7)
plt.title("Quantity Sold Distribution")
plt.xlabel("Quantity")

# 5. Price distribution
plt.subplot(3, 3, 5)
df["UnitPrice"].plot(kind="hist", bins=20, color="purple", alpha=0.7)
plt.title("Unit Price Distribution")
plt.xlabel("Price ($)")

# 6. Average rating per bread type
plt.subplot(3, 3, 6)
df.groupby("BreadType")["CustomerRating"].mean().plot(kind="bar", color="red")
plt.title("Average Customer Rating by Bread Type")
plt.ylabel("Rating (1-5)")

# 7. Discount frequency
plt.subplot(3, 3, 7)
df["Discount"].value_counts().sort_index().plot(kind="bar", color="brown")
plt.title("Discount Frequency")
plt.xlabel("Discount Rate")
plt.ylabel("Count")

# 8. Correlation between price and quantity
plt.subplot(3, 3, 8)
plt.scatter(df["UnitPrice"], df["QuantitySold"], alpha=0.5, color="blue")
plt.title("Unit Price vs Quantity Sold")
plt.xlabel("Unit Price ($)")
plt.ylabel("Quantity Sold")

# 9. Total monthly sales
plt.subplot(3, 3, 9)
df.groupby(df["Date"].dt.to_period("M"))["FinalSale"].sum().plot()
plt.title("Total Monthly Sales")
plt.ylabel("Sales ($)")

plt.tight_layout()
plt.show()
