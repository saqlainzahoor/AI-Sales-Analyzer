import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("sales_data.csv")

print(data)

data["Total_Sales"] = data["Quantity"] * data["Price"]
print("Total Revenue:", data["Total_Sales"].sum())

print(data)
best_product = data.loc[data["Total_Sales"].idxmax(), "Product"]
print("Best Selling Product:", best_product)
total_revenue = data["Total_Sales"].sum()
best_product = data.loc[data["Total_Sales"].idxmax(), "Product"]

print("\n--- Sales Summary ---")
print("Total Revenue:", total_revenue)
print("Best Selling Product:", best_product)
print("Number of Products:", len(data))

plt.bar(data["Product"], data["Total_Sales"])
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Sales by Product")
plt.xticks(rotation=45)
plt.show()