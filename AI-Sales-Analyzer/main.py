import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# 1. Load Sales Data
# =========================

data = pd.read_csv("sales_data.csv")


# =========================
# 2. Calculate Total Sales
# =========================

data["Total_Sales"] = data["Quantity"] * data["Price"]


# =========================
# 3. Sales Summary
# =========================

total_revenue = data["Total_Sales"].sum()

best_product = data.loc[
    data["Total_Sales"].idxmax(), "Product"
]

number_of_products = len(data)

print("\n==============================")
print("       SALES SUMMARY")
print("==============================")

print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Best Selling Product: {best_product}")
print(f"Number of Products: {number_of_products}")


# =========================
# 4. Sales Visualization
# =========================

plt.figure(figsize=(10, 6))

plt.bar(
    data["Product"],
    data["Total_Sales"]
)

plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Sales by Product")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# =========================
# 5. Machine Learning Model
# =========================

X = data[["Quantity", "Price"]]
y = data["Total_Sales"]

model = LinearRegression()

model.fit(X, y)


# =========================
# 6. AI Sales Prediction
# =========================

print("\n==============================")
print("       AI SALES PREDICTION")
print("==============================")

quantity = float(input("Enter Quantity: "))
price = float(input("Enter Price: "))

new_data = pd.DataFrame({
    "Quantity": [quantity],
    "Price": [price]
})

prediction = model.predict(new_data)

print(f"\nPredicted Sales: {prediction[0]:,.2f}")


# =========================
# 7. Model Evaluation
# =========================

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n==============================")
print("       MODEL EVALUATION")
print("==============================")

print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

