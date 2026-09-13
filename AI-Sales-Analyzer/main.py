import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# 1. Load Sales Data
# =========================

data = pd.read_csv("sales_data.csv")


# =========================
# 2. Sales Summary
# =========================

total_revenue = data["Total_Sales"].sum()

best_product = (
    data.groupby("Product")["Total_Sales"]
    .sum()
    .idxmax()
)

number_of_products = data["Product"].nunique()

print("\n==============================")
print("       SALES SUMMARY")
print("==============================")

print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Best Selling Product: {best_product}")
print(f"Number of Products: {number_of_products}")


# =========================
# 3. Sales Visualization
# =========================

product_sales = (
    data.groupby("Product")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

plt.bar(
    product_sales.index,
    product_sales.values
)

plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Total Sales by Product")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# =========================
# 4. Prepare ML Data
# =========================

features = [
    "Product",
    "Quantity",
    "Price",
    "Discount",
    "Region",
    "Advertising_Spend",
    "Customer_Type"
]

X = data[features]
y = data["Total_Sales"]


# =========================
# 5. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 6. Preprocessing
# =========================

categorical_features = [
    "Product",
    "Region",
    "Customer_Type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================
# 7. Machine Learning Model
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

model.fit(X_train, y_train)


# =========================
# 8. AI Sales Prediction
# =========================

print("\n==============================")
print("       AI SALES PREDICTION")
print("==============================")

product = input("Enter Product: ")
quantity = float(input("Enter Quantity: "))
price = float(input("Enter Price: "))
discount = float(input("Enter Discount (%): "))
region = input("Enter Region: ")
advertising_spend = float(input("Enter Advertising Spend: "))
customer_type = input("Enter Customer Type: ")

new_data = pd.DataFrame({
    "Product": [product],
    "Quantity": [quantity],
    "Price": [price],
    "Discount": [discount],
    "Region": [region],
    "Advertising_Spend": [advertising_spend],
    "Customer_Type": [customer_type]
})

prediction = model.predict(new_data)

print(f"\nPredicted Sales: {prediction[0]:,.2f}")


# =========================
# 9. Model Evaluation
# =========================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("       MODEL EVALUATION")
print("==============================")

print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

