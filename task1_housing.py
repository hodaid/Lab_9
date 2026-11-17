import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

model = LinearRegression()
model.fit(X, y)

print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)

prediction_input = pd.DataFrame({"distance_100m": [5.0]})
predicted_price = model.predict(prediction_input)
print(f"Predicted price at 500m: {predicted_price[0]:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", alpha=0.5, label="Actual Transactions")
plt.plot(
    X, model.predict(X), color="red", linewidth=2, label="Fitted Linear Model"
)
plt.title("Task 1: Real Estate Valuation - Distance vs Price")
plt.xlabel("Distance to Nearest MRT Station (per 100m)")
plt.ylabel("House Price of Unit Area")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("task1_housing.png")
plt.close()