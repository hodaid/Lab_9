import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=[target])
y = concrete[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_baseline = LinearRegression()
baseline_scores = -cross_val_score(
    linear_baseline, X_train, y_train, cv=folds, scoring=scoring
)
print(f"Linear Baseline Mean CV RMSE: {baseline_scores.mean():.2f} MPa")

ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])
param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}
search = GridSearchCV(
    ridge_pipeline, param_grid, cv=folds, scoring=scoring, refit=True
)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
results["mean_rmse"] = -results["mean_test_score"]
top_results = results.sort_values(by="rank_test_score").head(3)

print("\nTop 3 Grid Search Settings:")
for idx, row in top_results.iterrows():
    print(
        f"Rank {int(row['rank_test_score'])}: Params = {row['params']} | Mean CV RMSE = {row['mean_rmse']:.2f} MPa"
    )

print(f"\nBest Parameters Found: {search.best_params_}")
print(f"Best Mean CV RMSE: {-search.best_score_:.2f} MPa")

test_predictions = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"Test RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

plt.figure(figsize=(7, 7))
plt.scatter(
    y_test, test_predictions, color="purple", alpha=0.6, label="Predictions"
)
min_val = min(y_test.min(), test_predictions.min())
max_val = max(y_test.max(), test_predictions.max())
plt.plot(
    [min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal y=x"
)
plt.title("Task 3: Actual vs Predicted Concrete Strength")
plt.xlabel("Actual Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("task3_concrete.png")
plt.close()