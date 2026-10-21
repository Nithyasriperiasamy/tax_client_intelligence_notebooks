"""
ridge_regression_manual.py
----------------------------
INSIGHT 3: Predict Client Acquisition Cost Payback Period
Algorithm: Ridge Regression (Linear Regression + L2 penalty),
implemented FROM SCRATCH using batch gradient descent (no scikit-learn).

Business Question: "How many months until this client becomes
profitable, given what it cost to acquire them (ads, referral bonus,
onboarding time)?"

Why Ridge here specifically: acquisition_channel_cost, onboarding_hours,
expected_annual_fee, and expected_retention_years are all strongly
correlated with each other (costlier acquisition channels tend to bring
in clients who need more onboarding, who tend to be higher-fee clients,
who tend to stay longer). Plain linear regression can become unstable /
overfit on this kind of multicollinearity; Ridge's L2 penalty shrinks
and stabilizes the coefficients, giving a more reliable prediction.

Business use: know which acquisition channels/clients pay back fastest,
so marketing and sales can prioritize spend on channels that convert
into profitable clients quickly.
"""

import numpy as np
import pandas as pd
from utils import train_test_split, standardize, print_regression_report


class RidgeRegressionScratch:
    """Linear regression with an L2 penalty on the weights (not the bias)."""

    def __init__(self, lr=0.05, n_iters=3000, alpha=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha  # regularization strength (lambda)
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # MSE gradient + L2 penalty gradient (2*alpha*w), bias not penalized
            dw = (2 / n_samples) * (X.T @ error) + (2 * self.alpha / n_samples) * self.weights
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 200 == 0 or i == self.n_iters - 1:
                mse = np.mean(error ** 2)
                penalty = self.alpha * np.sum(self.weights ** 2)
                self.loss_history.append(mse + penalty)

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


def plain_linear_for_comparison(X_train, y_train, X_test, lr=0.05, n_iters=3000):
    """A zero-regularization run (alpha=0) of the same class, used purely
    to show Ridge vs plain Linear Regression side by side on this
    correlated feature set."""
    model = RidgeRegressionScratch(lr=lr, n_iters=n_iters, alpha=0.0)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def main():
    df = pd.read_csv("data/client_data.csv")

    # Deliberately correlated feature set (this is what makes Ridge worth it)
    feature_cols = [
        "acquisition_channel_cost",
        "onboarding_hours",
        "expected_annual_fee",
        "expected_retention_years",
    ]
    target_col = "payback_period_months"

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_s, mean, std = standardize(X_train)
    X_test_s, _, _ = standardize(X_test, mean, std)

    print("=" * 60)
    print("INSIGHT 3: Client Acquisition Payback Period (Ridge Regression)")
    print("=" * 60)
    print(f"Features used (intentionally correlated): {feature_cols}")
    print(f"Feature correlation matrix:\n{pd.DataFrame(X_train, columns=feature_cols).corr().round(2)}")

    # --- Plain linear regression baseline (alpha = 0) ---
    y_pred_plain, plain_model = plain_linear_for_comparison(X_train_s, y_train, X_test_s)
    print_regression_report(y_test, y_pred_plain, title="Plain Linear Regression (alpha=0) - TEST")

    # --- Ridge regression ---
    ridge = RidgeRegressionScratch(lr=0.1, n_iters=4000, alpha=1.0)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = ridge.predict(X_test_s)
    print_regression_report(y_test, y_pred_ridge, title="Ridge Regression (alpha=1.0) - TEST")

    print("\nCoefficient comparison (Plain Linear vs Ridge) - shows Ridge stabilizing weights:")
    for name, w_plain, w_ridge in zip(feature_cols, plain_model.weights, ridge.weights):
        print(f"  {name:30s}  plain={w_plain:8.3f}   ridge={w_ridge:8.3f}")

    # --- Business insight: fastest vs slowest paying-back clients ---
    X_all_s, _, _ = standardize(X, mean, std)
    df["predicted_payback_months"] = ridge.predict(X_all_s)

    fastest = df.sort_values("predicted_payback_months", ascending=True).head(10)
    print("\nTop 10 fastest-payback clients (best acquisition ROI):")
    print(fastest[["client_id", "acquisition_channel_cost", "onboarding_hours",
                    "expected_annual_fee", "expected_retention_years",
                    "predicted_payback_months"]].round(2).to_string(index=False))

    slowest = df.sort_values("predicted_payback_months", ascending=False).head(5)
    print("\nTop 5 slowest-payback clients (worst acquisition ROI, review channel/pricing):")
    print(slowest[["client_id", "acquisition_channel_cost", "onboarding_hours",
                    "expected_annual_fee", "expected_retention_years",
                    "predicted_payback_months"]].round(2).to_string(index=False))

    avg_payback = df["predicted_payback_months"].mean()
    print(f"\nAverage predicted payback period across all clients: {avg_payback:.2f} months")


if __name__ == "__main__":
    main()
