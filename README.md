# Multiple Linear Regression from Scratch (The Normal Equation)

This repository contains a custom implementation of Multiple Linear Regression using **Matrix Algebra**. The project demonstrates how to solve for regression coefficients using the closed-form **Normal Equation** and compares the results against Scikit-Learn's standard implementation.

## 📄 Project Overview

The goal of this project is to demystify the mathematics behind Multiple Linear Regression. Instead of relying on a black-box library, we build a custom class `MyLR` that calculates the optimal coefficients using numpy matrix operations.

**Key features:**
* Implementation of the **Ordinary Least Squares (OLS)** solution using the Normal Equation.
* Handling of the intercept term by modifying the design matrix.
* Verification of results by comparing custom coefficients with Scikit-Learn's `LinearRegression`.

## 📊 Dataset

The project uses the standard **Diabetes Dataset** provided by Scikit-Learn:
* **Samples:** 442
* **Features:** 10 (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
* **Target:** Quantitative measure of disease progression one year after baseline.

## 🧮 Mathematical Approach

To find the coefficients ($\beta$) that minimize the error for multiple features, we use the **Normal Equation**:

$$\beta = (X^T X)^{-1} X^T y$$

Where:
* $\beta$ is the vector of coefficients (including the intercept).
* $X$ is the input feature matrix (Design Matrix).
* $X^T$ is the transpose of $X$.
* $y$ is the vector of target values.
* $(X^T X)^{-1}$ is the inverse of the dot product of the transposed matrix and original matrix.

## 🛠️ Implementation Details

The core logic is implemented in the `MyLR` class. Note how we insert a column of `1`s into the matrix to handle the intercept (bias) term mathematically:

```python
class MyLR:
    def fit(self, X_train, y_train):
        # Insert a column of 1s at the beginning of X for the intercept
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Calculate coefficients using the Normal Equation
        # beta = (X^T * X)^-1 * X^T * y
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        
        # Separate intercept (beta_0) and coefficients (beta_1...n)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
