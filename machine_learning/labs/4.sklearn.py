import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# generate x
x = np.random.uniform(10, 100, 1000).reshape(-1, 1)

# generate y ~ log(x) with noise
y = (2.5 * np.log(x) + 0.3 * np.random.randn(*x.shape)).reshape(-1, 1)


indices = np.arange(len(x))
np.random.shuffle(indices)
# Apply to x and y
x = x[indices]
y = y[indices]


n = len(x)
train_end = int(0.7 * n)   # 70%
cv_end = int(0.9 * n)     # 70% + 20%

# Split
x_train = x[:train_end]
y_train = y[:train_end]

x_cv = x[train_end:cv_end]
y_cv = y[train_end:cv_end]

x_test = x[cv_end:]
y_test = y[cv_end:]


print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

plt.scatter(x_train, y_train, marker='x', c='r', label='Train')
plt.scatter(x_cv, y_cv, marker='x', c='b', label='Validation')
plt.scatter(x_test, y_test, marker='x', c='g', label='Test')
plt.legend()
plt.show()

# Initialize the class
scaler_linear = StandardScaler()
# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(x_train)

plt.scatter(X_train_scaled, y_train, marker='x', c='r', label='Train Scaled')
plt.legend()
plt.show()

print(f"""
Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}
""")

# Initialize the class
linear_model = LinearRegression()
# linear_model = LogisticRegression(solver='saga', max_iter=1_000_000, tol=1e-5,)

start = time.time()
# Train the model
linear_model.fit(X_train_scaled, y_train)
end = time.time()
print(f"Training time: {end - start:.6f} seconds")

# Feed the scaled training set and get the predictions
yhat_train = linear_model.predict(X_train_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"training MSE: {mean_squared_error(y_train, yhat_train) / 2}")
print(f"training Accuracy: {linear_model.score(x_train, y_train)}")

# use the mean and standard deviation of the training set
# only calling transform() method instead of fit_transform()

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled = scaler_linear.transform(x_cv)

# Feed the scaled cross validation set
yhat_cv = linear_model.predict(X_cv_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat_cv) / 2}")

# -------------------------------------> Create additional features

# Initialize lists to save the errors, models, and feature transforms
degrees = range(1, 11)
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

# Each adding one more degree of polynomial higher than the last.
for degree in degrees:
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)

    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)

    # Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)


plt.plot(degrees, train_mses, label="Train")
plt.plot(degrees, cv_mses, label="CV")

plt.xlabel("Degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")

# Add polynomial features to the test set
X_test_mapped = polys[degree-1].transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"""
Training MSE: {train_mses[degree-1]:.2f}
Cross Validation MSE: {cv_mses[degree-1]:.2f}
Test MSE: {test_mse:.2f}
""")
