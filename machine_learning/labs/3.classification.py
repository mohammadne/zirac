import math
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def compute_gradient(
    x: NDArray[np.float64],  # shape (m, n)
    y: NDArray,  # shape (m,)
    w: NDArray[np.float64],  # shape (n,)
    b: float
) -> tuple[NDArray[np.float64], float, float]:
    m: int = x.shape[0]  # len(y)

    z = np.dot(x, w) + b  # shape (m,)
    prediction = 1 / (1 + np.exp(-z))  # shape (m,)
    error = prediction - y  # shape (m,)

    gradient_w = np.dot(x.T, error) / m  # shape (n,)
    gradient_b = np.sum(error) / m  # scalar
    cost = np.sum(-1*y*np.log(prediction)-(1-y) *
                  np.log(1-prediction)) / m  # scalar

    return gradient_w, gradient_b, cost


def gradient_descent(
    x: NDArray,
    y: NDArray,
    learning_rate: float,
    threshold: float,
    max_iterations: int = 10000
):
    iterations = 0
    history = []
    prev_cost = float('inf')

    x_mean = np.mean(x, axis=0)  # shape (n,)
    x_std = np.std(x, axis=0)  # shape (n,)
    x_normalized = (x - x_mean) / x_std

    w_normalized, b_normalized = np.zeros(x.shape[1]), 0

    for i in range(max_iterations):
        try:
            gradient_w, gradient_b, current_cost = compute_gradient(
                x_normalized, y, w_normalized, b_normalized)
            w_normalized = w_normalized - learning_rate*gradient_w
            b_normalized = b_normalized - learning_rate*gradient_b

            history.append([w_normalized, b_normalized, current_cost])

            if abs(prev_cost - current_cost) < threshold:
                break
            prev_cost = current_cost
        except Exception as e:
            print(i, "exception occured", e)
            break

        iterations += 1

    w = w_normalized / x_std
    b = b_normalized - np.sum((w_normalized * x_mean) / x_std)

    return w, b, iterations, history


if __name__ == "__main__":
    data = np.loadtxt("./data/100_classification_samples.csv",
                      delimiter=",", skiprows=1)

    # Split features (x) and labels (y)
    x_train = data[:, :-1]   # all rows, first 2 columns
    y_train = data[:, -1]    # all rows, 3rd column
    learning_rate = 5.0e-2
    threshold = 1e-5

    start = time.time()
    w, b, iterations, history = gradient_descent(
        x_train, y_train, learning_rate, threshold, max_iterations=1000000)
    end = time.time()
    print(f"Training time: {end - start:.6f} seconds")
    print(
        f"w: {w}\nb: {b}\niterations: {iterations}\nlast_cost: {history[-1][-1]}")

    for i in range(len(y_train)):
        prediction = np.dot(x_train[i], w)+b > 0
        print(f"actual: {y_train[i]}, predicted: {prediction}")

    def plot_classification(ax, x, y, w, b):
        # Scatter both classes in one loop
        for cls, color in [(0, 'red'), (1, 'blue')]:
            ax.scatter(*x[y == cls].T, color=color,
                       label=f'Class {cls}', alpha=0.7)

        # Decision boundary
        x_vals = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)
        ax.plot(x_vals, -(b + w[0]*x_vals) / w[1], 'k--', label='Boundary')

        ax.set(title='Sample Classification',
               xlabel='Feature 1', ylabel='Feature 2')
        ax.legend()

    def plot_learning_curve(ax, history):
        ax.plot(range(len(history)), [h[-1]
                for h in history], 'b', lw=1.5, label='Cost')
        ax.set(title='Learning Curve', xlabel='Iterations', ylabel='Cost')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_classification(axes[0], x_train, y_train, w, b)
    plot_learning_curve(axes[1], history)
    plt.tight_layout()
    plt.show()
