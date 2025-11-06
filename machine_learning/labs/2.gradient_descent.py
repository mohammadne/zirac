import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Coefficients = tuple[NDArray[np.float64], float]


def line_equation(point1: Coefficients, point2: Coefficients) -> Coefficients:
    diff_x = point2[0] - point1[0]
    diff_y = point2[1] - point1[1]

    # Compute w (slope vector) using least-squares projection
    w = (diff_y / np.dot(diff_x, diff_x)) * diff_x

    # Compute b (intercept): y = w·x + b  →  b = y1 - w·x1
    b = point1[1] - np.dot(w, point1[0])

    return w, b


def compute_gradient(
    x: NDArray[np.float64],  # shape (m, n)
    y: NDArray[np.float64],  # shape (m,)
    w: NDArray[np.float64],  # shape (n,)
    b: float
) -> tuple[NDArray[np.float64], float, float]:
    m: int = x.shape[0]  # len(y)

    predictions = np.dot(x, w) + b      # shape (m,)
    error = predictions - y       # shape (m,)

    gradient_w = np.dot(x.T, error) / m  # shape (n,)
    gradient_b = np.sum(error) / m  # scalar
    cost = np.sum(error ** 2) / (2 * m)

    return gradient_w, gradient_b, cost


def gradient_descent(
    x: NDArray,
    y: NDArray[np.float64],
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

    # choose an appropriate initial w and b by choosing 2 random examples
    random_point_indexes = np.random.choice(x.shape[0], size=2, replace=False)
    random_point_equation = line_equation(
        (x_normalized[random_point_indexes[0]],
         float(y[random_point_indexes[0]])),
        (x_normalized[random_point_indexes[1]],
         float(y[random_point_indexes[1]]))
    )
    w_normalized, b_normalized = random_point_equation[0], random_point_equation[1]

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
        except:
            print(i, "exception occured")
            break

        iterations += 1

    w = w_normalized / x_std
    b = b_normalized - np.sum((w_normalized * x_mean) / x_std)

    return w, b, iterations, history


if __name__ == "__main__":
    # x1: Size (sqft)
    # x2: Number of Bedrooms
    # x3: Number of floors
    # x4: Age of  Home
    # y: Price (1000s dollars)

    x_train = np.array([
        [2104, 5, 1, 45],
        [1416, 3, 2, 40],
        [852, 2, 1, 35],
        [952, 2, 1, 65],
        [1244, 3, 2, 64],
        [1947, 3, 2, 17]], dtype=np.float64)
    y_train = np.array([460, 232, 178, 271.5, 232, 509.8], dtype=np.float64)
    learning_rate = 5.0e-2
    threshold = 1e-5

    w, b, iterations, history = gradient_descent(
        x_train, y_train, learning_rate, threshold, max_iterations=1000000)
    print(
        f"w: {w}\nb: {b}\niterations: {iterations}\nlast_cost: {history[-1][-1]}")

    for i in range(len(y_train)):
        prediction = np.dot(x_train[i], w)+b
        print(f"actual: {y_train[i]}, predicted: {prediction}")

    cost_values = [item[-1] for item in history]
    iterations = range(len(history))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, cost_values, color='b', linewidth=1.5, label='Cost')
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
