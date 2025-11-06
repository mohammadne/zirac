import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def compute_cost(x: NDArray[np.float64], y: NDArray[np.float64], w: float, b: float) -> float:
    result: float = 0.0
    m: int = x.shape[0]
    for i in range(m):
        result += (w * x[i] + b - y[i]) ** 2
    return result / (2 * m)


def compute_gradient(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w: float,
    b: float,
) -> tuple[float, float]:
    m: int = x.shape[0]

    # gradient_w = 0
    # for i in range(m):
    #     gradient_w += x[i] * ((w*x[i] + b) - y[i])
    # gradient_w /= m

    # gradient_b = 0
    # for i in range(m):
    #     gradient_b += (w*x[i] + b) - y[i]
    # gradient_b /= m

    # using vectorization
    error = (w * x + b) - y
    gradient_w = np.dot(error, x) / m
    gradient_b = np.sum(error) / m

    return gradient_w, gradient_b


def gradient_descent(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rate: float,
    threshold: float,
    max_iterations: int = 10000
):
    iterations = 0
    history = []
    w, b = 0.0, 0.0
    prev_cost = float('inf')

    for _ in range(max_iterations):
        gradient_w, gradient_b = compute_gradient(x, y, w, b)
        w = w - learning_rate*gradient_w
        b = b - learning_rate*gradient_b

        # call compute_cost less frequently (for performacne)
        if iterations % 5 == 0:
            current_cost = compute_cost(x, y, w, b)
            history.append([w, b, current_cost])

            if abs(prev_cost - current_cost) < threshold:
                break

            prev_cost = current_cost

        iterations += 1

    return w, b, iterations, history


if __name__ == "__main__":
    x_train = np.array([1, 2, 3])
    y_train = np.array([3, 5, 7])
    learning_rate = 0.01
    threshold = 0.00001

    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Actual Points")
    plt.ylabel('Price (in 1000s of dollars)')
    plt.xlabel('Size (1000 sqft)')
    plt.show()

    w, b, iterations, history = gradient_descent(
        x_train, y_train, learning_rate, threshold)
    print(f"w: {w}\nb: {b}\niterations: {iterations}")

    points = np.array(history)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color='blue', s=50)  # Scatter plot # type: ignore
    # ax.scatter(x[0], y[0], z[0], color='red', s=80, label='Start')
    # ax.scatter(x[-1], y[-1], z[-1], color='green', s=80, label='End')
    ax.plot(x, y, z, color='gray', alpha=0.6)  # connect points with lines
    ax.legend()

    # Axis labels
    ax.set_title("")
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('J')

    plt.show()
