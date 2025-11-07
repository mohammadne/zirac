import numpy as np
import pandas as pd


def samples():
    np.random.seed(42)  # reproducibility

    # Generate class 0 (around x1, x2 ≈ 0–2)
    x0 = np.random.normal(loc=[1.0, 1.0], scale=0.4, size=(50, 2))
    y0 = np.zeros((50, 1))

    # Generate class 1 (around x1, x2 ≈ 2–3)
    x1 = np.random.normal(loc=[2.5, 2.0], scale=0.4, size=(50, 2))
    y1 = np.ones((50, 1))

    # Combine
    data = np.vstack((np.hstack((x0, y0)), np.hstack((x1, y1))))

    # Shuffle rows
    np.random.shuffle(data)

    # Save as DataFrame
    df = pd.DataFrame(data, columns=["x1", "x2", "y"])
    df.to_csv("100_classification_samples.csv", index=False)
