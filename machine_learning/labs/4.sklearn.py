import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = np.loadtxt("./data/100_classification_samples.csv",
                  delimiter=",", skiprows=1)

validation_count = 5
x_train = data[:-validation_count, :2]
y_train = data[:-validation_count, 2]
x_validation = data[-validation_count:, :2]
y_validation = data[-validation_count:, 2]


start = time.time()

lr_model = LogisticRegression(solver='saga', max_iter=1_000_000, tol=1e-5,)
lr_model.fit(x_train, y_train)

end = time.time()
print(f"Training time: {end - start:.6f} seconds")

y_pred = lr_model.predict(x_validation)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(x_train, y_train))
