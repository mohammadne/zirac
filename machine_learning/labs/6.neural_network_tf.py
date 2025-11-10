import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def plt_mc(X, y):
    plt.figure(figsize=(6, 5))
    classes = np.unique(y)
    colors = ['r', 'g', 'b', 'y', 'pink', 'orange']

    for c in classes:
        points = X[y == c]
        plt.scatter(points[:, 0], points[:, 1],
                    color=colors[c], label=f'Class {c}')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Training Data by Class")
    plt.legend()
    plt.grid(True)
    plt.show()


def coffee_roasting():
    # CoffeeRoasting data (Temperature, Duration) -> Delicious
    X = np.array([[100, 1], [200, 1], [300, 1], [400, 1], [250, 2], [300, 3]])
    Y = np.array([[0], [1], [1], [0], [1], [0]])

    print(X.shape, Y.shape)

    print("pre normalization")
    print(f"T Max, Min: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
    print(f"D Max, Min: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print("post normalization")
    print(f"T Max, Min: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
    print(f"D Max, Min: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

    Xt = np.tile(Xn, (1000, 1))
    Yt = np.tile(Y, (1000, 1))
    print(Xt.shape, Yt.shape)

    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='relu', name='layer1'),
            Dense(1, activation='sigmoid', name='layer2')
        ]
    )

    # provides a description of the network
    model.summary()

    # (W1 parameters (input size)  + b1 parameters) * neurons
    L1_num_params = (2 + 1) * 3
    # (W2 parameters (input size)  + b2 parameters) * neurons
    L2_num_params = (3 + 1) * 1
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    model.fit(Xt, Yt, epochs=10)

    W1, b1 = model.get_layer("layer1").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    # W1(2, 3):
    #  [[ 6.4977016  6.033808   6.810511 ]
    #  [ 2.2868907 -3.389847  -0.9079014]]
    # b1(3,): [-4.430765  4.60956   6.558609]

    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
    # W2(3, 1):
    #  [[-9.243866 ]
    #  [ 5.2992144]
    #  [ 3.6569176]]
    # b2(1,): [-3.809796]

    X_test = np.array([
        [150, 1],   # positive example
        [50, 1]])   # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")


def softmax():
    # make 4-class dataset for classification
    X_train, y_train = make_blobs(
        centers=[[-5, 2], [-2, -2], [1, 2], [5, -2]],
        n_samples=2000, cluster_std=1.0, random_state=30,
    )

    model = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(4, activation='linear')  # instead of softmax activation
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(X_train, y_train, epochs=10)

    prediction = model.predict(X_train)
    print(f"two example output vectors:\n {prediction[:2]}")
    print("largest value", np.max(prediction),
          "smallest value", np.min(prediction))

    softmax_prediction = tf.nn.softmax(prediction).numpy()
    print(f"two example output vectors:\n {softmax_prediction[:2]}")
    print("largest value", np.max(softmax_prediction),
          "smallest value", np.min(softmax_prediction))

    for i in range(5):
        print(f"{prediction[i]}, category: {np.argmax(prediction[i])}")


def multi_class_classification():
    """
    This function has a beautiful intuition -)))
    the first layer has 3 neurons (units) which intutively are the 3 lines that seperates the points
    the output of first layer is a0, a1 and a2
        a0 (vertical line) -> for class 0, 2 and 4 is zero and for class 1, 3 and 5 is their z (w1.x + b1)
        a1 (horizontal line 1) -> for class 0 and 1 is zero and for class 2, 3, 4 and 5 is their y (w2.x + b2)
        a2 (horizontal line 2) -> for class 0, 1, 2 and 3 is zero and for class 4 and 5 is their y (w3.x + b3)
            it means by W and b, the z for class 0 is a = [0, 0, 0] and
            for an example in class 3 -> a = [11.124, 34.534, 0]
    in the second linear layer we use SparseCategoricalCrossentropy in 4 neurons
    and output a vector of 4 numbers (a) which has the probability of a value to
    be in a class
    """

    # make 6-class dataset for classification
    X_train, y_train = make_blobs(
        n_samples=200, cluster_std=0.8, random_state=20,
        centers=np.array([[-5, 2], [-2, -2], [1, 2],
                         [4, -2], [7, 2], [10, -2]]),
    )

    plt_mc(X_train, y_train)

    print(f"unique classes {np.unique(y_train)}")
    print(f"10 class representation {y_train[:10]}")
    print(
        f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")

    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            Dense(3, activation='relu',   name="L1"),
            Dense(6, activation='linear', name="L2")
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    model.fit(X_train, y_train, epochs=200)

    l1 = model.get_layer("L1")
    W1, b1 = l1.get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)

    l2 = model.get_layer("L2")
    W2, b2 = l2.get_weights()
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

    prediction = model.predict(X_train)

    for i in range(5):
        print(f"""
        X_train:
        {X_train[i]}
        prediction:
        {prediction[i]}
        category:
        {np.argmax(prediction[i])}
        """)


if __name__ == "__main__":
    multi_class_classification()
