import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def custom_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, ))
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return (a_out)


def custom_dense_vectorized(AT, W, B):
    """
    Computes dense layer vectorized
    Args:
      AT (A transpose) -> ndarray (1,n)
      W                -> ndarray (n,j) : Weight matrix, n features per unit, j units
      B                -> ndarray (1,j) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    Z = np.matmul(AT, W) + B
    return sigmoid(Z)


def custom_sequential_inference(x, W1, b1, W2, b2):
    a1 = custom_dense(x,  W1, b1)
    a2 = custom_dense(a1, W2, b2)
    return (a2)
