import numpy as np


def softmax(x : np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def relu(x : np.ndarray | float, a : float = 1) -> np.ndarray | float:
    return (x > 0) * a * x