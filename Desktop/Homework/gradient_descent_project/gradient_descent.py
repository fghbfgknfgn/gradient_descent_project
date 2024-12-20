import numpy as np

def gradient_descent(x, y, lr=0.01, epochs=1000):
    n = len(y)
    w, b = 0, 0
    for _ in range(epochs):
        y_pred = w * x + b
        error = y_pred - y
        w -= lr * (-2 / n) * np.sum(x * error)
        b -= lr * (-2 / n) * np.sum(error)
    return w, b

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 4.1, 6.0, 8.1, 10.1])

w, b = gradient_descent(x, y)
print(f"w: {w}, b: {b}")
