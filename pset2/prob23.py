import numpy as np
import matplotlib.pyplot as plt

def func(x):
    if x <= 0:
        return -2*x
    if 0 < x < 4:
        return -1/2*x
    return x-6

f = np.vectorize(func, otypes=[float])
xs = np.linspace(-2, 6, num=100)
ys = f(xs)

plt.plot(xs, ys)
plt.show()