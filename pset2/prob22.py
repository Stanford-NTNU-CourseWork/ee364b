import numpy as np
import matplotlib.pyplot as plt

ms = [50, 80, 90]
n = 100
K_max = 2000
x = np.random.choice([-1, 1], size=n)
def get_gradient(x_curr, A, b):
    return A.T@(A@x_curr-b)

def alpha_k(x=None,f_best=None, k=None, g=None):
    return 1./k



def proj(x):
    x_temp = np.copy(x)
    for i, xi in enumerate(x):
        if xi >=1:
            x_temp[i] = 1
        elif xi<=-1:
            x_temp[i] = -1
    return x_temp

def gradient_descent(m):
    x0 = np.ones(n)
    A = np.random.randn(m, n)
    y = A@x
    x_curr = x0
    errors = [np.linalg.norm(x_curr-x)]
    xs = []
    for k in range(1, K_max+1):
        alpha = alpha_k(k=k)
        g = get_gradient(x_curr, A, y)

        x_curr = x_curr - alpha*g
        x_curr = proj(x_curr)
        err = np.linalg.norm(x_curr-x)
        errors.append(err)
        xs.append(x_curr)
    return errors, xs[-1]

errors, xs = gradient_descent(ms[0])
errors_80, xs_80 = gradient_descent(ms[1])
errors_90, xs_90 = gradient_descent(ms[2])

plt.plot(errors, label="50")
plt.plot(errors_80, label="80")
plt.plot(errors_90, label="90")
plt.legend()
plt.show()
plt.plot(np.arange(n), xs, label="m=50")
plt.plot(np.arange(n), xs_80, label="m=80")
plt.plot(np.arange(n), xs_90, label="m=90")
plt.plot(np.arange(n), x, label="original")
plt.legend()
plt.show()

    


