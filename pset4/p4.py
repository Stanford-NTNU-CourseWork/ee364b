

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
n = 20;# % number of variables
m = 100;# % number of terms
A = np.random.randn(m,n);
b = np.random.randn(m,1);

x = cp.Variable((n, 1))
obj = cp.max(
            A@x-b
            )
constraints = [
    cp.norm(x, "inf") <= 1
]
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()
p_star = prob.value
def f_a(xk):
    return np.amax(A@xk-b)

def g_a(xk):
    max_idx = np.argmax(A@xk-b)
    g = A[max_idx, :].reshape((n, 1))
    return g
def iterate(f_func, g_func):
    a_s = []
    K_max = 40
    bs = []
    x0 = np.ones((n, 1))
    xk = x0
    fk_best = None
    xs = [x0]
    Lks, Uks = [], []
    for k in range(1, K_max+1):
        fk = f_func(xk)
        if fk_best is None:
            fk_best = fk
        fk_best = min(fk, fk_best)
        
        g = g_func(xk)
        a_s.append(g.T)
        
        bs.append((fk-g.T@xk))
        Aapprox = np.vstack(a_s)
        Bapprox = np.vstack(bs)
        assert(Bapprox.shape==(k, 1))
        assert(Aapprox.shape == (k, n)), Aapprox.shape
        Uk = fk_best

        x = cp.Variable((n, 1))
        obj = cp.max(
            Aapprox@x+Bapprox
            )
        constraints = [
            cp.norm(x, "inf") <= 1
        ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()
        Lk = prob.value
        
        xk = x.value
        Lks.append(Lk)
        Uks.append(Uk)
        xs.append(xk)
    return xs, Lks, Uks, 

def plot(xs, Lks, Uks, vals):

    plt.plot(vals, label="fk")
    plt.plot(Lks, label="Lks")
    plt.plot(Uks, label="Uks")
    plt.plot(p_star*np.ones_like(vals), label="p_star")
    plt.legend()
    plt.show()


xs, Lks, Uks = iterate(f_a, g_a)
vals = [f_a(x) for x in xs]
plot(xs, Lks, Uks, vals)




### b)

c = np.random.uniform(-1, 1, size=(n, 1))
p_star = 0
def g_l2(x):
    return (x-c)/(np.linalg.norm(x-c))
def f_l2(x):
    return np.linalg.norm(x-c, 2)

xs, Lks, Uks = iterate(f_l2, g_l2)
vals = [f_l2(x) for x in xs]
plot(xs, Lks, Uks, vals)