import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from utils import projToSmplx_neg, projToSmplx
alphas = [2**(elem) for elem in range(-12, 8)]

N = 25
n = 500
m = 50
np.random.seed(1)

A = np.random.randn(m, n)
b = np.random.randn(m, 1)
##cvxpy 

x = cp.Variable((n, 1))
obj = cp.Minimize(cp.max(A@x+b))
constraints = [
    x>=0,
    np.ones((n, 1)).T@x == 1
]

prob = cp.Problem(obj, constraints)
prob.solve()
x_star = x.value
f_star = prob.value
print("CVXPY", prob.value)
#print("x: ", x.value)
K_max = 100

def f_a(xk):
    return np.amax(A@xk+b)

def g_a(xk):
    max_idx = np.argmax(A@xk.reshape((n, 1))+b.reshape((m,1)))
    print(max_idx, xk.shape)
    g = A[max_idx, :].reshape((n, 1))
    return g

def proj_grad(alpha, f, gradient):
    x0 = np.ones((n, 1))
    x0 = x0/np.linalg.norm(x0, 1)

    xk = x0
    fks = []
    fbests = []
    xs = [x0]
    for k in range(1, K_max+1):
        fk = f(xk)
        if len(fbests) == 0:
            fbest = fk
        fbest = min(fk, fbest)
        g = gradient(xk)
        
        xk = xk - alpha*g
        ##proj
        #xk = xk/np.linalg.norm(xk, 1)
        print(xk.shape)
        xk = projToSmplx(xk.flatten()).reshape((n, 1))
        fbests.append(fbest)
        fks.append(fk)
        xs.append(xk)

    return xs, fbests, fks
def mirror_descent(alpha, f, gradient):
    x0 = np.ones((n, 1))
    x0 = x0/np.linalg.norm(x0, 1)
    xk = x0
    fks = []
    fbests = []
    xs = []
    for k in range(1, K_max+1):
        fk = f(xk)
        if len(fbests) == 0:
            fbest = fk
        fbest = min(fk, fbest)
        g = gradient(xk)
        
        fbests.append(fbest)
        fks.append(fk)
        xs.append(xk)
        xk = xk*np.exp(-alpha*g)/np.sum(xk*np.exp(-alpha*g))


    return xs, fbests, fks
def plot(xk, fbests, fks, label):
    plt.plot(np.array(fbests)-f_star, label=label)
    plt.legend()


optimality_gap = None 
for i, alpha in enumerate(alphas):

    xs, fbests, fks = proj_grad(alpha, f_a, g_a)
    if optimality_gap is None:
        optimality_gap = fbests[-1] - f_star
        xs_best = xs
        fs_best = fbests
        fks_best = fks
    if fbests[-1]-f_star <= optimality_gap:
        best_idx = i
        optimality_gap = fbests[-1]-f_star
        xs_best = xs
        fs_best = fbests
        fks_best = fks
plot(xs_best, fs_best, fks_best, "Proj. grad")

for i, alpha in enumerate(alphas):

    xs, fbests, fks = mirror_descent(alpha, f_a, g_a)
    if optimality_gap is None:
        optimality_gap = fbests[-1] - f_star
        xs_best = xs
        fs_best = fbests
        fks_best = fks
    if fbests[-1]-f_star <= optimality_gap:
        best_idx = i
        optimality_gap = fbests[-1]-f_star
        xs_best = xs
        fs_best = fbests
        fks_best = fks


plot(xs_best, fs_best, fks_best, "Mirror")


plt.show()
