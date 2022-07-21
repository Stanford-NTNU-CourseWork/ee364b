import numpy as np 
import matplotlib.pyplot as plt

n = 1000
m = 200
lambd = 0.01
var = 1/m
sigma = np.sqrt(var)
A = np.random.randn(m, n)*sigma

k = 5

x_star = np.zeros((n))
x_star[:k] = np.ones((k))

b = A@x_star

x0 = np.zeros((n))

def get_gradient(x):
    first = A.T@(A@x-b)
    second = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi==0:
            second[i] = 0#np.abs(xi)
        else:
            second[i] = np.sign(xi)
    return first+lambd*second

def f(x):
    return 1./2*np.linalg.norm(A@x-b)**2 + lambd*np.linalg.norm(x, 1)
def get_polyak(x, f_best, k, g):
    return (f(x)-f_best+1./k)/(np.linalg.norm(g)**2)
def alpha_const(x, f_best, k, g):
    return 0.001
def alpha_sqrt(x, f_best, k, g):
    return 1./np.sqrt(k)
def alpha_k(x,f_best, k, g):
    return 1./k

K_max = 200



def subgradient_method(x0, K_max, alpha_func, beta=0):
    xs = []
    fs = [f(x0)]
    x_curr = x0
    x_prev = x0
    f_best = f(x_curr)


    for k in range(1, K_max+1):
        g = get_gradient(x_curr)
        alpha = alpha_func(x_curr, f_best, k, g)
        x_temp = x_curr
        x_curr = x_curr - alpha*g + beta*(x_curr - x_prev)
      
        x_prev = x_temp
        f_best = min(f_best, f(x_curr))
        fs.append(f(x_curr))
    return fs

fs_const = subgradient_method(x0, K_max, alpha_const)
fs_k = subgradient_method(x0, K_max, alpha_k)
fs_ksqrt = subgradient_method(x0, K_max, alpha_sqrt)
fs_polyak = subgradient_method(x0, K_max, get_polyak)




plt.plot(np.arange(1,len(fs_const)+1), fs_const, label="const")

plt.plot(np.arange(1,len(fs_k)+1), fs_k, label="1/k")

plt.plot(np.arange(1,len(fs_ksqrt)+1), fs_ksqrt, label="1/sqrt(k)")
plt.plot(np.arange(1,len(fs_polyak)+1), fs_polyak, label="polyak")
plt.legend()
plt.ylim([0, 10])
plt.show()

beta = 0.5
fs_const = subgradient_method(x0, K_max, alpha_const, beta)
fs_k = subgradient_method(x0, K_max, alpha_k, beta)
fs_ksqrt = subgradient_method(x0, K_max, alpha_sqrt, beta)
fs_polyak = subgradient_method(x0, K_max, get_polyak, beta)

plt.plot(np.arange(1,len(fs_const)+1), fs_const, label="const")

plt.plot(np.arange(1,len(fs_k)+1), fs_k, label="1/k")

plt.plot(np.arange(1,len(fs_ksqrt)+1), fs_ksqrt, label="1/sqrt(k)")
plt.plot(np.arange(1,len(fs_polyak)+1), fs_polyak, label="polyak")
plt.legend()
plt.ylim([0, 10])
plt.show()