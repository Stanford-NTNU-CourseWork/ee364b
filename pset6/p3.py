import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import cho_solve, cho_factor
from datetime import datetime
m, n = 100, 500

w = np.random.randn(n, 1)

xstar = np.maximum(w, 0)
lambda_star = np.maximum(-w, 0)

A = np.random.randn(m, n)
nu_star = np.random.randn(m, 1)
b = A@xstar
c = -A.T@nu_star + lambda_star
optimal_value = c.T@xstar

F = np.vstack((
    np.hstack((A, np.zeros((m, m+n)))),
    np.hstack((np.zeros((n, n)), -A.T, np.eye(n))),
    np.hstack((c.flatten(), b.flatten(), np.zeros(( n,))))
))
g = np.vstack((
    b, c, np.zeros(1,)
))
print(g.shape)

c, low = cho_factor(F@F.T) 
def A_proj(x):
    x = np.copy(x)
    return x-F.T@cho_solve((c, low), F@x-g)

def C_proj(x):
    x = np.copy(x)
    x[:n] = np.maximum(x[:n], np.zeros_like(x[:n]))
    x[-n:] = np.maximum(x[-n:],  np.zeros_like(x[-n:]))
    return x


z0 = -np.ones((n+n+m, 1))

MAX_ITER = 1000
z = np.copy(z0)
residuals = []
t1 = datetime.now()
eps = 1e-8
for k in tqdm(range(MAX_ITER)):

    zkhalf = A_proj(z)
    z = C_proj(zkhalf)

    res = np.linalg.norm(z-zkhalf)

    residuals.append(res)
    if res<eps:
        t2 = datetime.now()

        print(t2-t1)
        break
t2 = datetime.now()
print("Vanilla : ", t2-t1)
residuals = np.array(residuals)

plt.plot(residuals)
plt.title("Residuals")
plt.savefig("residuals_p3")
plt.show()
## Dykstra
z = np.copy(z0)

t1 = datetime.now()
residuals  = []
for k in tqdm(range(MAX_ITER)):

    xkhalf = A_proj(z)
    zkhalf = np.copy(2*xkhalf - z)

    x = C_proj(zkhalf)
    z += np.copy(x-xkhalf)
    res = np.linalg.norm(x-xkhalf)

    residuals.append(res)
    if res<eps:
        t2 = datetime.now()

        print(t2-t1)
        break
t2 = datetime.now()
print("Dykstra : ", t2-t1)

residuals = np.array(residuals)

plt.plot(residuals)
plt.title("Residuals")
plt.savefig("residuals_p3_dykstra")
plt.show()
## Dykstra