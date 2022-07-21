import cvxpy as cp
import numpy as np

A = np.genfromtxt("pset4/Amatrix.csv", delimiter=",")
b = np.genfromtxt("pset4/bvector.csv", delimiter=",")

print(A.shape)
print(b.shape)
m, n = A.shape
B = cp.Variable((n,n))
d = cp.Variable((n, 1))
obj = cp.Maximize(cp.log_det(B))

constraints = [
    cp.norm2(B@ai) + ai.T@d <= bi for ai, bi in zip(A, b)
]

prob = cp.Problem(obj, constraints)
prob.solve()
print(d.value)
x_ellipsoid = d.value

## Largest euclidian ball
alpha = cp.Variable((1))
constraints.append(
    B==alpha*np.eye(n)
)

prob = cp.Problem(obj, constraints)
prob.solve()
print(d.value)

x_ball = d.value

M =int(1e6)
np.random.seed(1)

Mellipsoid = 0
Mc = 0
Mball = 0
g = np.ones((n, 1))
for i in range(M):
    x = np.random.uniform(-0.5, 0.5, size = (n,1))
    if np.any(A@x>b):
        continue
    Mc+=1

    if g.T@(x-x_ellipsoid)>=0:
        Mellipsoid+=1
    if g.T@(x-x_ball)>=0:
        Mball+=1



Rellipsoid = Mellipsoid/Mc
Rball = Mball/Mc
print("Rellipsoid", Rellipsoid)
print("Rball", Rball)