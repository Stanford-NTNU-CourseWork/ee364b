import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from tqdm import tqdm
from datetime import datetime
exponent = 12
m, n = 2**exponent, 400

b = np.random.randn(m, )
A = np.random.randn(m, n)*scipy.sparse.diags(np.linspace(0.001, 100, n))
AtA = A.T@A
Atb = A.T@b
normatb = np.linalg.norm(Atb)
x0 = np.zeros((n,))

norms = []
def get_hadamard(exponent):
    H = np.array([[1, 1], [1, -1]])
    
    for i in range(1, exponent):
        H = np.vstack((
            np.hstack((H, H)),
            np.hstack((H, -H))
        ))
    
    return H/np.sqrt(2**exponent)

#print(get_hadamard(2))
p=20
S = scipy.sparse.diags(np.random.choice([1, -1], size=(m,)))
SA = S@A

random_indices = np.random.choice(m,size=n+p,replace=False)
H = scipy.linalg.hadamard(m)/np.sqrt(m)
H_r = H[random_indices, :]

Z = H_r@SA
M_inv = Z.T@Z
M = np.linalg.inv(M_inv)

def residual_norm(x):
    return np.linalg.norm(Atb-AtA@x)

def callback_function(xk, norms):
    norms.append(residual_norm(xk))
norms_with=[]
norms_without = []
t1 = datetime.now()
xs = scipy.sparse.linalg.cg(AtA, Atb, x0=x0, tol=1e-05, M=M,callback= lambda x: callback_function(x, norms_with), maxiter=1000,atol=None)
t2 = datetime.now()
xs = scipy.sparse.linalg.cg(AtA, Atb, x0=x0, tol=1e-05,callback= lambda x: callback_function(x, norms_without), maxiter=1000,atol=None)
t3 = datetime.now()
print("CG w M", t2-t1)
print("CG vanilla", t3-t1)
def cond(W):
    Eigs = np.linalg.eigvals(W)
    return np.max(Eigs)/np.min(Eigs)
M_sqrt = scipy.linalg.sqrtm(M)
print("Condition number M", cond(M_sqrt@AtA@M_sqrt.T))
print("Condition number AtA", cond(AtA))
plt.figure()
plt.plot(norms_with/normatb)
plt.title("p4_norms_cg_with_M")
plt.semilogy()
plt.savefig("p4_norms_cg_with_M")

plt.figure()
plt.plot(norms_without/normatb)
plt.title("p4_norms_cg_without_M")
plt.semilogy()
plt.savefig("p4_norms_cg_without_M")
plt.show()


x0 = np.zeros((n, 1))

r = np.copy(Atb)
rho0 = np.linalg.norm(r)**2
x = np.copy(x0)
rho_m1 = rho0
rho_m2 = rho0
K_max = 1000
eps = 1e-8
norms = []
print(cond(AtA))
### Our A is AtA, our b is Atb
for k in tqdm(range(1, K_max+1)):
    if np.sqrt(rho_m1)<= eps*np.linalg.norm(Atb):
        break

    if k == 1:
        p = np.copy(r)
    else:
        p = np.copy(r)+rho_m1/rho_m2*p
    w = AtA@p
    alpha = rho_m1/(p.T@w)
    x += alpha*p
    r -= alpha*w 

    rho_m2 = rho_m1
    rho_m1 = np.linalg.norm(r)**2
    norms.append(residual_norm(x))
norms = np.array(norms)/normatb
plt.plot(norms)
plt.savefig("p4_norms_cg")

plt.show()


## Defining M: 


assert(M.shape==(n, n))
#M = np.linalg.inv(A.T@phi.T@phi@A)



x0 = np.zeros((n,))



x = np.zeros((n, 1))
r = np.copy(Atb-AtA@x)
p = np.copy(r)
z = M@r
rho1 = r.T@z
rho = rho1

plt.plot(np.array(norms)/normatb)
plt.savefig("p4_norms_cg_M")
plt.show()
#raise Exception

for k in range(1, K_max+1):
    if np.sqrt(rho)<=eps*np.linalg.norm(Atb) or np.linalg.norm(r)<= eps*np.linalg.norm(Atb):
        break

    w = AtA@p
    alpha = rho/(w.T@p)
    x += alpha*p
    r -= alpha*w
    z = M@r
    rho_kp1 = z.T@r
    p = z+(rho_kp1/rho)*p
    rho = rho_kp1
    norms.append(residual_norm(x))

norms = np.array(norms)/normatb
print(len(norms))
print(norms[-1])
plt.plot(norms)
plt.savefig("p4_norms_cg_M")
plt.show()