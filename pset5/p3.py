from asyncio import base_tasks
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
X = np.loadtxt("X_train.csv", delimiter=",")
Y = np.loadtxt("Y_train.csv", delimiter=",").reshape((-1, 1))
n = X.shape[0]

X = np.hstack((np.ones((n, 1)), X))
print(X.shape, Y.shape)
print(1./np.linalg.eigvals(X.T@X)[0])


p = X.shape[1]-1
ps = np.array([1, 1, 5, 6, 2, 1, 1, 1])
ws = np.array([np.sqrt(pi) for pi in ps])
rho = 1
lambd=0.02
K_max = 10000
step_size = 0.005


I = np.eye(p+1)
def beta_nxt(beta_k, nu_k, alpha_k):

    res =  np.linalg.solve(1./n*X.T@X + rho*I, (1./n*X.T@Y-nu_k+alpha_k*rho))
    #print(1.0/n*X.T@(X@res-Y)+nu_k+rho*(res-alpha_k))
    return res.reshape((p+1, 1))

def alpha_nxt(beta_kp1, nu_k, alpha_k):
    alpha_kp1 = np.zeros_like(alpha_k)
    alpha_kp1[0] = nu_k[0]/rho+beta_kp1[0]

    lower = 1
    for w, pi in zip(ws, ps):
        nu_j = nu_k[lower:lower+pi]
        beta_j = beta_kp1[lower:lower+pi]
        vec = nu_j + rho*beta_j
        vec_norm = np.linalg.norm(vec, 2)
        alpha_j = np.abs(vec_norm-lambd*w)/rho*vec/vec_norm
        if vec_norm<lambd*w:
            alpha_j = 0
        
        alpha_kp1[lower:lower+pi] = alpha_j

        lower = lower+pi
    


    return alpha_kp1.reshape((p+1, 1))

def nu_nxt(beta_kp1, nu_kp, alpha_kp1):
    return nu_kp + rho*(beta_kp1-alpha_kp1).reshape((p+1, 1))

def prox(v, lambd_k):
    lower = 1
    beta = np.zeros((p+1, 1))
    beta[0] = v[0]

    for w, pi in zip(ws, ps):
        v_j = v[lower:lower+pi]
        v_norm = np.linalg.norm(v_j, 2)
        if (1-lambd*w*lambd_k/v_norm)>=0:
            beta_j = (1-lambd*w*lambd_k/v_norm)*v_j
        else:
            beta_j = 0
        beta[lower:lower+pi] = beta_j
        lower = lower+pi

    
    return beta

def f(x):
    betas = []
    lower = 1

    for pi in ps:
        betas.append(x[lower:lower+pi])
        lower = lower+pi
    return 1./(2*n)*np.linalg.norm(X@x-Y)**2 + lambd*np.sum([w*np.linalg.norm(beta) for w, beta in zip(ws, betas)])


def gradient_f(beta):
    return 1/n*X.T@(X@beta-Y)



beta_0 = np.zeros((p+1, 1))
alpha_0 = np.zeros((p+1, 1))
nu_0 = np.zeros((p+1, 1))

def admm():

    beta_k, alpha_k, nu_k = np.copy(beta_0), np.copy(alpha_0), np.copy(nu_0) 
    betas, alphas = [beta_k], [alpha_k]
    for k in tqdm(range(K_max)):
        
        beta_k = beta_nxt(beta_k, nu_k, alpha_k)
        alpha_k = alpha_nxt(beta_k, nu_k, alpha_k)
        nu_k = nu_nxt(beta_k, nu_k, alpha_k)

        betas.append(beta_k)
        alphas.append(alpha_k)


    return betas, alphas


def proximal_algorithm():

    beta_k = np.copy(beta_0)
    betas = [beta_k]
    for k in tqdm(range(K_max)):
        
        grad = gradient_f(beta_k)
        z = beta_k - step_size*grad
        z = prox(z, lambd_k=step_size)
        
        beta_k = z
        betas.append(beta_k)


    return betas



def proximal_acc_algorithm():

    beta_k = np.copy(beta_0)
    beta_km1 = beta_k
    betas = [beta_k]

    for k in tqdm(range(1, K_max+1)):
        w_k = k/(k+3)
        beta_k = beta_k + w_k*(beta_k-beta_km1)
        grad = gradient_f(beta_k)
        z = beta_k - step_size*grad
        z = prox(z, lambd_k=step_size)
        beta_km1 = beta_k
        beta_k = z
        betas.append(beta_k)


    return betas
beta_ks_prox = proximal_algorithm()
beta_ks_prox_acc = proximal_acc_algorithm()
beta_ks_admm, _ = admm()

print("Components: admm", beta_ks_admm[-1])
print("Components: prox", beta_ks_prox[-1])

lower = 0
print()
for pi in ps:
    print("Lower: ", lower)
    print("admm", beta_ks_admm[-1][lower:lower+pi])
    print("prox", beta_ks_prox[-1][lower:lower+pi])
    print()
    lower = lower+pi


#prob.solve()
#print(prob.value)
#f_star = prob.value
f_star = 49.9649
#print("admm opt", f(beta_ks_admm[-1]))
plt.semilogy([f(x)-f_star for x in beta_ks_prox], label="fk-f_star, prox")
plt.semilogy([f(x)-f_star for x in beta_ks_prox_acc], label="fk-f_star, prox acc")
plt.semilogy([f(x)-f_star for x in beta_ks_admm], label="fk-f_star, admm")
plt.legend()
plt.show()
