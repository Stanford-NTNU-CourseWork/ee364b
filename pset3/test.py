
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt

  


def obj(x1, x2, x3, A1, A2, A3, b):
    #return f_extra([x1, x2, x3])
    J = 3
    n = 100
    m = 10
    lambd = 0.5
    obj_val = 0.5*np.linalg.norm(b - A1@x1 - A2@x2 - A3@x3) + lambd*(np.linalg.norm(x1)+np.linalg.norm(x2)+np.linalg.norm(x3))
    return obj_val
def f_extra(xs_curr):
    
    left = 0.5*np.linalg.norm(b-As[0]@xs_curr[0]- As[1]@xs_curr[1]- As[2]@xs_curr[2])**2
    right = 0. 
    for A, x in zip(As, xs_curr):
        right += lambd*np.linalg.norm(x)
    return left+right


J = 3
n = 100
m = 10
lambd = 0.5


As = [np.random.uniform(low=0, high=1./np.sqrt(m), size=(m, n)) for _ in range(J)]



xs1 = [np.random.uniform(low = 0, high=1./np.sqrt(n), size=(n,1)) for _ in range(J)]




A1 = As[0]
A2 = As[1]
A3 = As[2]
x1 = xs1[0]
x2 = xs1[1]
x3 = xs1[2]


b = np.zeros((m, 1))
for A, x in zip(As, xs1):
    b += A@x 
x0s = [np.random.uniform(low = 0, high=1./np.sqrt(n), size=(n,1)) for _ in range(J)]


x10 = x0s[0]
x20 = x0s[1]
x30 = x0s[2]
x1k = x10
x2k = x20
x3k = x30

obj_list = []
max_viol = []
def get_single_gradient(x, A, xs_curr):
    x1k, x2k, x3k = xs_curr[0], xs_curr[1], xs_curr[2]
    if np.linalg.norm(x) == 0:
        return  -A.T @ (b - (A1 @ x1k + A2 @ x2k + A3 @ x3k))
    return -A.T @ (b - (A1 @ x1k + A2 @ x2k + A3 @ x3k)) + lambd * (x/np.linalg.norm(x))

K_max = 1000
for i in tqdm(range(K_max)):
    
    grads = []
    xs = [x1k, x2k, x3k]
    viols = []
    viols_inds = []
    for j in range(J):
        xk = xs[j]
        viol1 = As[j] @ xk
        viol1_ind = np.argmin(viol1)
        viols.append(viol1)
        viols_inds.append(viol1_ind)
        
        if viol1[viol1_ind] < 0:
            g1 = -As[j][viol1_ind].T
        else:
            g1 = get_single_gradient(xk, As[j], [x1k, x2k, x3k])

        grads.append(g1)

    alpha = 1/(i+1)
    alpha = 1./np.sqrt(i+1)
    upd_idx = np.argmin(np.array([viols[0][viols_inds[0]].squeeze(),viols[1][viols_inds[1]].squeeze(),viols[2][viols_inds[2]].squeeze()]))
    xs[upd_idx] -= alpha*grads[upd_idx].reshape((n, 1))

    for j, (x, g) in enumerate(zip(xs, grads)):
        #continue
        xs[j] -= alpha*g.reshape((n, 1))
    x1k, x2k, x3k = xs[0], xs[1], xs[2]
        
    obj_list.append(obj(x1k, x2k, x3k, A1, A2, A3, b))
    
    max_viol.append(np.minimum(viols[0][viols_inds[0]],viols[1][viols_inds[1]],viols[2][viols_inds[2]]))


    max_viols = []

    for m in max_viol:
        max_viols.append(min(m, 0))


plt.plot(obj_list, label="objectives")
plt.plot(np.abs(max_viol), label="max violation")
plt.legend()
plt.show()



