import numpy as np
import matplotlib.pyplot as plt
from utils import *

n = 10
R_emp = log_opt_return_sample(10000)

def gradient(x):
    samples = log_opt_return_sample(1)
    
    r = samples
    
    return r/np.dot(x.flatten(), r.flatten()), samples

x = 1.4*np.ones((n, 1))
K_max = 1000

best = []
obj_hats = []
def iterate(x0, c=False):
    x = x0
    for i in range(K_max):
        alpha = 1./(1+i)
        grad, samples = gradient(x)
        grad = grad.reshape((n, 1))  
        
        x += alpha*grad
        #projection step
        if c == False:
            x = projToSmplx(x.reshape((n, 1))).reshape((n, 1))
        else: 
            x = projToSmplx_neg(x.reshape((n, 1))).reshape((n, 1))

        #Not necessary as already done. 
        #x = np.maximum(x, np.zeros_like(x))
        obj_hat = np.mean(np.log(R_emp.T@x))
        if len(best) == 0 or obj_hat > max_obj_hat:
            max_obj_hat = obj_hat
        best.append(max_obj_hat)
        obj_hats.append(obj_hat)
    return best


def projToSmplx_neg(v):
    """
    Input: v is an array with shape (n,)
    Output: x is the result of projection
    Description: project v into simplex
    """
    v = v.squeeze()
    u = np.sort(v)[::-1]
    sv = u.cumsum()
    ind = np.nonzero(u > (sv- 1) / np.arange(1, u.size+1))[0][-1]
    tau = (sv[ind] - 1) / (ind + 1) 
    x = v
    return x
iterate(np.copy(x), True)
plt.plot(best, label="best")
plt.plot(obj_hats, label="obj")
plt.legend()
plt.show()
