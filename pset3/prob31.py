import enum
from tkinter import X
import numpy as np
import matplotlib.pyplot as plt

J = 3
n = 100
m = 10 
lambd = 0.5

interval_top = 1./np.sqrt(m) # non inclusive, but continuous so not an issue. 
interval_bottom = 0

A1 = np.random.uniform(0, 1/np.sqrt(m), (m, n))
A2 = np.random.uniform(0, 1/np.sqrt(m), (m, n))
A3 = np.random.uniform(0, 1/np.sqrt(m), (m, n))
As = [A1, A2, A3]

x1 = np.random.uniform(0,1/np.sqrt(n), (n, 1))
x2 = np.random.uniform(0,1/np.sqrt(n), (n, 1))
x3 = np.random.uniform(0,1/np.sqrt(n), (n, 1))
xs1 = [x1, x2, x3]

xs1 = [np.random.uniform(low = 0, high=1./np.sqrt(n), size=(n,1)) for _ in range(J)]



#A = np.hstack(As)
#x = np.vstack(xs)

#assert(A.shape == (m, n*J)), f"A.shape: {A.shape}"
#assert(x.shape == (n*J,1 ))

K_max = 20
def obj(x1, x2, x3, A1, A2, A3, b):
    J = 3
    n = 100
    m = 10
    lambd = 0.5
    obj_val = 0.5*np.linalg.norm(b - A1@x1 - A2@x2 - A3@x3)**2 + lambd*(np.linalg.norm(x1)+np.linalg.norm(x2)+np.linalg.norm(x3))
    return obj_val

def get_single_gradient(x, A, xs_curr):
    print(x.shape, A.shape)
    if np.linalg.norm(x) == 0:
        right = 0
    else: 
        right = lambd*(x/np.linalg.norm(x))
    g = -A.T@(b-(As[0]@xs_curr[0]+As[1]@xs_curr[1]+As[2]@xs_curr[2]))+right
    return g


def f_extra(xs_curr):
    return obj(xs_curr[0], xs_curr[1], xs_curr[2],A1, A2, A3, b )
    sum_ = b
    
    for A, x in zip(As, xs_curr):
        sum_ += -A@x
    left = 1./2* np.linalg.norm(b-A[0]@xs_curr[0]- A[1]@xs_curr[1]- A[2]@xs_curr[2])**2
    right = 0. 
    for A, x in zip(As, xs_curr):
        right += lambd*np.linalg.norm(x)
    return left+right




x0s = [np.zeros((n, 1)) for j in range(J)]
x0s = [np.random.uniform(low = 0, high=1./np.sqrt(n), size=(n,1)) for _ in range(J)]

xs_curr = x0s
objs = [f_extra(xs_curr)]
K_max = 100
max_violations = []


def iterate():
    for k in range(1, K_max+1):
        gs = []

        for j in range(J):
            res =  As[j]@xs_curr[j]
            res_smallest = np.argmin(res)
            if res[res_smallest] < 0:
                gs.append(-As[j][res_smallest, :].T)
            else:
                gs.append(get_single_gradient(xs_curr[j], As[j], xs_curr))
            

        #if len(violations) == 0:
            #gs = get_gradients_obj(xs)

        alpha = 1/k
        #alpha = 1./np.sqrt(k)
        #print(xs[0])
        for j, x in enumerate(xs_curr):
            #print(gs[j].reshape((n, 1)))
            xs_curr[j] = xs_curr[j] - alpha*(gs[j].reshape((n, 1)))
        #print([x_old-x for x_old, x in zip(xs_old, xs)])
        #max_violations.append(max_violation)
        objs.append(f_extra(xs_curr))

iterate()


#x0 = np.zeros((n*J, 1))
#xcurr = x0

def f(x):
    
    left = 1./2* np.linalg.norm(A@x-b)**2
    right = lambd*sum([np.linalg.norm(x[j*n:(j+1)*n]) for j in range(J)])
    return  left+right



plt.plot(objs, label="objs")
plt.plot(max_violations, label="violations")
plt.legend()
plt.show()




def get_gradients_obj(xs):

    grads = []
    sum_ = b
    for A, x in zip(As, xs):
        grads.append(get_single_gradient(x, A, xs))
    return grads
    grads = []
    for A, x in zip(As, xs):
        sum_ += -A@x
    for A, x in zip(As, xs):
        left = -A.T@(sum_)
        print(left.shape)
        right = lambd*x/np.linalg.norm(x)
        print(left+right)
        grads.append(left+right)
    return grads

def iterate_vectorized():
    for k in range(1, K_max+1):
        violations = []
        max_violation = 0
        g = np.zeros((n*J,))
        for j in range(J):
            for i in range(m):
                a_i = As[j][i, :]
                assert(a_i.shape==(n,)), f"ai shape: {a_i.shape}"
                x_j = xcurr[j*n:(j+1)*n]
                assert(x_j.shape==(n,1)), f"xj shape: {x_j.shape}"
                res = np.dot(a_i, x_j.flatten())
                if res>=0:
                    continue
                if -res > max_violation:
                    max_violation = -res
                    g = np.zeros((n*J,))

                    g[j*n:(j+1)*n] = -a_i

                violations.append(-res)
        if len(violations) == 0:
            g = get_gradient_obj(x)
        else: 
            max_violation = max(violations)

        alpha = 1./k
        alpha = 1./np.sqrt(k)
        print(xcurr.shape, g.shape)
        xcurr = xcurr -alpha*g.reshape((n*J, 1))
        print(xcurr.shape)

        max_violations.append(max_violation)
        objs.append(f(xcurr))



def get_gradient_obj(x):
    left = A.T@(A@x-b)
    xs = []
    for j in range(J):
        xj = x[j*n:(j+1)*n]/np.linalg.norm(x[j*n:(j+1)*n])
        xs.append(xj)
    right = np.vstack(xs)
    return left+right


    


