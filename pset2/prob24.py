import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

n = 2
sigma_1 = 1.
sigma_2 = 0.5
x = np.array([0,4.])
y = np.array([7./4., 0.])
Sigma = np.diag(np.array([sigma_1, sigma_2]))
Sigma_sqrt = np.diag([np.sqrt(sigma_1), np.sqrt(sigma_2)])






def f(alpha):
    return(y[0]**2)*sigma_1/(1+sigma_1*alpha)**2+(y[1]**2)*sigma_2/(1+sigma_2*alpha)**2-1
def proj2(x):

    if np.linalg.norm(Sigma_sqrt@x)<=1:
        return x
    alpha = fsolve(f, 1)
    return np.linalg.solve(alpha*Sigma+np.eye(2),y)
    #return x/np.linalg.norm(Sigma_sqrt@x)
def proj1(x):
    x_ = np.copy(x)
    x_upd = np.array([0., 0.])
    for i in range(x_.shape[0]):
        xi = x_[i]
        if (xi-y[i])>1:
            x_upd[i] = y[i] + 1
            #diff = xi-y[i]-1
            #x_upd[i] = xi-diff

        elif (xi-y[i])<-1:
            x_upd[i] = y[i]-1
            #diff = xi-y[i]+1
            #x_upd[i] = xi-diff
          
        else: 
            x_upd[i] = xi
    return x_upd
def test_proj1():
    print("Testing proj1")

    x = np.array([0, 2])
    x_new = proj1(x)
    print("x_New", x_new)
    print("x_New-y", x_new-y)


    print(np.linalg.norm(x_new-y, ord=np.inf))
    x = np.array([2, 1])
    x_new = proj1(x)
    print("x_New", x_new)
    print("x_New-y", x_new-y)

    print(np.linalg.norm(x_new-y, ord=np.inf))

    x = np.array([-2, 1])
    x_new = proj1(x)
    print("x_New", x_new)
    print("x_New-y", x_new-y)

    print(np.linalg.norm(x_new-y, ord=np.inf))

#test_proj1()
K_max = 20
x0 = x
x_curr = np.copy(x0)
objs = []
dist1s = []
dist2s = []
for k in range(1, K_max+1):
    #alpha = 5./np.sqrt(k)
    print(x_curr-y, proj1(x_curr)-y)
    
    dist1 = np.linalg.norm(x_curr-proj1(x_curr))
    dist2 = np.linalg.norm(x_curr-proj2(x_curr))
    objs.append(dist1+dist2)


    dist1s.append(dist1)
    dist2s.append(dist2)

    
    if dist1 == 0 and dist2 == 0:
        print("Both distances 0")
        break
    if dist2 == 0:
        x_curr = proj1(x_curr)
        #g = (x_curr-proj1(x_curr))/dist1
    elif dist1 == 0: 
        x_curr = proj2(x_curr)

        
    elif dist2>=dist1:
        x_curr = proj2(x_curr)

    else:
        x_curr = proj1(x_curr)


    print(dist1, dist2)
    #x_curr = x_curr - alpha*g
print(dist1, dist2)
print( np.linalg.norm(Sigma_sqrt@x_curr), np.linalg.norm(x_curr-y, ord=np.inf))
plt.plot(objs, label="objs")
plt.plot(dist1s, label="dist1s")
plt.plot(dist2s, label="dist2s")
plt.legend()
plt.show()

