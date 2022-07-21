import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.linalg
#randn('state',0);
np.random.seed(0)
n = 30;                                             # number of stocks
B = 5;                                              # budget
Beta = 0.1;                                         # fixed cost
Alpha = 0.05*np.random.uniform(0, 1, size=(n,1));                             # linear cost
mu = np.linspace(0.03,0.3,30).T;                        # mean return
stddev = np.linspace(0,0.4,30)
t = np.random.randn(n, n)
s = t@t.T

Sigma = np.linalg.inv(scipy.linalg.sqrtm(np.diag(np.diag(s))))@s@s.T@np.linalg.inv(scipy.linalg.sqrtm(np.diag(np.diag(s))))
Sigma = np.diag(stddev)@Sigma@np.diag(stddev);            #covariance of return
Rmin = 0.4

threshold = 1e-3


def solve(gamma, Sigma):
    x = cp.Variable((n, 1), nonneg=True)

    obj = cp.Minimize(cp.quad_form(x, Sigma))
  
    constraints = [
        mu.T@x >= Rmin,
        cp.sum(x) + Beta*gamma*cp.sum(x) + cp.sum(cp.multiply(Alpha, x)) <=B
    ]
 
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    before_stddevs=np.sqrt(prob.value)
    
    nonzero = x.value>=threshold
    zeros = x.value<threshold
    nonzero_cnt = np.sum(nonzero).astype(np.float32)

    constraints = [
        mu.T@x >= Rmin,
        cp.sum(x) + Beta*nonzero_cnt + Alpha.T@x<=B, 
        x[np.where(zeros)[0]] == 0
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    stddev = np.sqrt(prob.value)

    
    return stddev, x.value, nonzero_cnt, nonzero, before_stddevs

### Compute lower bound b:
x = cp.Variable((n, 1), nonneg=True)

obj = cp.Minimize(cp.quad_form(x, Sigma))
  
constraints = [
        mu.T@x >= Rmin,
        cp.sum(x) + 0+ Alpha.T@x <=B
    ]
prob = cp.Problem(obj, constraints)
prob.solve()
print("Lower bound b", np.sqrt(prob.value))

gammas = np.linspace(0, 25, 50)
stddevs = []
xs = []
nonzero_cnts = []
nonzeros = []
bf_stddevs = []
for gamma in gammas:
    stddev, x, nonzero_cnt, nonzero, bf_stddev = solve(gamma, Sigma)
    stddevs.append(stddev)
    xs.append(x)
    nonzero_cnts.append(nonzero_cnt)
    nonzeros.append(nonzero)
    bf_stddevs.append(bf_stddev)

stddevs = np.array(stddevs).flatten()
best_idx = np.argmin(stddevs)

plt.plot(gammas, stddevs, label ="After polishing")
plt.plot(gammas, bf_stddevs, label="Before polishing")
plt.legend()
plt.figure()
plt.plot(gammas, nonzero_cnts)
print("Best portfolio", best_idx, gamma, gammas[best_idx])
print("Nonzeros: ", np.where(nonzeros[best_idx])[0])
print(xs[best_idx][np.where(nonzeros[best_idx])[0]])
print(stddevs[best_idx])
sigma_min = stddevs[best_idx]
plt.show()


### Finding u
u = np.zeros((n, 1))
for i in range(n):
    x = cp.Variable((n, 1), nonneg=True)
    obj = cp.Maximize(x[i])

    constraints = [
            mu.T@x >= Rmin,
            cp.quad_form(x, Sigma+np.eye(n)*1e-9)<=sigma_min**2,
            cp.sum(x) + Beta*nonzero_cnt + Alpha.T@x<=B
            #cp.sum(x) + Beta*30 + Alpha.T@x<=B
        ]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    u[i] = x[i].value
x = cp.Variable((n, 1), nonneg=True)
v = 1./u
obj = cp.Minimize(cp.quad_form(x, Sigma+np.eye(30)*1e-9))
constraints = [
    mu.T@x >= Rmin,
    cp.sum(x) + Beta*cp.sum(cp.multiply(v, x)) + cp.sum(cp.multiply(Alpha, x)) <=B
]


prob = cp.Problem(obj, constraints)
prob.solve()
print(prob.status, "Final lower bound", np.sqrt(prob.value))