import numpy as np
import scipy.linalg

np.random.seed(0)
n = 30;                                             # number of stocks
B = 5;                                              # budget
Beta = 0.1;                                         # fixed cost
Alpha = 0.05*np.random.uniform(0, 1, size=(n,1));                             # linear cost
mu = np.linspace(0.03,0.3,30).T;                        # mean return
stddev = np.linspace(0,0.4,30)
t = np.random.randn(n).reshape((n, 1))
s = t@t.T
Sigma = np.linalg.inv(scipy.linalg.sqrtm(np.diag(np.diag(s))))@s@s.T@np.linalg.inv(scipy.linalg.sqrtm(np.diag(np.diag(s))))
Sigma = np.diag(stddev)@Sigma@np.diag(stddev);            #covariance of return
Rmin = 0.4