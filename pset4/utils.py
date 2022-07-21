""" 
Implements some HW3 utility functions for the stochastic programming problem
"""
import numpy as np
import scipy.linalg
def projToSmplx(v):
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
    x = np.maximum( v - tau, np.zeros_like(v) )
    return x
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
def log_opt_return_sample(m):
    """
    return oracle
     - outputs n X m matrix of log normal mixture
    """
    # asset dimension 
    n = 10;
    # generate samples from mixture distribution
    # ------------------------------------
    # means
    mu1 = np.array([0.4, 0.2, 0.4, 0.7, 0.9, 0.3, 0.9, 0.5, 0.5, 0.1])
    mu2 = np.array([5.1, 0.9, 0.7, 1.6, 2.2, 4.4, 3.2, 6.2, 4.2, 0.7])
    # covariance
    A = np.random.randn(n, n)
    A = A.dot(A.T)
    B = np.random.randn(n, n) 
    B = B.dot(B.T)
    C1 = np.diag(1/np.sqrt(np.diag(A))).dot(A.dot( 
np.diag((1/np.sqrt(np.diag(A))))))
    C2 = np.diag(1/np.sqrt(np.diag(B))).dot(B.dot( 
np.diag((1/np.sqrt(np.diag(B))))))
    sigmas1 = np.array([0.01, 0.03, 0.05, 0.04, 0.05, 0.09, 0.05, 0.01, 0.03, 
0.12])
    sigmas2 = np.array([0.81, 0.31, 0.74, 0.91, 0.67, 0.71, 0.31, 0.42, 0.51, 
0.41])
    sigma1 = np.diag(sigmas1).dot( C1.dot(np.diag(sigmas1)) )
    s1Half = scipy.linalg.sqrtm(sigma1)
    sigma2 = np.diag(sigmas2).dot(C2.dot(np.diag(sigmas2)))
    s2Half = scipy.linalg.sqrtm(sigma2)
    # bernoulli trials
    p = np.tile(np.random.rand(1,m) <= 0.9, (n, 1)) 
    q = 1 - p
    # samples from first distribution 
    r1 = np.exp(np.tile(mu1, (m,1)).T + s1Half.dot(np.random.randn(n,m)))
    r1 = p*r1
    # samples from second distribution 
    r2 = np.exp(np.tile(mu2, (m,1)).T + s2Half.dot(np.random.randn(n,m)))
    r2 = q*r2
    # mixture
    R = r1 + r2
    return R