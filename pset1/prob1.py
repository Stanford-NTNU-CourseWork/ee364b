# EE364b Convex Optimization II Homework 1. Spring 2022
# Distributed Subgradient Calculation using Dask
from time import time

import dask
import matplotlib.pyplot as plt
import numpy as np
def inprod(x, y):
    return np.dot(x,y)
n, m = 100000, 4
#generate a1, ..., am in Rn randomly
def generate_data(m, n):

    data = np.random.randn(m,n)
    x= np.random.randn(n)
    return data, x
def compute_subgradient(data, x):
    output = []

    for i in range(data.shape[0]):
        output.append(inprod(data[i,:],x))
    index = np.argmax(output)

trials_count = 100
timings = []
for i in range(trials_count):

    start = time()
    data, x = generate_data(m,n)
    compute_subgradient(data, x)
    end = time()
    timings.append(end-start)

# generate x in Rn 
plt.hist(timings, bins=10)
plt.show()
print("Time spent for the computation without parallelization:",time()-start)