import math
import numpy as np
import matplotlib.pyplot as plt

def spiral_xy(i, spiral_num, n):
    rm=13
    dn=n//2
    φ = i/5.5 * math.pi # 16
    r = (rm/2) * ((dn - i)/(dn)) # 104 
    x = (r * math.cos(φ) * spiral_num)/rm + 0.5
    y = (r * math.sin(φ) * spiral_num)/rm + 0.5
    return (x, y)

def spiral(spiral_num, n):
    return [spiral_xy(i, spiral_num, n) for i in range(n//2)]


# generate the spiral data
n=30
d=3
a = spiral(1, n)
b = spiral(-1, n)
X=2*np.concatenate((a,b),axis=0)-1
X=X#/np.max(np.abs(X))

X=np.append(X,np.ones((n,1)),axis=1)
y=np.concatenate((np.ones(n//2),-np.ones(n//2)))

# visualize the spiral data
pos=np.where(y==1)
neg=np.where(y==-1)

plt.plot(X[pos,0],X[pos,1],'rx');
plt.plot(X[neg,0],X[neg,1],'gx');


# Sample diagonal matrices
P=100
Umat=np.random.randn(d,P)
dmat=(X@Umat>=0)



# Generate test data to visaulize decision boundaries
shift=0.2
numsamp=1000
x1 = np.linspace(X[:, 0].min()-shift, X[:, 0].max()+shift, numsamp)
x2 = np.linspace(X[:, 1].min()-shift, X[:, 1].max()+shift, numsamp)

# meshgrid will give regular array-like located points
Xs, Ys = np.meshgrid(x1, x2)  
Xtest=np.concatenate((Xs.reshape(-1,1),Ys.reshape(-1,1),np.ones((numsamp**2,1))),axis=1)
