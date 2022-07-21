from hw7_q1 import *
import cvxpy as cp



def get_D(i):
    # hi = np.random.randn(d, 1)
    # diagonal = (X@hi >= 0).astype(int)
    # D = np.diag(diagonal.flatten())
    return np.diag(dmat[:,i].flatten())

P = 100
beta = 0.9
Ds = [get_D(i) for i in range(P)]
u_mat = cp.Variable((d, P))
up_mat = cp.Variable((d, P))



dj_sum = 0
cost = 0
constr = []

for i in range(P):
    Di = np.diag(dmat[:, i])
    dj_sum += Di@X@(u_mat[:, i] - up_mat[:, i])
    cost += beta*(cp.norm(u_mat[:, i]) + cp.norm(up_mat[:, i]))
    
    constr += [(2*Di - np.eye(n))@X@u_mat[:, i] >= 0]
    constr += [(2*Di - np.eye(n))@X@up_mat[:, i] >= 0]
    
cost = cp.norm(dj_sum - y)**2
# cost = cp.Minimize(
#     cp.square(cp.norm2(
#         cp.sum([(Ds[j]@X@(u[:, j]-u_p[:,j])) for j in range(P)])-y
#     ))+
#     beta*cp.sum([cp.norm2(u[:,j].flatten()) + cp.norm2(u_p[:, j].flatten()) for j in range(P)])
# )

constraints = []
I = np.eye(n)
for i in range(P):
    constraints.append(
       (2*Ds[i]-I)@X@u_mat[:, i].flatten()>=0
    )
    constraints.append(
       (2*Ds[i]-I)@X@up_mat[:, i].flatten()>=0
    )

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()
print(prob.value)
print(prob.status)


w1 = np.zeros((d, P))
w2 = np.zeros((1, P))
w1p = np.zeros((d, P))
w2p = np.zeros((1, P))


for i in range(P):
    w1[:, i] = u_mat.value[:, i]/np.sqrt(np.linalg.norm(u_mat.value[:, i]))    
    w2[:, i] = np.sqrt(np.linalg.norm(u_mat.value[:, i]))
    w1p[:, i] = up_mat.value[:, i]/np.sqrt(np.linalg.norm(up_mat.value[:, i]))    
    w2p[:, i] = np.sqrt(np.linalg.norm(up_mat.value[:, i]))

def f(x):
    total = 0
    for j in range(P):
        total += max(0, np.dot(x, w1[:,j]))*w2[:, j]
        total -= max(0, np.dot(x, w1p[:,j]))*w2p[:, j]
    return total


def get_mse(X, y):
    mse = np.mean([
    (f(x).flatten()-yt.flatten() )**2
    for x, yt in zip(X, y)])

    return mse
print(get_mse(X, y))
print(get_mse(Xtest, y))