from mimetypes import init
import cvxpy as cp
import numpy as np

from traj_avoid_data import *
assert(N==3)


def get_avoidance_constraints(p1k, p2k, p3k, p1, p2, p3):
    constraints = []
    diff12, diff13, diff23 = p1k-p2k, p1k-p3k, p2k-p3k
    
    v1 = cp.Variable((T, 2))
    v2 = cp.Variable((T, 2))
    v3 = cp.Variable((T, 2))
    constraints += [
        v1 == p1-p1k,
        v2 == p2-p2k,
        v3 == p3-p3k
    ]
    for t in range(1, T-1):
        norm12 = np.linalg.norm(diff12[t])
        norm13 = np.linalg.norm(diff13[t])
        norm23 = np.linalg.norm(diff23[t])

        constr1 = norm12 + 1./norm12*(v1[t].T-v2[t].T)@diff12[t]>=D
        constr2 = norm13 + 1./norm13*(v1[t].T-v3[t].T)@diff13[t]>=D
        constr3 = norm23 + 1./norm23*(v2[t].T-v3[t].T)@diff23[t]>=D
       
        constraints.append(constr1)
        constraints.append(constr2)        
        constraints.append(constr3)
    return constraints





def optimize(p10, p20, p30):

    p10[0] = np.copy(p1init)
    p10[-1] = np.copy(p1final)
    p20[0] = np.copy(p2init)
    p20[-1] = np.copy(p2final)
    p30[0] = np.copy(p3init)
    p30[-1] = np.copy(p3final)
    p1 = cp.Variable((T, 2))
    p2 = cp.Variable((T, 2))
    p3 = cp.Variable((T, 2))

    obj = cp.Minimize(
        cp.sum_squares(cp.norm2(p1[1:] - p1[:-1], axis=1))+
        cp.sum_squares(cp.norm2(p2[1:] - p2[:-1], axis=1))+
        cp.sum_squares(cp.norm2(p3[1:] - p3[:-1], axis=1))
    )

    init_final_constraints = [ 
        p1[0] == p1init, p1[-1] == p1final,
        p2[0] == p2init, p2[-1] == p2final,
        p3[0] == p3init, p3[-1] == p3final,
    ]
    MAX_ITER = 30
    p1k, p2k, p3k = p10, p20, p30
    values = []
    for iter in range(MAX_ITER):
        
        av_constr = get_avoidance_constraints(p1k, p2k, p3k, p1, p2, p3)
        constraints = init_final_constraints+av_constr
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print(prob.status)
        print(prob.value)

        p1k = p1.value
        p2k = p2.value
        p3k = p3.value
        # assert(np.all(np.linalg.norm(p1k-p2k, axis=1)>=D))
        # assert(np.all(np.linalg.norm(p1k-p3k, axis=1)>=D))
        # assert(np.all(np.linalg.norm(p2k-p3k, axis=1)>=D))
        if iter == 0:
            dist1, dist2, dist3 = np.linalg.norm(p1k-p2k, axis=1), np.linalg.norm(p1k-p3k, axis=1), np.linalg.norm(p3k-p2k, axis=1)
        values.append(prob.value)

    return p1k, p2k, p3k, prob.value, values, dist1, dist2, dist3



def plot(p1, p2, p3):
    p1, p2, p3 = p1.T, p2.T, p3.T
    plt.figure()
    plt.axis('equal')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    p1_data = plt.plot(p1[0,:], p1[1,:], color='C0')[0]
    p2_data = plt.plot(p2[0,:], p2[1,:], color='C1')[0]
    p3_data = plt.plot(p3[0,:], p3[1,:], color='C2')[0]

    for i in range(T):
        # if you want diamonds like matlab, use marker='d'. it is filled in, though
        plt.scatter(p1[0,:], p1[1,:], color='C0', marker='.')
        plt.scatter(p2[0,:], p2[1,:], color='C1', marker='.')
        plt.scatter(p3[0,:], p3[1,:], color='C2', marker='.')
        p1_data.set_data(p1[0,i]+x, p1[1,i]+y)
        p2_data.set_data(p2[0,i]+x, p2[1,i]+y)
        p3_data.set_data(p3[0,i]+x, p3[1,i]+y)
        ax = plt.gca()
        ax.add_patch(Rectangle([-1, -1], 2, 2, facecolor='none', edgecolor='black', linestyle=':'))
        plt.pause(0.1)
    plt.show()

def get_randoms():
    np.tile(p2init[:, np.newaxis], (1, T)) + (p2final-p2init)[:,np.newaxis] * np.linspace(0,1,T)
    np.tile(p3init[:, np.newaxis], (1, T)) + (p3final-p3init)[:,np.newaxis] * np.linspace(0,1,T)

    p10 = np.random.randn(T, 2)*10+ (np.tile(p1init[:, np.newaxis], (1, T)) + (p1final-p1init)[:,np.newaxis] * np.linspace(0,1,T)).T
    p20 = np.random.randn(T, 2)*10 + (np.tile(p2init[:, np.newaxis], (1, T)) + (p2final-p2init)[:,np.newaxis] * np.linspace(0,1,T)).T
    p30 = np.random.randn(T, 2)*10+ (np.tile(p3init[:, np.newaxis], (1, T)) + (p3final-p3init)[:,np.newaxis] * np.linspace(0,1,T)).T


    return p10, p20, p30

best_val = 100
best_p1, best_p2, best_p3 = None,None, None
vals = []
dist1s, dist2s, dist3s = [], [], []
K = 5
for _ in range(K):

    p10, p20, p30 = get_randoms()
    p1, p2, p3, val, values, dist1, dist2, dist3 = optimize(p10, p20, p30)

    dist1s.append(dist1)
    dist2s.append(dist2)
    dist3s.append(dist3)

    vals.append(values)
    if val <= best_val:
        best_p1, best_p2, best_p3 = np.copy(p1), np.copy(p2), np.copy(p3)
        best_val = val

dists = np.hstack((np.linalg.norm(p1-p2, axis=1).reshape((-1, 1)), np.linalg.norm(p1-p3, axis=1).reshape((-1, 1)), np.linalg.norm(p3-p2, axis=1).reshape((-1, 1))))
min_dists = np.min(dists, axis=1)
plt.figure()
plt.plot(min_dists)
plt.title("min dist")
fig, ax = plt.subplots(K)
for i, (dist1, dist2, dist3) in enumerate(zip(dist1s, dist2s, dist3s)):
    ax[i].plot(dist1, label="dist(p1,p2)")
    ax[i].plot(dist2, label="dist(p1,p3)")
    ax[i].plot(dist3, label="dist(p2,p3)")
    ax[i].plot([D]*len(dist1), label="threshold")
    ax[i].set_ylim([D-0.2, D+0.2])
    ax[i].set_title(f"Run: {i}, distances")
    ax[i].legend()
plt.tight_layout()
plt.figure()
for values in vals:
    plt.plot(values)
plt.title("J vs iter")
plt.semilogy()

plt.show()
print(best_val)
plot(best_p1, best_p2, best_p3)
    