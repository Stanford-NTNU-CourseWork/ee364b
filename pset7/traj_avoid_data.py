import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.patches import Rectangle

# from http://python4econ.blogspot.com/2013/03/matlabs-cylinder-command-in-python.html
def cylinder(r,n):
    '''
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:  r - a vector of radii
             n - number of coordinates to return for each element in r

    OUTPUTS: x,y,z - coordinates of points
    '''

    # ensure that r is a column vector
    r = np.atleast_2d(r)
    r_rows,r_cols = r.shape
    
    if r_cols > r_rows:
        r = r.T

    # find points along x and y axes
    points  = np.linspace(0,2*np.pi,n+1)
    x = np.cos(points)*r
    y = np.sin(points)*r

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(0,1,len(r)))
    z = np.ones((1,n+1))*rpoints.T
    
    return x,y,z

n = 2
N = 3
T = 30
D = 0.4

# initial and final positions
p1init = np.array([-1, 0])
p2init = np.array([-1, 0.5])
p3init = np.array([-1, 1])

p1final = np.array([1, 0])
p2final = np.array([1, -0.5])
p3final = np.array([1, -1])

# uniform speed colliding trajectories
p1 = np.tile(p1init[:, np.newaxis], (1, T)) + (p1final-p1init)[:,np.newaxis] * np.linspace(0,1,T)
p2 = np.tile(p2init[:, np.newaxis], (1, T)) + (p2final-p2init)[:,np.newaxis] * np.linspace(0,1,T)
p3 = np.tile(p3init[:, np.newaxis], (1, T)) + (p3final-p3init)[:,np.newaxis] * np.linspace(0,1,T)

x, y, z = cylinder(D/2, 100)
x = x.squeeze()
y = y.squeeze()
z = z.squeeze()

# plt.figure()
# plt.axis('equal')
# plt.axis([-1.5, 1.5, -1.5, 1.5])
# p1_data = plt.plot(p1[0,:], p1[1,:], color='C0')[0]
# p2_data = plt.plot(p2[0,:], p2[1,:], color='C1')[0]
# p3_data = plt.plot(p3[0,:], p3[1,:], color='C2')[0]

# for i in range(T):
#     # if you want diamonds like matlab, use marker='d'. it is filled in, though
#     plt.scatter(p1[0,:], p1[1,:], color='C0', marker='.')
#     plt.scatter(p2[0,:], p2[1,:], color='C1', marker='.')
#     plt.scatter(p3[0,:], p3[1,:], color='C2', marker='.')
#     p1_data.set_data(p1[0,i]+x, p1[1,i]+y)
#     p2_data.set_data(p2[0,i]+x, p2[1,i]+y)
#     p3_data.set_data(p3[0,i]+x, p3[1,i]+y)
#     ax = plt.gca()
#     ax.add_patch(Rectangle([-1, -1], 2, 2, facecolor='none', edgecolor='black', linestyle=':'))
#     plt.pause(0.1)
# plt.show()