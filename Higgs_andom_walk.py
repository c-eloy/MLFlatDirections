import numpy as np
import matplotlib.pyplot as plt
import random

# Threshold to determine how flat the direction is
threshold = 1e-4

# Definition of the potential
def V(x,y):
    return np.square(1-(np.square(x)+np.square(y))) 

# Random walk to dertermine a possible neighbouring point in a flat direction. Tests random points, return the first
# satisfying the contraint. n is the numbers of tries, and the number of test angles. The radius is also chosen randomly.
def findpoint(x_init,y_init,n,ref):
    for i in range(1, n):
        # Pick an angle and a radius randomly
        theta = random.choice(np.linspace(0, 360, n)*np.pi/180)
        r = random.randrange(1, 11, 1)*5e-2
        x_new, y_new = (x_init+r*np.cos(theta),y_init+r*np.sin(theta))

        # Test new point
        diffpot = np.abs(ref-V(x_new,y_new))
        if diffpot <= threshold:
            break
    
    # Return x_new, y_new if they are different from the initial values, and the initial values otherwise
    if diffpot <= threshold:
        return [x_new, y_new]
    else:
        return [x_init, y_init]

# Performs a p steps random walk from x_init, y_init, using findpoint with n tries.
def randomwalk(x_init,y_init,n,p):
    # Potential at initial point
    V_init = V(x_init,y_init)
    data = [findpoint(x_init,y_init,n,V_init)]

    for i in range(1, p):
        data.append(findpoint(np.array(data)[i-1,0],np.array(data)[i-1,1],n,V_init))

    return np.array(data)

data = randomwalk(-1,0,100,1000)

angles = np.linspace(0,2*np.pi,100)
plt.plot(np.cos(angles),np.sin(angles))
plt.scatter(-1,0,s=100)
plt.scatter(data[:,0], data[:,1],c='r',s=5)
plt.show()