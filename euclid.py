from math import sqrt
import numpy as np

point0 = [3,4]
point1 = [1,8]

euclidean = sqrt((point0[0]-point1[0])**2 + (point0[1]-point1[1])**2)
euclidean1 = np.sqrt(np.sum(np.square(np.array(point0)-np.array(point1))))
euclidean2 = np.linalg.norm(np.array(point0)-np.array(point1))

print(euclidean)
print(euclidean1)
print(euclidean2)