import numpy as np
from scipy.spatial import cKDTree
import time

start = time.time()
matrix = np.random.rand(1000, 125)
search_vec = np.random.rand(125)
k = cKDTree(matrix).query(search_vec, k=1)[1]
result = matrix[cKDTree(matrix).query(search_vec, k=1)[1]]
stop = time.time()
print(stop-start)
