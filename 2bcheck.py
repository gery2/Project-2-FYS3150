
import numpy as np
from scipy.sparse import diags

import time
t0= time.clock()

N = 6

diagonals = [np.full(N-1, 2), np.full(N-2, -1), np.full(N-2, -1)]
a = diags(diagonals, [0, -1, 1]).toarray()

eigenvalues, eigenvectors = np.linalg.eigh(a)


print(np.diag(eigenvalues))

N = 3
delta = np.zeros(N)

d = 2
a = -1

for j in range(1, N+1):
    delta[j-1] = d + 2*a*np.cos((j*np.pi)/(N+1)) #analytical eigenvalues
print(delta)


t1 = time.clock() - t0
print("Time elapsed: ", t1 - t0) # CPU seconds elapsed (floating point)
