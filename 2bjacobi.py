import numpy as np
from scipy.sparse import diags
eps = 10**-8
N = 4 # N-1 is matrix dimensions

import time
t0= time.clock()

def make_matrix(d, d2):
    diagonals = [np.full(N-1, d), np.full(N-2, d2), np.full(N-2, d2)]
    a = diags(diagonals, [0, -1, 1]).toarray()
    return a

rho_min = 0; rho_max = 1
h = (rho_max - rho_min)/N

a = make_matrix(2/(h**2), -1/(h**2))
print(a)

def find_max(a):
    k = 0; l = 0
    max_value = 0
    for i in range(N-1):
        for j in range(N-1):
            if i == j:
                continue
            else:
                if np.abs(a[i][j]) > max_value:
                    max_value = np.abs(a[i][j])
                    k = i; l = j

    return k, l
k, l = find_max(a)


def jacobi_rotate(a, k, l):
    t = 0
    t_list = np.zeros(2)

    τ = (a[l][l] - a[k][k])/(2*a[k][l])
    t_list[0] = -τ + np.sqrt(1 + τ**2)
    t_list[1] = -τ - np.sqrt(1 + τ**2)


    t_minabs = min(np.abs(t_list))  #choosing the smallest of these roots
    if t_minabs == np.abs(t_list[0]):
        t = t_list[0]
    else:
        t = t_list[1]

    c = 1/(np.sqrt(1+t**2))
    s = t*c
    b = np.copy(a)

    for i in range(N-1):

        b[i][k] = a[i][k]*c - a[i][l]*s
        b[i][l] = a[i][l]*c + a[i][k]*s
        b[k][i] = b[i][k]
        b[l][i] = b[i][l]
    b[k][k] = a[k][k]*c**2 - 2*a[k][l]*c*s + a[l][l]*s**2
    b[l][l] = a[l][l]*c**2 + 2*a[k][l]*c*s + a[k][k]*s**2
    b[k][l] = 0
    b[l][k] = 0
    if  abs((a[k][k] - a[l][l])*c*s + a[k][l]*(c**2 - s**2))**2 > eps:
        raise ValueError()
    return b

count = 0
max_value = 1
while abs(max_value) > eps:
    k, l = find_max(a)
    a = jacobi_rotate(a, k, l)
    k, l = find_max(a)
    max_value = a[k][l]
    count += 1
print(a)
print('Similarity transformations =', count)



t1 = time.clock() - t0
print("Time elapsed: ", t1 - t0) # CPU seconds elapsed (floating point)










#
