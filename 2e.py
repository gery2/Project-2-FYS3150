import numpy as np
from scipy.sparse import diags
eps = 10**-8
N = 21 # N-1 is matrix dimensions


rho_min = 0; rho_max = 5
h = (rho_max - rho_min)/N
omega = 0.01

rho = np.zeros(N); d = np.zeros(N)
a = np.zeros((N, N))
e = -1/(h**2)
for i in range(N):
    rho[i] = (i+1)*h
    d[i] = 2/(h**2) + omega**2*(rho[i])**2 + 1/rho[i]

for i in range(1, N-1):
    a[i][i] = d[i]
    a[i][i+1] = e
    a[i][i-1] = e

a[0][0] = d[0]
a[0][1] = e
a[-1][-1] = d[-1]
a[-1][-2] = e


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


def jacobi_rotate(a, k, l, R):

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
        if k != i and l != i:
            b[i][k] = a[i][k]*c - a[i][l]*s
            b[i][l] = a[i][l]*c + a[i][k]*s
            b[k][i] = b[i][k]
            b[l][i] = b[i][l]

        r_ik = R[i][k];
        r_il = R[i][l];
        R[i][k] = c*r_ik - s*r_il;
        R[i][l] = c*r_il + s*r_ik;


    b[k][k] = a[k][k]*c**2 - 2*a[k][l]*c*s + a[l][l]*s**2
    b[l][l] = a[l][l]*c**2 + 2*a[k][l]*c*s + a[k][k]*s**2
    b[k][l] = 0
    b[l][k] = 0
    if  abs((a[k][k] - a[l][l])*c*s + a[k][l]*(c**2 - s**2))**2 > eps:
        raise ValueError()

    return b, R
R = np.eye(N)
count = 0
max_value = 1
while abs(max_value) > eps:
    k, l = find_max(a)
    a, R = jacobi_rotate(a, k, l, R)
    k, l = find_max(a)
    max_value = a[k][l]
    count += 1

vector = np.diag(a)
indexes = np.argsort(vector)
vector = vector[indexes]
R = R[:,indexes]
print('Similarity transformations =', count)
print('R0 = ',R[0])

def Average(R):
    return sum(R) / len(R)
average = Average(R[0])
print('Ground state average = ', average)





#
