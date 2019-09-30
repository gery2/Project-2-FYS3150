import numpy as np
from scipy.sparse import diags
eps = 10**-8
N = 4 # N-1 is matrix dimensions

def make_matrix(d, d2):
    diagonals = [np.full(N-1, d), np.full(N-2, d2), np.full(N-2, d2)]
    a = diags(diagonals, [0, -1, 1]).toarray()
    return a

rho_min = 0; rho_max = 1
h = (rho_max - rho_min)/N

a = make_matrix(2, -1)
#a = make_matrix(2/(h**2), -1/(h**2))

a1 = a[0][0]; b1 = a[0][1]

lam_ran_1 = [(np.abs(a1) - 2*np.abs(b1)), (np.abs(a1) + 2*np.abs(b1))]
print(lam_ran_1)


def bisection(f,a,b,N):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisection(f,1,2,25)
    1.618033990263939
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisection(f,0,1,10)
    0.5
    '''
    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2


P = np.zeros(N)

P[0] = 1
p = lambda x: (a1 - x)
P[1] = bisection(p, 0, 4, N)

p = lambda x: (a1 - x)*P[1] - b1**2*P[0]
P[2] = bisection(p, 0, P[1], N)

p = lambda x: (a1 - x)*P[2] - b1**2*P[1]
P[3] = bisection(p, 0, P[2], N)
print(a)
print(P)
