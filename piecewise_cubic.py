from sympy import *
import numpy as np

def create_spline_function(x, a, b, c, d):
    def spline_function (xq):
        if isinstance (xq, (list, np.ndarray)): return list(map(spline_function, xq))
        assert  (x[0] <= xq <= x[-1]), "Query outside boundaries of the data!"
        i = len(x) - 2
        for idx in range(1, len(x)):
            if x[idx] > xq:
                i = idx - 1
                break
        xi = x[i]
        return a[i] + b[i]*(xq - xi) + c[i]*(xq - xi)**2 + d[i]*(xq - xi)**3

    return spline_function

def cubic_interpolation(x, a):
    h = [x[i+1] - x[i] for i in range(len(x) - 1) ]
    A = np.identity(len(x))
    B = np.zeros(len(x))

    for i in range(1, len(x) - 1):
        A[i, i-1] = h[i-1]
        A[i, i] =  2 * ( h[i] + h[i-1] )
        A[i, i+1] = h[i]
        B[i] =  3/h[i] * ( a[i+1] - a[i] ) - 3/h[i-1] * ( a[i] - a[i-1] )

    c = np.linalg.solve(A, B)
    b = [  1/h[i] * ( a[i+1] - a[i] ) - h[i]/3 * ( c[i+1] + 2*c[i] ) for i in range(len(x) - 1) ]
    d = [  1/(3 * h[i]) * ( c[i+1] - c[i] ) for i in range(len(x) - 1) ]

    return create_spline_function(x, a, b, c, d)
