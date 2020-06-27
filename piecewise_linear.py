import numpy as np
import sys

def linear_interpolation(x, y):
    def function(xq):
        if isinstance (xq, (list, np.ndarray)): return list(map(function, xq))
        assert  (x[0] <= xq <= x[-1]), "Query outside boundaries of the data!"
        i = len(x) - 2
        for idx in range(1, len(x)):
            if x[idx] > xq:
                i = idx - 1
                break
        return y[i] + ( (y[i+1]-y[i]) / (x[i+1]-x[i]) ) * (xq - x[i])
    return function
