import numpy as np
import sys

'''
Returns a functions that calculates the interpolation
'''
def create_splitwise_function(t, y, z):
    def spline_function (xq):
        if isinstance (xq, (list, np.ndarray)): return list(map(spline_function, xq))  # If it is a list, iterate the values in this function
        assert  (t[0] <= xq <= t[-1]), "Query outside boundaries of the data!"
        i = len(t) - 2  # if loop does not find, it is the last equation
        for idx in range(1, len(t)):
            if t[idx] > xq:
                i = idx - 1
                break
        return  ( z[i+1] - z[i] ) / ( 2 * (t[i+1] - t[i]) ) * ( xq - t[i] )**2 + z[i] * (xq - t[i]) + y[i]
    return spline_function

def quadratic_interpolation(t, y):
    # Calculating the all the values of z
    z = [0]
    for i in range(len(t) - 1):
        z.append( -z[i] + 2 * (y[i+1] - y[i]) / (t[i+1] - t[i]) )

    return create_splitwise_function(t, y, z)
