import numpy as np

def nearest_neighbour(x,y):
    def nearest_function(xq):
        if isinstance (xq, (list, np.ndarray)): return list(map(nearest_function, xq))
        assert  (x[0] <= xq <= x[-1]), "Query outside boundaries of the data!"
        i = len(x) - 1  # if loop does not find, it is the last
        for idx in range(1, len(x)):
            if x[idx] > xq:
                i = idx - 1 if abs(xq - x[idx]) > abs(xq - x[idx-1]) else idx
                break
        return y[i]

    return nearest_function
