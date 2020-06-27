import numpy as np
import sys

def rms(pred, real):
    return np.sqrt( np.mean( np.square( np.subtract(pred, real) ) ) )
	
def abs_error(pred, real):
	return np.mean( np.abs( np.subtract(pred, real) ) )
	
def std(pred, real):
	return np.std( np.abs( np.subtract(pred, real) ) )

def var(pred, real):
	return np.square( std(pred, real) )
	
def max(pred, real):
	return np.max( np.abs( np.subtract(pred, real) ) )

def min(pred, real):
	return np.min( np.abs( np.subtract(pred, real) ) )
