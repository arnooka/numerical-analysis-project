import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from piecewise_cubic import cubic_interpolation
from piecewise_quadratic import quadratic_interpolation
from nearest_neighbour import nearest_neighbour
from piecewise_linear import linear_interpolation
from error import *


path = "/Users/pedroalbuquerque/Dropbox/Faculdade/UNL/numerical_analysis/final_proj/data/appended/"
points_train = 50
points_to_jump = 800
points_test = 400

plt.rc('font', family='Helvetica')
sns.set()

'''
Plots chart with the interpolations
'''
def plot_interpolation(x_train, y_train, x_test, y_test, f, f2, f3, f4, xnew=None):
    dmin = np.min(x_train)
    dmax = np.max(x_train)
    if xnew is None: xnew = np.linspace(dmin, dmax, num=100000, endpoint=True)
    plt.plot(xnew, f(xnew), '-', xnew, f2(xnew), '-', xnew, f3(xnew), '-', xnew, f4(xnew), '-',  x_train, y_train, 'ko', x_test, y_test, 'rx', markersize=3.5)
    plt.legend(['Nearest neighbor', 'Linear', 'Quadratic', 'Cubic', 'Train', 'Test'], loc='best')
    plt.show()


# ========================
# Preparing the data
# ========================

z_timestep = np.loadtxt(path + "z_timestep.csv", dtype=object)  # load file
z_timestep = z_timestep[1:].astype(np.float64)  # convert to number keeping precision
#idxs = np.array(range(0,len(z_timestep), int(len(z_timestep) / points_train) ))  # Drop experiences in interval
idxs = np.array(range(0,len(z_timestep), points_to_jump ))  # Drop experiences in interval
x = z_timestep[idxs]  # Filter out experiences

# Creating the array with the test data
idxs_ignored = np.zeros(len(z_timestep), dtype=bool)
idxs_ignored [idxs] = True
idxs_ignored [:idxs[0]] = True
idxs_ignored [idxs[-1]:] = True

idxs_ignored = np.where(~idxs_ignored)[0]
test_mask = [ idxs_ignored[i] for i in range(0, len(idxs_ignored), int( len(idxs_ignored) / points_test ) ) ]

z_test = z_timestep[test_mask]


z = np.loadtxt(path + "z.csv", dtype=object)[1:]
z = z.astype(np.float64)
y = z[idxs]

# ====================================
# Creating the interpolation functions
# ====================================

f_c = cubic_interpolation(x, y)
#f_q = interp1d(x, y, kind='quadratic')
f_q = quadratic_interpolation(x, y)
f_n = nearest_neighbour(x, y)
f_l = linear_interpolation(x, y)

# ========================
# Performs the evaluation
# ========================
x_true = z_timestep[test_mask]
y_true = z[test_mask]
y_pred = [ f(x_true) for f in [f_n, f_l, f_q, f_c] ]
y_error = [ np.abs(np.subtract(yp, y_true)) for yp in y_pred]

techniques = ["Nearest Neighbor", "Linear Interpolation", "Quadratic Interpolation", "Cubic Interpolation"]
for i in range(len(y_pred)):
    print ("-----------------------------------------")
    print ("Error analysis - {}".format(techniques[i]))
    print ("RMS: {}".format( rms(y_pred[i], y_true) ) )
    print ("abs: {}".format( abs_error(y_pred[i], y_true ) ))
    print ("std: {}".format( std(y_pred[i], y_true) ))
    print ("var: {}".format( var(y_pred[i], y_true) ))
    print ("max: {}".format( max(y_pred[i], y_true) ))
    print ("min: {}".format( min(y_pred[i], y_true) ))


# Plot the sampling, testing points and the methods response
#plot_interpolation(x, y, x_true, y_true, f_n, f_l, f_q, f_c )

# Bar plot with RMS
#sns.set(style="whitegrid")
#ax = sns.barplot(x=["Nearest neighbor", "Linear", "Quadratic", "Cubic"], y=[rms(yp, y_true) for yp in y_pred])
#ax = sns.barplot(x=["Linear", "Quadratic", "Cubic"], y=[rms(yp, y_true) for yp in y_pred[1:]])
#ax.set(ylabel='Root Mean Squared Error')

df = pd.DataFrame(np.stack(y_error, axis = 1), columns=["Nearest Neighbor", "Linear", "Quadratic", "Cubic"])
ax = sns.boxplot(data=df, showfliers=False)  # Boxplot

#sns.savefig("/Users/pedroalbuquerque/Dropbox/Faculdade/UNL/numerical_analysis/final_proj/data/graphs/{}\_box\.png".format(points_to_jump) )
#ax.set( ylabel='Absolute Error')
#ax = sns.lineplot(data=df)  # Lines
#ax.set(xlabel='Inference Point', ylabel='Absolute Error')
plt.show()

