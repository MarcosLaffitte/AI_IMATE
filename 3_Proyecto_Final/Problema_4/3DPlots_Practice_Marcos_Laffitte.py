####################################################################################################
#                                                                                                  #
# - Test of 3d plots                                                                               #
#                                                                                                  #
# - ref https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html  #
#                                                                                                  #
####################################################################################################
# code #############################################################################################

# dependencies info ################################################################################
"""
> scipy 1.4.1
> seaborn 0.10.0
> numpy 1.18.1
> pandas 1.0.1
> matplotlib 3.1.3
> scikit-learn 0.22.1
> keras 2.3.1
> tensorflow 2.2.0
>> anaconda 2020.02
>>> python 3.7
"""

# dependencies #####################################################################################
# already in python --------------------------------------------------------------------------------
import math
import warnings
from copy import deepcopy

# not in python ------------------------------------------------------------------------------------
from tqdm import tqdm
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

# 3d plots -----------------------------------------------------------------------------------------
from mpl_toolkits import mplot3d

# some stuff ---------------------------------------------------------------------------------------
sns.set(rc={"figure.figsize":(11.7,8.27)})
warnings.simplefilter("ignore")

# main #############################################################################################

# Data for a three-dimensional line
fig = plt.figure()
ax = plt.axes(projection = "3d")
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, "gray")
plt.show()
plt.close()

# Data for three-dimensional scattered points
fig = plt.figure()
ax = plt.axes(projection = "3d")
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c = zdata, cmap = "Greens");
plt.show()
plt.close()

# Contour surface
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(60, 35)
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.contour3D(X, Y, Z, 50, cmap = "binary")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
plt.close()

# Wire frames
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(60, 35)
ax.plot_wireframe(X, Y, Z, color = "black")
ax.set_title("wireframe")
plt.show()
plt.close()

# Filled wire frames
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(60, 35)
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = "viridis", edgecolor = "none")
ax.set_title("surface");
plt.show()
plt.close()

# Scatter and their Triangulation
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(60, 35)
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
ax.scatter(x, y, z, c = z, cmap = "viridis", linewidth = 0.5)
plt.show()
plt.close()
#---
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(60, 35)
ax.plot_trisurf(x, y, z, cmap = "viridis", edgecolor = "none")
plt.show()
plt.close()

# end ##############################################################################################
####################################################################################################











