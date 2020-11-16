####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Final Project - Problem 4                                                                      #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                            #
#                                                                                                  #
# - Problem 4: Using a multilayer perceptron with Keras produce a cone with the dimension          #
#              of your election.                                                                   #
#                                                                                                  #
# - Ecuación cartesiana de un cono centrado en el origen:                                          #
#                                                                                                  #
#                             x^2 + y^2 = (R/h)^2 * z^2                                            #
#                                                                                                  #
#                     para el máximo radio R a la máxima altura h                                  #
#                                                                                                  #
# - Cono escogido:     R = 0.5, h = 1,    con    -0.5 <= x, y <= 0.5,    0 <= z <= 1               #
#                                                                                                  #
# - run (linux - ubuntu 20):   python [thisScript.py]                                              #
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
from itertools import product

# not in python ------------------------------------------------------------------------------------
from tqdm import tqdm
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

# neural network -----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 3d plots -----------------------------------------------------------------------------------------
from mpl_toolkits import mplot3d

# some stuff ------------------------------------------------------------------------------------
import os
sns.set(rc={"figure.figsize":(11.7,8.27)})
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
# main #############################################################################################
print("\n")
print("------ Final Project - Problem 4 -----")
print("\n")

# plot cone ----------------------------------------------------------------------------------------
print("> Building original cone ...")

# cone equation
def f(x, y):
    return(2*(np.sqrt(x**2 + y**2)))

# 3d plot parameters and variables
x = []
y = []
z = []
norm = 0

# get x, y points
xRaw = np.linspace(-0.5, 0.5, 71)
yRaw = np.linspace(-0.5, 0.5, 71)
pairs = list(product(xRaw, yRaw))
for (x0, y0) in pairs:
    norm = np.sqrt(x0 ** 2 + y0 ** 2)
    if(norm <= 0.5):
        x.append(x0)
        y.append(y0)
        z.append(f(x0, y0))
        
print("> Plotting original cone ...")
        
# plot scatter data of real cone
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(25, 35)
ax.scatter(x, y, z, c = z, cmap = "viridis", linewidth = 0.5)
ax.set_title("Original Cone Scatter Data");
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.savefig("Problem_4_Original_Cone_Scatter_Data.pdf")
plt.show()
plt.close()

# plot scatter data of real cone
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(25, 35)
ax.plot_trisurf(x, y, z, cmap = "viridis", edgecolor = "none")
ax.set_title("Original Triangulated Cone");
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.savefig("Problem_4_Original_Triangulated_Cone.pdf")
plt.show()
plt.close()

# train neural network -----------------------------------------------------------------------------
# make trainign and test pairs
dataX = np.array(list(zip(x, y)))
dataY = np.array(z).reshape(-1, 1)

# obtain training set and test set
trainPercent = 0.6
testPercentage = 1 - trainPercent
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = testPercentage, shuffle = True)

print("> Building neural network ...")

# build model
model = Sequential()
model.add(Dense(units = 50, input_dim = 2, activation = "sigmoid"))
model.add(Dense(units = 50, activation = "sigmoid"))
model.add(Dense(units = 50, activation = "sigmoid"))
model.add(Dense(units = 1, activation = "linear"))
model.compile(optimizer = "adam", loss = "MeanSquaredError")

print("> Training network ...")

# train network
model.fit(trainX, trainY, epochs = 100, batch_size = 6, verbose = 1)

print("> Obtaining learnt cone ...")

# get learnt cone
learntCone = model.predict(dataX)
zL = learntCone.flatten()

# stop keras session
tf.keras.backend.clear_session()

print("> Plotting results ...")

# plot scatter data of real cone
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(25, 35)
ax.scatter(x, y, zL, c = z, cmap = "viridis", linewidth = 0.5)
ax.set_title("Learnt Cone Scatter Data");
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.savefig("Problem_4_Learnt_Cone_Scatter_Data.pdf")
plt.show()
plt.close()

# plot scatter data of real cone
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.view_init(25, 35)
ax.plot_trisurf(x, y, zL, cmap = "viridis", edgecolor = "none")
ax.set_title("Learnt Triangulated Cone");
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.savefig("Problem_4_Learnt_Triangulated_Cone.pdf")
plt.show()
plt.close()

print("\n")
print("Finished Problem 4")
print("\n")
# end ##############################################################################################
####################################################################################################









