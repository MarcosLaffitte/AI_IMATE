####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Final Project - Problem 3                                                                      #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                            #
#                                                                                                  #
# - Description: determine a measure to be used as marker to describe the progression of HIV from  #
#                a set of data obtained from 12 patients.                                          #
#                                                                                                  #
# - methods: 2 logistic regressions, knn and ensemble of these methods                             #
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

# some stuff ------------------------------------------------------------------------------------
sns.set(rc={"figure.figsize":(11.7,8.27)})
warnings.simplefilter("ignore")

# variables ########################################################################################
# data ---------------------------------------------------------------------------------------------
fileName = "problem3.csv"
theAge = None   # age
theCho = None   # cholesterol
theSug = None   # sugar
theTCe = None   # tcell
theOut = None   # hiv progression id
theSTP = None   # sugar and tcell pairs

# control ------------------------------------------------------------------------------------------
maxFold = 50
trainPercentage = 0.6
testPercentage = 1 - trainPercentage
i = 0
j = 0
rounds = 0
theFontSize = 15

# analysis -----------------------------------------------------------------------------------------
xTrainSug = None
xTestSug = None
#---
xTrainTCe = None
xTestTCe = None
#---
xTrainSTP = None
xTestSTP = None
#---
yTrain = None
yTest = None

# models -------------------------------------------------------------------------------------------
modelSug = None
modelTCe = None
modelSTP = None

# predictions --------------------------------------------------------------------------------------
predSug = None
predTCe = None
predSTP = None
predEnsemble = None

# score --------------------------------------------------------------------------------------------
fSug = 0
fTCe = 0
fSTP = 0
fEnsemble = 0
fSugTot = []
fTCeTot = []
fSTPTot = []
fEnsembleTot = []
kFolds = []

# functions ########################################################################################
# function: get data -------------------------------------------------------------------------------
def getData():
    # function message
    print("- Obtaining data ...")
    # local variables
    inputLen = 12
    someAge = None
    someCho = None
    someSug = None
    someTCe = None
    someOut = None
    # load file
    someAge = pd.read_csv(fileName, sep = ",", usecols = list(range(1, inputLen + 1)), nrows = 1, skiprows = 0, header = None, dtype = "int").stack().to_numpy().reshape((-1, 1))
    someCho = pd.read_csv(fileName, sep = ",", usecols = list(range(1, inputLen + 1)), nrows = 1, skiprows = 1, header = None, dtype = "int").stack().to_numpy().reshape((-1, 1))
    someSug = pd.read_csv(fileName, sep = ",", usecols = list(range(1, inputLen + 1)), nrows = 1, skiprows = 2, header = None, dtype = "int").stack().to_numpy().reshape((-1, 1))
    someTCe = pd.read_csv(fileName, sep = ",", usecols = list(range(1, inputLen + 1)), nrows = 1, skiprows = 3, header = None, dtype = "int").stack().to_numpy().reshape((-1, 1))
    someOut = pd.read_csv(fileName, sep = ",", usecols = list(range(1, inputLen + 1)), nrows = 1, skiprows = 4, header = None, dtype = "int").stack().to_numpy().reshape((-1, 1))
    # end of function
    return(someAge, someCho, someSug, someTCe, someOut)

# function: correlation matrix ---------------------------------------------------------------------
def plotCorrelationMatrix():
    # function message
    print("- Plotting correlation matrix ...")
    # local variables
    someMatrix = None
    someLabels = ["Age", "Cholesterol", "Sugar", "TCell", "Target"]
    # make complete correlation matrix
    someMatrix = np.corrcoef([theAge.flatten(), theCho.flatten(), theSug.flatten(), theTCe.flatten(), theOut.flatten()])
    sns.heatmap(data = someMatrix, annot = True, cmap = "RdYlBu_r", linewidths = 0.5, xticklabels = someLabels, yticklabels = someLabels)
    plt.savefig("Probelm_3_Matrix.pdf")
    plt.close()
    # end of function

# function: build pairs of sugar and tcell ---------------------------------------------------------
def pairSugarAndTCell():
    # function message
    print("- Pairing sugar and tcell data ...")
    # local variables
    someFontSize = 15
    someSTP = None
    # make pairs for each patient
    someSTP = np.concatenate((theSug, theTCe), axis = 1)
    # make scatter plot
    plt.scatter(theSug[0:6], theTCe[0:6], s = 400, label = "No progression")
    plt.scatter(theSug[6:], theTCe[6:], s = 400, label = "Progression")
    # figure atributes
    plt.title("Problem 3 - Scatter Of Sugar and TCell Count for HIV progression data", fontsize = someFontSize)
    plt.xlabel("Sugar", fontsize = someFontSize)
    plt.ylabel("TCell", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Problem_3_Scatter.pdf")
    plt.close()    
    # end of function
    return(someSTP)
    
# main #############################################################################################
print("\n")
print("------ Final Project - Problem 3 -----")

# preprocessing ------------------------------------------------------------------------------------
print("\n")
print("> Preprocessing")
# get data
theAge, theCho, theSug, theTCe, theOut = getData()
# make correlation matrix
plotCorrelationMatrix()
# build pairs of Sugar and TCell
theSTP = pairSugarAndTCell()

# analysis -----------------------------------------------------------------------------------------
print("\n")
print("> Developing multiple cross validation over logistic regressions, knn and ensemble ...")
for i in tqdm(range(maxFold)):
    # reinitialize f-score
    fSug = 0
    fTCe = 0
    fSTP = 0
    fEnsemble = 0
    rounds = 0
    for j in range(i + 1):
        # split data
        xTrainSTP, xTestSTP, yTrain, yTest = train_test_split(theSTP, theOut, test_size = testPercentage, shuffle = True)
        xTrainSug = xTrainSTP[:,[0]]
        xTestSug = xTestSTP[:,[0]]
        xTrainTCe = xTrainSTP[:,[1]]
        xTestTCe = xTestSTP[:,[1]]
        # logistic regression sugar
        modelSug = LogisticRegression()
        modelSug.fit(xTrainSug, yTrain)
        predSug = modelSug.predict(xTestSug)
        fSug = fSug + f1_score(yTest, predSug)
        # logistic regression tcells
        modelTCe = LogisticRegression()
        modelTCe.fit(xTrainTCe, yTrain)
        predTCe = modelTCe.predict(xTestTCe)
        fTCe = fTCe + f1_score(yTest, predTCe)
        # knn with sugar and tcell pairs
        modelSTP = KNeighborsClassifier(n_neighbors = 4)
        modelSTP.fit(xTrainSTP, yTrain)
        predSTP = modelSTP.predict(xTestSTP)
        fSTP = fSTP + f1_score(yTest, predSTP)
        # ensemble
        predEnsemble = predSug + predTCe + predSTP
        predEnsemble = (1/3) * predEnsemble
        predEnsemble = np.rint(predEnsemble).astype(int)
        fEnsemble = fEnsemble + f1_score(yTest, predEnsemble)
        # count round
        rounds = rounds + 1
    # get average scores
    fSug = fSug / rounds
    fTCe = fTCe / rounds
    fSTP = fSTP / rounds
    fEnsemble = fEnsemble / rounds
    # save values
    fSugTot.append(fSug)
    fTCeTot.append(fTCe)
    fSTPTot.append(fSTP)
    fEnsembleTot.append(fEnsemble)

# plot results -------------------------------------------------------------------------------------
print("\n")
print("> Plotting cross validation results ...")
# make data
kFolds = list(range(1, maxFold + 1))
# axis
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
# plot lines
plt.plot(kFolds, fSugTot, linestyle = "--", linewidth = 0.9, marker = "o", label = "Logistic Regression Sugar Data")
plt.plot(kFolds, fTCeTot, linestyle = "--", linewidth = 0.9, marker = "o", label = "Logistic Regression TCell Data")
plt.plot(kFolds, fSTPTot, linestyle = "--", linewidth = 0.9, marker = "o", label = "KNN of Sugar and TCell Pairs")
plt.plot(kFolds, fEnsembleTot, linestyle = "--", linewidth = 0.9, marker = "o", label = "Ensemble of the 3 methods")
# fix axis
plt.ylim((0, 1.05))
plt.xlim((0.25, maxFold + 0.75)) 
# figure atributes
plt.title("Problem 3 - F-measure for K-fold cross validations", fontsize = theFontSize)
plt.xlabel("K for K-fold cross validation", fontsize = theFontSize)
plt.ylabel("F-measure", fontsize = theFontSize)
# legend
plt.legend(fontsize = theFontSize)
# savefigure
plt.savefig("Problem_3_FM.pdf")
plt.close()

print("\n")
print("Finished Problem 3")
print("\n")
# end ##############################################################################################
####################################################################################################









