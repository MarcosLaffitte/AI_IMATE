####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Final Project - Problem 2                                                                      #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                            #
#                                                                                                  #
# Problem 2. For the problem 1, it could be possible to develop a SVM to predict the test data.    #
#            Using a polynomial kernel, Would it be possible to give a good prediction of the      #
#            test data of problem 1? If not, give an explanation about it.                         #
#                                                                                                  #
#  - run (linux - ubuntu 20):   python [thisScript.py]                                             #
#                                                                                                  #
####################################################################################################
# code #############################################################################################

# dependencies info ################################################################################
"""
> tqdm 4.46.0

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
from sklearn.svm import SVR
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

# some stuff ------------------------------------------------------------------------------------
sns.set(rc={"figure.figsize":(11.7,8.27)})
warnings.simplefilter("ignore")

# variables ########################################################################################
# input --------------------------------------------------------------------------------------------
dataFile = "problem1.csv"

# data ---------------------------------------------------------------------------------------------
trainX = None
trainY = None
testX = None
testY = None

# models -------------------------------------------------------------------------------------------
maxDeg = 10   # can be increased to produce more models
totDegs = list(range(1, maxDeg + 1))

# data scaler --------------------------------------------------------------------------------------
inputScaler = None
outputScaler = None

# functions ########################################################################################
# function: get data -------------------------------------------------------------------------------
def getData():
    # function message
    print("- Obtaining data ...")
    # local variables
    theTrainX = None
    theTrainY = None
    theTestX = None
    theTestY = None
    someDataTrain = None
    someDataTest = None
    someDataTestVals = []
    someColumns = ["X_training", "Y_training", "X_test", "Y_test"]
    totDataX = None
    totDataY = None
    theInputScaler = MinMaxScaler()
    theOutputScaler = MinMaxScaler()
    # get training and test data
    theTrainX = pd.read_csv(dataFile, usecols = [someColumns[0]]).to_numpy()
    theTrainY = pd.read_csv(dataFile, usecols = [someColumns[1]]).to_numpy()
    theTestX = pd.read_csv(dataFile, usecols = [someColumns[2]]).dropna().to_numpy()
    theTestY = pd.read_csv(dataFile, usecols = [someColumns[3]]).dropna().to_numpy()
    # concatenate data in order to scale it
    totDataX = np.concatenate((theTrainX, theTestX))
    totDataY = np.concatenate((theTrainY, theTestY))
    # fit scalers
    theInputScaler.fit(totDataX)
    theOutputScaler.fit(totDataY)
    # scale data
    theTrainX = theInputScaler.transform(theTrainX)
    theTestX = theInputScaler.transform(theTestX)
    theTrainY = theOutputScaler.transform(theTrainY)
    theTestY = theOutputScaler.transform(theTestY)
    # end of function
    return(theTrainX, theTrainY, theTestX, theTestY, theInputScaler, theOutputScaler)

# function: make analysis over polynomial and svm --------------------------------------------------
def makeAnalysis():
    # function message
    print("- Training SVM and fitting polynomials ...")
    # local varibales
    someFontSize = 15
    someLearntPoly = None
    someLearntSVM = None
    somePredictPoly = None
    somePredictSVM = None
    modelPoly = None
    modelSVM = None
    someRSSTestPoly = 0
    someAICTestPoly = 0
    totAICTestPoly = []
    someRSSTrainPoly = 0
    someAICTrainPoly = 0
    totAICTrainPoly = []
    someRSSTestSVM = 0
    someAICTestSVM = 0
    totAICTestSVM = []
    someRSSTrainSVM = 0
    someAICTrainSVM = 0
    totAICTrainSVM = []
    minAICPoly = 0
    minAICSVM = 0
    bestDegreePoly = 0
    bestDegreeSVM = 0
    everyModelPoly = dict()
    everyModelSVM = dict()
    totDataX = None
    totDataY = None
    coefs = None
    nTrain = 0
    nTest = 0
    m = 0
    i = 0
    # train both models
    nTrain = len(trainX)
    nTest = len(testX)
    # get concatenated data
    totDataX = np.concatenate((trainX, testX))
    totDataY = np.concatenate((trainY, testY))
    # training loop
    for eachDeg in tqdm(totDegs):
        # get number of parameters for both models i.e degree
        m = eachDeg + 1
        # polynomial
        coefs = poly.polyfit(trainX.flatten(), trainY.flatten(), eachDeg)
        modelPoly = poly.Polynomial(coefs)
        someLearntPoly = modelPoly(trainX)
        somePredictPoly = modelPoly(testX)
        someRSSTrainPoly = (1/2) * np.sum((outputScaler.inverse_transform(trainY) - outputScaler.inverse_transform(someLearntPoly))**2)
        someRSSTestPoly = (1/2) * np.sum((outputScaler.inverse_transform(testY) - outputScaler.inverse_transform(somePredictPoly))**2)
        someAICTrainPoly = nTrain * math.log10(someRSSTrainPoly / nTrain) + (2 * m * nTrain / (nTrain - m - 1))
        someAICTestPoly = nTest * math.log10(someRSSTestPoly / nTest) + (2 * m * nTest / (nTest - m - 1))
        totAICTrainPoly.append(someAICTrainPoly)
        totAICTestPoly.append(someAICTestPoly)
        everyModelPoly[eachDeg] = modelPoly
        # SVM
        modelSVM = SVR(kernel = "poly", degree = eachDeg)
        modelSVM.fit(trainX, trainY)
        someLearntSVM = modelSVM.predict(trainX).reshape(-1, 1)
        somePredictSVM = modelSVM.predict(testX).reshape(-1, 1)
        someRSSTrainSVM = (1/2) * np.sum((outputScaler.inverse_transform(trainY) - outputScaler.inverse_transform(someLearntSVM))**2)
        someRSSTestSVM = (1/2) * np.sum((outputScaler.inverse_transform(testY) - outputScaler.inverse_transform(somePredictSVM))**2)
        someAICTrainSVM = nTrain * math.log10(someRSSTrainSVM / nTrain) + (2 * m * nTrain / (nTrain - m - 1))
        someAICTestSVM = nTest * math.log10(someRSSTestSVM / nTest) + (2 * m * nTest / (nTest - m - 1))
        totAICTrainSVM.append(someAICTrainSVM)
        totAICTestSVM.append(someAICTestSVM)
        everyModelSVM[eachDeg] = modelSVM
    # plot aic comparison
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.plot(totDegs, totAICTestPoly, marker = "o", color = "b", linewidth = 1, label = "Poly - Test Data")
    plt.plot(totDegs, totAICTestSVM, marker = "o", color = "r", linewidth = 1, label = "SVM - Test Data")
    plt.plot(totDegs, totAICTrainPoly, linestyle = "--", color = "b", marker = "o", linewidth = 1, label = "Poly - Training Data")
    plt.plot(totDegs, totAICTrainSVM, linestyle = "--", color = "r", marker = "o", linewidth = 1, label = "SVM - Training Data")
    # figure atributes
    plt.title("Problem 2 - AIC Comparison Between Models", fontsize = someFontSize)
    plt.xlabel("Degree", fontsize = someFontSize)
    plt.ylabel("Akaike Information Criterion (AIC)", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Problem_2_AIC_scaled.pdf")
    plt.close()
    # plot results svm
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[1].predict(totDataX).reshape(-1, 1)),
             color = "pink", linestyle = "--", label = "SVM - deg 1")
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[2].predict(totDataX).reshape(-1, 1)),
             color = "lightsalmon", linestyle = "--", label = "SVM - deg 2")
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[4].predict(totDataX).reshape(-1, 1)),
             color = "salmon", linestyle = "--", label = "SVM - deg 4")
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[6].predict(totDataX).reshape(-1, 1)),
             color = "darksalmon", linestyle = "--", label = "SVM - deg 6")
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[8].predict(totDataX).reshape(-1, 1)),
             color = "orangered", linestyle = "--", label = "SVM - deg 8")
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(everyModelSVM[10].predict(totDataX).reshape(-1, 1)),
             color = "r", linestyle = "--", label = "SVM - deg 10")
    plt.scatter(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(trainY), label = "Training Set")
    plt.scatter(inputScaler.inverse_transform(testX), outputScaler.inverse_transform(testY), label = "Test Set")
    # figure atributes
    plt.title("Problem 2 - SVM Results", fontsize = someFontSize)
    plt.xlabel("X values", fontsize = someFontSize)
    plt.ylabel("Y values", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Problem_2_SVM_scaled.pdf")
    plt.close()
    # end of function
    
# main #############################################################################################
print("\n")
print("------ Final Project - Problem 2 -----")

# preprocessing ------------------------------------------------------------------------------------
print("\n")
print("> Preprocessing")
# get data
trainX, trainY, testX, testY, inputScaler, outputScaler = getData()

# analysis -----------------------------------------------------------------------------------------
print("\n")
print("> Analysis")
# train models
makeAnalysis()

print("\n")
print("Finished Problem 2")
print("\n")
# end ##############################################################################################
####################################################################################################
