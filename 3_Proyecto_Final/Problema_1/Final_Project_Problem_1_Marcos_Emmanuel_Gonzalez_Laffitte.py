####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Final Project - Problem 1                                                                      #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                            #
#                                                                                                  #
# - PROBLEM 1 - using "problem 1.csv - training"                                                   #
#	A) Find the polynomial that fits the best the training data                                # 
#	B) Using the AIC criteria, find the best polynomial that can fit the data.                 #
#	C) Cross validate the polynomial with the data set called "problem1.csv - test"            #
#                                                                                                  #
#  - run (linux - ubuntu 20):   python [thisScript.py]                                             #
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
maxDeg = 15   # can be increased to produce more models
polyDegs = list(range(1, maxDeg + 1))

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

# function: visualize data -------------------------------------------------------------------------
def previewData():
    # function message
    print("- Visualize Data ...")
    # local variables
    someFontSize = 15
    # make scatter
    plt.scatter(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(trainY), label = "Training Data")
    plt.scatter(inputScaler.inverse_transform(testX), outputScaler.inverse_transform(testY), label = "Test Data")
    # figure atributes
    plt.title("Problem 1 - Preview Scatter of Polynomial", fontsize = someFontSize)
    plt.xlabel("X values", fontsize = someFontSize)
    plt.ylabel("Y values", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("PreviewOfData.pdf")
    plt.close()
    # end of function

# function: develope part A ------------------------------------------------------------------------
def partA():
    # function message
    print("- Training models and obtaining RSS over training data ...")
    # local varibales
    someFontSize = 15
    someLearnt = None
    coefs = None
    model = None
    somePolyRSS = 0
    someRSS = []
    subRSS = []
    subDegs = []
    minRSS = 1
    bestDegree = 0
    everyCoefSet = dict()
    i = 0
    # train each model
    for eachDeg in polyDegs:
        coefs = poly.polyfit(trainX.flatten(), trainY.flatten(), eachDeg)
        model = poly.Polynomial(coefs)
        someLearnt = model(trainX)
        somePolyRSS = (1/2) * np.sum((outputScaler.inverse_transform(trainY) - outputScaler.inverse_transform(someLearnt))**2)
        someRSS.append(somePolyRSS)
        everyCoefSet[eachDeg] = coefs
        if(somePolyRSS < minRSS):
            bestDegree = eachDeg
            minRSS = somePolyRSS
    # choose values to make plot
    for i in range(len(polyDegs)):
        if(someRSS[i] <= 1):
            subRSS.append(someRSS[i])
            subDegs.append(polyDegs[i])
    # make plot
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.plot(subDegs, subRSS, linestyle = "--", marker = "o", linewidth = 1)
    plt.plot([bestDegree], [minRSS], marker = "o", markersize = 10, color = "r", linestyle = "None",
             label = "Best degree: "
             + str(bestDegree)
             + "  - RSS Training: "
             + str(round(minRSS, 4)))
    # figure atributes
    plt.title(r"Problem 1 - Part A - RSS vs Degree of Polynomial  (for RSS $\leq$ 1)", fontsize = someFontSize)
    plt.xlabel("Degree", fontsize = someFontSize)
    plt.ylabel("Residual Sum of Squares (RSS)", fontsize = someFontSize)
    # annotation
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_A.pdf")
    plt.close()
    # remake best model
    model = poly.Polynomial(everyCoefSet[bestDegree])
    someLearnt = model(trainX)
    # plot best result
    plt.scatter(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(trainY), label = "Training Data")
    plt.plot(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(someLearnt), linestyle = "--", color = "r", linewidth = 2, label = "Learnt")
    # figure atributes
    plt.title("Problem 1 - Polynomial Learnt by Best Model based on RSS\n"
              + "Degree: "
              + str(bestDegree)
              + " - RSS Training: "
              + str(round(minRSS, 4)), fontsize = someFontSize)
    plt.xlabel("X values", fontsize = someFontSize)
    plt.ylabel("Y values", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_A_Results.pdf")
    plt.close()    
    # print best degree
    print("- Best degree based on RSS (training set): " + str(bestDegree) + " - RSS Training: " + str(round(minRSS, 4)))
    # print best coefficients
    print("- Best polynomial coefficients (w.r.t scaled data): ")
    print(everyCoefSet[bestDegree])
    # end of function

# function: develope part B ------------------------------------------------------------------------
def partB():
    # function message
    print("- Training models and obtaining AIC over training data ...")
    # local varibales
    someFontSize = 15
    someLearnt = None
    coefs = None
    model = None
    somePolyRSS = 0
    somePolyAIC = 0
    someAIC = []
    minAIC = 0
    bestDegree = 0
    everyCoefSet = dict()
    i = 0
    n = 0
    m = 0
    # train each model
    n = len(trainX)
    for eachDeg in polyDegs:
        coefs = poly.polyfit(trainX.flatten(), trainY.flatten(), eachDeg)
        model = poly.Polynomial(coefs)
        someLearnt = model(trainX)
        somePolyRSS = (1/2) * np.sum((outputScaler.inverse_transform(trainY) - outputScaler.inverse_transform(someLearnt))**2)
        m = len(coefs)
        somePolyAIC = n * math.log10(somePolyRSS / n) + (2 * m * n / (n - m - 1))
        someAIC.append(somePolyAIC)
        everyCoefSet[eachDeg] = coefs
    # choose best degree based on AIC
    minAIC = max(someAIC)
    for i in range(len(someAIC)):
        if(someAIC[i] < minAIC):
            minAIC = someAIC[i]
            bestDegree = polyDegs[i]
    # make plot
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.plot(polyDegs, someAIC, linestyle = "--", marker = "o", linewidth = 1)
    plt.plot([bestDegree], [minAIC], marker = "o", markersize = 10, color = "r", linestyle = "None",
             label = "Best degree: "
             + str(bestDegree)
             + "  - AIC Training: "
             + str(round(minAIC, 4)))
    # figure atributes
    plt.title(r"Problem 1 - Part B - AIC vs Degree of Polynomial", fontsize = someFontSize)
    plt.xlabel("Degree", fontsize = someFontSize)
    plt.ylabel("Akaike Information Criterion (AIC)", fontsize = someFontSize)
    # annotation
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_B.pdf")
    plt.close()
    # remake best model
    model = poly.Polynomial(everyCoefSet[bestDegree])
    someLearnt = model(trainX)
    # plot best result
    plt.scatter(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(trainY), label = "Training Data")
    plt.plot(inputScaler.inverse_transform(trainX), outputScaler.inverse_transform(someLearnt), linestyle = "--", color = "r", linewidth = 2, label = "Learnt")
    # figure atributes
    plt.title("Problem 1 - Polynomial Learnt by Best Model based on AIC\n"
              + "Degree: "
              + str(bestDegree)
              + " - AIC Training: "
              + str(round(minAIC, 4)), fontsize = someFontSize)
    plt.xlabel("X values", fontsize = someFontSize)
    plt.ylabel("Y values", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_B_Results.pdf")
    plt.close()    
    # print best degree
    print("- Best degree based on AIC (training set): " + str(bestDegree) + " - AIC Training: " + str(round(minAIC, 4)))
    # print best coefficients
    print("- Best polynomial coefficients (w.r.t scaled data): ")
    print(everyCoefSet[bestDegree])
    # end of function

# function: develope part C ------------------------------------------------------------------------
def partC():
    # function message
    print("- Training models and obtaining AIC over training and test sets ...")
    # local varibales
    someFontSize = 15
    someLearnt = None
    coefs = None
    model = None
    somePolyRSSTrain = 0
    somePolyAICTrain = 0
    somePolyRSSTest = 0
    somePolyAICTest = 0
    someAICTrain = []
    someAICTest = []
    minAIC = 0
    bestDegreeAIC = 0
    everyCoefSet = dict()
    i = 0
    nTrain = 0
    nTest = 0
    m = 0
    totDataX = None
    # train each model
    nTrain = len(trainX)
    nTest = len(testX)
    for eachDeg in polyDegs:
        coefs = poly.polyfit(trainX.flatten(), trainY.flatten(), eachDeg)
        model = poly.Polynomial(coefs)
        someLearnt = model(trainX)
        somePredict = model(testX)
        somePolyRSSTrain = (1/2) * np.sum((outputScaler.inverse_transform(trainY) - outputScaler.inverse_transform(someLearnt))**2)
        somePolyRSSTest = (1/2) * np.sum((outputScaler.inverse_transform(testY) - outputScaler.inverse_transform(somePredict))**2)
        m = len(coefs)
        somePolyAICTrain = nTrain * math.log10(somePolyRSSTrain / nTrain) + (2 * m * nTrain / (nTrain - m - 1))
        somePolyAICTest = nTest * math.log10(somePolyRSSTest / nTest) + (2 * m * nTest / (nTest - m - 1))
        someAICTrain.append(somePolyAICTrain)
        someAICTest.append(somePolyAICTest)
        everyCoefSet[eachDeg] = coefs
    # get best test aic
    minAIC = max(someAICTest)
    for i in range(len(someAICTest)):
        if(someAICTest[i] < minAIC):
            minAIC = someAICTest[i]
            bestDegreeAIC = polyDegs[i]
    # plot aic comparison
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.plot(polyDegs, someAICTrain, linestyle = "--", marker = "o", linewidth = 1, label = "Training Data")
    plt.plot(polyDegs, someAICTest, linestyle = "--", marker = "o", linewidth = 1, label = "Test Data")
    plt.axvline(bestDegreeAIC, ymin = 0, ymax = 1, color = "r", linestyle = "--", linewidth = 1, label = "Best Degree based\n on AIC Test")
    # figure atributes
    plt.title("Problem 1 - AIC Comparison Between Models\n"
              + "Best Model - Degree: "
              + str(bestDegreeAIC)
              + " - AIC Test: "
              + str(round(minAIC, 4)), fontsize = someFontSize)
    plt.xlabel("Degree", fontsize = someFontSize)
    plt.ylabel("Akaike Information Criterion (AIC)", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_C_AIC.pdf")
    plt.close()
    # plot aic results against real and rss
    totDataX = np.concatenate((trainX, testX))
    totDataY = np.concatenate((trainY, testY))
    # plot best result
    #bestDegreeAIC = 15 # test polynomial
    plt.scatter(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(totDataY), s = 15, label = "Training and Test Data")
    model = poly.Polynomial(everyCoefSet[bestDegreeAIC])
    someLearnt = model(totDataX)
    plt.plot(inputScaler.inverse_transform(totDataX), outputScaler.inverse_transform(someLearnt),
             color = "r", linestyle = "--", linewidth = 2, label = "AIC Test Data - Deg: " + str(bestDegreeAIC))
    # figure atributes
    plt.title("Problem 1 - Best Fit Polynomial \n", fontsize = someFontSize)
    plt.xlabel("X values", fontsize = someFontSize)
    plt.ylabel("Y values", fontsize = someFontSize)
    # legend
    plt.legend(fontsize = someFontSize)
    # savefigure
    plt.savefig("Part_C_Results.pdf")
    plt.close()
    # print best degree
    print("- Best degree based on AIC (test set): " + str(bestDegreeAIC) + " - AIC Test: " + str(round(minAIC, 4)))
    # print best coefficients
    print("- Best polynomial coefficients (w.r.t scaled data): ")
    print(everyCoefSet[bestDegreeAIC])
    # end of function
    
# main #############################################################################################
print("\n")
print("------ Final Project - Problem 1 -----")

# preprocessing ------------------------------------------------------------------------------------
print("\n")
print("> Preprocessing")
# get data
trainX, trainY, testX, testY, inputScaler, outputScaler = getData()
# visualize data
previewData()

# analysis -----------------------------------------------------------------------------------------
print("\n")
print("> Analysis")

# A) Find the polynomial that fits the best the training data
print("\n")
print("A) Find the polynomial that fits the best the training data")
# train models
partA()

# B) Using the AIC criteria, find the best polynomial that can fit the data.
print("\n")
print("B) Using the AIC criteria, find the best polynomial that can fit the data")
# train models
partB()

# C) Cross validate the polynomial with the test data set
print("\n")
print("C) Cross validate the polynomial with the test data set")
# train models
partC()

print("\n")
print("Finished Problem 1")
print("\n")
# end ##############################################################################################
####################################################################################################
