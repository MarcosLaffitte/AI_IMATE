####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Proj1: Boston House Dataset Analysis                                                           #
#   http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html                                 #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                             #
#                                                                                                  #
# - Description: predict price of houses in data set, using a number of variables determined       #
#                by the abs value for pearson correlation coef give by the user (default 0.7).     #
#                                                                                                  #
#  - run (linux):   python3.5 thisScript.py                                                        #
#                                                                                                  #
####################################################################################################
# code #############################################################################################

# dependencies info ################################################################################
"""
> scipy
  Version: 1.1.0
  Summary: SciPy: Scientific Library for Python
  Home-page: https://www.scipy.org

> seaborn
  Version: 0.9.0
  Summary: seaborn: statistical data visualization
  Home-page: https://seaborn.pydata.org

> numpy
  Version: 1.17.0
  Summary: NumPy is the fundamental package for array computing with Python.
  Home-page: https://www.numpy.org

> pandas
  Version: 0.25.1
  Summary: Powerful data structures for data analysis, time series, and statistics
  Home-page: http://pandas.pydata.org

> matplotlib
  Version: 3.0.2
  Summary: Python plotting package
  Home-page: http://matplotlib.org

> scikit-learn
  Version: 0.20.0
  Summary: A set of python modules for machine learning and data mining
  Home-page: http://scikit-learn.org
"""

# dependencies #####################################################################################
# already in python --------------------------------------------------------------------------------
import math
import warnings
from copy import deepcopy

# not in python ------------------------------------------------------------------------------------
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# some stuff ------------------------------------------------------------------------------------
sns.set(rc={"figure.figsize":(11.7,8.27)})
warnings.simplefilter("ignore")
plt.switch_backend("agg")

# variables ########################################################################################
# input --------------------------------------------------------------------------------------------
corrCoefAbs = 0.7

# data ---------------------------------------------------------------------------------------------
theData = None
bestVariables = None
models = []
RSSvTrainTot = dict()
RSSvTestTot = dict()
AICvTrainTot = dict()
AICvTestTot = dict()
rssTr = 0
rssTe = 0
aicTr = 0
aicTe = 0
fitCoefs = []
modelResults = ()
modelEvaluation = []
theBestModelEver = ()

# control ------------------------------------------------------------------------------------------
corrCoefAbs = abs(corrCoefAbs)
if(corrCoefAbs > 1):
    corrCoefAbs = 0.7

# constant -----------------------------------------------------------------------------------------
myTarget = "MEDV"
trainSetPercent = (0.60, 0.65, 0.70, 0.75, 0.80)

# functions ########################################################################################
# function: get data from scikit and append target MEDV --------------------------------------------
def getData():
    # function message
    print("\n")
    print("- Obtaining data set ...")
    # local variables
    dataSet = None
    # get dataset
    dataSet = load_boston()
    dataSetDF = pd.DataFrame(dataSet.data, columns = dataSet.feature_names)
    dataSetDF["MEDV"] = dataSet.target
    # end of function
    return(dataSetDF)

# function: isnull and visual of data rows ---------------------------------------------------------
def makeVisualDataRows(someDF):
    # local variables
    headNum = 5
    nullArr = None
    headArr = None
    # call for isnull info
    print("\n")
    print("- Isnull check ...")
    nullArr = someDF.isnull().sum()
    print(nullArr)
    # call for head info
    print("\n")
    print("- Dataset check ...")
    headArr = someDF.head(headNum)
    print(headArr)
    # end of function

# function: visual of  MEDV distrubition -----------------------------------------------------------
def plotDistributionTarget(theDF, theTarget, theLabel):
    # function message
    print("\n")
    print("- Plotting distribution of target variable ...")
    # local variables
    binsNum = 30
    avTarget = 0
    stdTarget = 0
    # get average target
    avTarget = round(theDF[theTarget].mean(), 2)
    # get stddev of target
    stdTarget = round(theDF[theTarget].std(), 2)
    # make plot
    sns.distplot(theDF[theTarget], bins = binsNum)
    plt.title("Mean: " + str(avTarget) + "\n"
              "StdDev: " + str(stdTarget))
    plt.axvline(x = avTarget, color = "k", linestyle = "--", linewidth = 1.7)
    plt.xlabel(theLabel)
    plt.savefig(theTarget + "_distribution.pdf", dpi = 300)
    plt.close()
    # end of function

# function: visual of correlation matrices ---------------------------------------------------------
def plotCorrelationMatrices(theDF, theTarget, theMinCorrAbs):
    # function message
    print("\n")
    print("- Plotting correlation matrices ...")    
    # local variables
    someCopyDF = deepcopy(theDF)
    corrVec = None
    corrOrd = None
    compCorrMat = None
    reduCorrMat = None
    chosenOnes = None
    # get correlation coefficient with target and reorder columns
    corrVec = abs(someCopyDF.corr().round(2)[theTarget]).sort_values()
    corrOrd = corrVec.index
    someCopyDF = someCopyDF[corrOrd]
    chosenOnes = (corrVec[corrVec >= theMinCorrAbs]).drop([theTarget]).sort_values(ascending = False).index.tolist()
    # get complete correlation matrix
    compCorrMat = someCopyDF.corr().round(2)
    sns.heatmap(data = compCorrMat, annot = True, cmap = "RdYlBu_r", linewidths = 0.5)
    plt.savefig(theTarget + "_completeCorrMat.pdf", dpi = 300)
    plt.close()
    # get reduced correlation matrix
    reduCorrMat = someCopyDF.corr().round(2)
    reduCorrMat = reduCorrMat.mask(abs(reduCorrMat) < theMinCorrAbs)
    sns.heatmap(data = reduCorrMat, annot = True, cmap = "RdYlBu_r", linewidths = 0.5)
    plt.savefig(theTarget + "_reducedCorrMat.pdf", dpi = 300)
    plt.close()
    # end of function
    return(chosenOnes)

# function: determine best correlation models ------------------------------------------------------
def determineModelTuples(theBestVaribales):
    # function message
    print("\n")
    print("- Obtaining models ...")
    # local variables
    i = 0
    modelsNum = 0
    theModels = dict()
    # create models
    modelsNum = len(theBestVaribales)
    for i in range(modelsNum):
        for j in range(i, modelsNum):
            if(j in list(theModels.keys())):
                theModels[j].append(theBestVaribales[i])
            else:
                theModels[j] = [theBestVaribales[i]]
    # end of function
    return(theModels)

# function: analyze model --------------------------------------------------------------------------
def analyzeModel(theModel, theTrainSize, theDF, theTarget):
    # local variables
    theDataCols = []
    X = None
    Y = None
    xTrain = None
    xTest = None
    yTrain = None
    yTest = None
    linModel = None
    yPredict = None
    RSSvTrain = 0
    RSSvTest = 0
    AICvTrain = 0
    AICvTest = 0
    theCoefs = 0
    # get train and test sets
    X = pd.DataFrame(np.c_[theDF[theModel]], columns = theModel)
    Y = theDF[theTarget]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = (1 - theTrainSize), random_state = 5)
    # train model
    linModel = LinearRegression()
    linModel.fit(xTrain, yTrain)
    theCoefs = list(linModel.coef_)
    # evaluate model
    yLeanrt = linModel.predict(xTrain)
    yPredict = linModel.predict(xTest)
    # get RSSv
    RSSvTrain = (1/2) * np.sum((yTrain - yLeanrt)**2)
    RSSvTest = (1/2) * np.sum((yTest - yPredict)**2)
    # get AICv
    AICvTrain = len(yTrain) * math.log10(RSSvTrain / len(yTrain)) + 2 * len(theModel) * len(yTrain) / (len(yTrain) - len(theModel) - 1)
    AICvTest = len(yTest) * math.log10(RSSvTest / len(yTest)) + 2 * len(theModel) * len(yTest) / (len(yTest) - len(theModel) - 1)
    # end of function
    return(RSSvTrain, RSSvTest, AICvTrain, AICvTest, theCoefs)

# function: plot rss and aic lines for models ------------------------------------------------------
def plotRSSandAIC(theBestModel, RSSvTrainInfo, RSSvTestInfo, AICvTrainInfo, AICvTestInfo, xValues):
    # function message
    print("\n")
    print("- Plotting RSS and AIC results ...")
    # local variables
    i = 0
    everyModel = []
    bestModelText = ""
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
              "b", "g", "r", "c", "m"]
    # get all models
    everyModel = list(RSSvTrainInfo.keys())
    # build best model text
    bestModelText = "- Best model found to have:" + "\n"
    bestModelText = bestModelText + "* AIC test: " + str(theBestModel[0]) + "\n"
    bestModelText = bestModelText + "* Train Set %: " + str(theBestModel[1] * 100) + "\n"
    bestModelText = bestModelText + "* Variables: " + ", ".join(theBestModel[2]) + "\n"
    bestModelText = bestModelText + "* Coeficients: " + ", ".join([str(a) for a in theBestModel[3]])
    # print best model info
    print("\n")
    print(bestModelText)
    # plot rss lines
    for i in range(len(everyModel)):
        plt.plot(RSSvTrainInfo[everyModel[i]], color = colors[i], linestyle = "--", linewidth = 1.5, marker = ".",
        label = ", ".join(list(everyModel[i])) + " - train")
    for i in range(len(everyModel)):
        plt.plot(RSSvTestInfo[everyModel[i]], color = colors[i], linewidth = 2, marker = "o",
                 label = ", ".join(list(everyModel[i])) + " - test")
    plt.xticks(range(len(xValues)), [a*100 for a in xValues])
    plt.title(bestModelText)
    plt.ylabel("RSS")
    plt.xlabel("Training Set %")
    if(len(everyModel) < 5):
        plt.legend()
    plt.savefig("RSS.pdf", dpi = 300)
    plt.close()
    # plot rss lines
    for i in range(len(everyModel)):
        plt.plot(AICvTrainInfo[everyModel[i]], color = colors[i], linestyle = "--", linewidth = 1.5, marker = ".",
        label = ", ".join(list(everyModel[i])) + " - train")
    for i in range(len(everyModel)):
        plt.plot(AICvTestInfo[everyModel[i]], color = colors[i], linewidth = 2, marker = "o",
                 label = ", ".join(list(everyModel[i])) + " - test")
    plt.xticks(range(len(xValues)), [a*100 for a in xValues])
    plt.title(bestModelText)
    plt.ylabel("AIC")
    plt.xlabel("Training Set %")
    if(len(everyModel) < 5):
        plt.legend()
    plt.savefig("AIC.pdf", dpi = 300)
    plt.close()
    # end of function

# function: plot results ---------------------------------------------------------------------------
def plotScatters(myTarget, theVariables, theDF):
    # function message
    print("\n")
    print("- Plotting scatter plots of target vs variables in best fit model ...")
    # local variables
    i = 0
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
              "b", "g", "r", "c", "m"]
    # scatters
    for i in range(len(theVariables)):
        sns.scatterplot(x = theVariables[i], y = myTarget, data = theDF, color = colors[i])
        plt.xlabel(theVariables[i])
        plt.ylabel(myTarget)
        plt.title("Scatter: " + myTarget + " vs " + theVariables[i])
        plt.savefig(myTarget + "_vs_" + theVariables[i] + ".pdf", dpi = 300)
        plt.close()
    # end of function
    
# main #############################################################################################
# start message
print("\n\n")
print(">>> Boston House Dataset Analysis - @MarcosLaffitte")

# get data from scikit
theData = getData()

# isnull and rows check
makeVisualDataRows(theData)

# MEDV distrubition
plotDistributionTarget(theData, myTarget, "MEDV [ $1000's ]")

# correlation matrices
bestVariables = plotCorrelationMatrices(theData, myTarget, corrCoefAbs)

# determine increasing tupples of variables (models)
models = determineModelTuples(bestVariables)

# train and evaluate models
print("\n")
print("- Training and evaluating models ...")
for i in range(len(list(models.keys()))):
    RSSvTrainTot[tuple(models[i])] = []
    RSSvTestTot[tuple(models[i])] = []
    AICvTrainTot[tuple(models[i])] = []
    AICvTestTot[tuple(models[i])] = []
    for j in range(len(trainSetPercent)):
        (rssTr, rssTe, aicTr, aicTe, fitCoefs) = analyzeModel(models[i], trainSetPercent[j], theData, myTarget)
        RSSvTrainTot[tuple(models[i])].append(rssTr)
        RSSvTestTot[tuple(models[i])].append(rssTe)
        AICvTrainTot[tuple(models[i])].append(aicTr)
        AICvTestTot[tuple(models[i])].append(aicTe)
        modelResults = (aicTe, trainSetPercent[j], models[i], fitCoefs)
        modelEvaluation.append(modelResults)

# get best error based on AICv for test
modelEvaluation.sort()
theBestModelEver = modelEvaluation[0]

# plot rss and aic lines of models and output best model
plotRSSandAIC(theBestModelEver, RSSvTrainTot, RSSvTestTot, AICvTrainTot, AICvTestTot, trainSetPercent)

# plot variables vs target scatter plot for best model's variables
plotScatters(myTarget, theBestModelEver[2], theData)

# start message
print("\n")
print(">>> Finished!")
print("\n\n")

# end ##############################################################################################
####################################################################################################

