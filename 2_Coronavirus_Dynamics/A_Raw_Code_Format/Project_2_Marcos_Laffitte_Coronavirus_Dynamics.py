####################################################################################################
#                                                                                                  #
# - By: @MarcosLaffitte                                                                            #
#   https://github.com/MarcosLaffitte                                                              #
#                                                                                                  #
# - Proj2: Coronavirus Dynamics                                                                    #
#   dataset obtained from: https://github.com/CSSEGISandData/COVID-19                              #
#   up to march 19 2020                                                                            #
#                                                                                                  #
# - Masters in Mathematics                                                                         #
#                                                                                                  #
# - UNAM-IMATE, Juriquilla, Qro, Mex                                                               #
#                                                                                                  #
# - Course on Artificial Inteligence                                                               #
#                                                                                                  #
# - Prof: Esteban Hernandez Vargas, PhD                                                            #
#                                                                                                  #
# - Description: determine and evaluate models of coronavirus confirmed cases growth, considering  #
#                cummulative data as net population of confirmed cases (exponential growth).       #
#                                                                                                  #
# - Models:                                                                                        #
#   * 2nd degree polynomial                                                                        #
#   * 3nd degree polynomial                                                                        #
#   * 4nd degree polynomial                                                                        #
#   * 5nd degree polynomial                                                                        #
#   * exponential function                                                                         #
#                                                                                                  #
# - run (linux):      python3.5 thisScript.py                                                      #
#                                                                                                  #
# - data files                                                                                     #
#                             time_series_19-covid-Confirmed.csv                                   #
#                                                                                                  #
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

# not in python ------------------------------------------------------------------------------------
import numpy.polynomial.polynomial as poly
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# some stuff ------------------------------------------------------------------------------------
sns.set(rc={"figure.figsize":(8,25)})

# variables ########################################################################################
# input --------------------------------------------------------------------------------------------
# available countries: Italy, Japan, Spain, Germany, Iran, Mexico
theCountry = "Mexico"

# data ---------------------------------------------------------------------------------------------
# raw cummulative data
timeSeriesConfirmed = []

# constant -----------------------------------------------------------------------------------------
fileConfirmed = "time_series_19-covid-Confirmed.csv"
avilable = ["Italy", "Japan", "Spain", "Germany", "Iran", "Mexico"]
timeSteps = 58
time = list(range(timeSteps))
 ------------------------------------------------------------------------------------------
if(not theCountry in avilable):
    theCountry = "Italy"

# functions ########################################################################################
# function: get data -------------------------------------------------------------------------------
def getDataFrom():
    # function message
    print("\n")
    print("- Obtaining data from " + theCountry + " ...")
    # local variables
    theConfirmed = None
    theColumnHeaders = []
    firstColumn = ""
    secondColumn = ""
    thirdColumn = ""
    fourthColumn = ""
    theNa = []
    theInconsistenIndex = []
    colsConfirmed = []
    value = None
    # get data
    theConfirmed = pd.read_csv(fileConfirmed)
    # get intial columns
    theColumnHeaders = list(theConfirmed.columns.values)
    firstColumn = theColumnHeaders[0]
    secondColumn = theColumnHeaders[1]
    thirdColumn = theColumnHeaders[2]
    fourthColumn = theColumnHeaders[3]
    # get data from whole countries
    theConfirmed = theConfirmed.drop([firstColumn, thirdColumn, fourthColumn], axis = 1).set_index(secondColumn).transpose()
    theConfirmed = theConfirmed.loc[:, ~theConfirmed.columns.duplicated(keep = False)]
    theConfirmed = list(theConfirmed[theCountry].values)
    # end of function
    return(theConfirmed)

# main #############################################################################################
# start message
print("\n\n")
print(">>> Coronavirus Confirmed Cases - @MarcosLaffitte")

# get data
seriesConfirmed = getDataFrom()
seriesConfirmed = [val + 1 for val in seriesConfirmed]
naturalLogConfirmed = [np.log(val) for val in seriesConfirmed]

# train models
print("\n")
print("- Fitting models to data ...")
print("i) 2nd degree polynomial")
coefPoly2 = poly.polyfit(time, seriesConfirmed, 2)
print("ii) 3th degree polynomial")
coefPoly3 = poly.polyfit(time, seriesConfirmed, 3)
print("iii) 4th degree polynomial")
coefPoly4 = poly.polyfit(time, seriesConfirmed, 4)
print("iv) 5th degree polynomial")
coefPoly5 = poly.polyfit(time, seriesConfirmed, 5)
print("v) exponential function")
coefExp = poly.polyfit(time, naturalLogConfirmed, 1)

# obtaining predictions
myPoly2 = poly.Polynomial(coefPoly2)
myPoly3 = poly.Polynomial(coefPoly3)
myPoly4 = poly.Polynomial(coefPoly4)
myPoly5 = poly.Polynomial(coefPoly5)
myExp = poly.Polynomial(coefExp)
poly2Prediction = np.asarray([myPoly2(eachTime) for eachTime in time])
poly3Prediction = np.asarray([myPoly3(eachTime) for eachTime in time])
poly4Prediction = np.asarray([myPoly4(eachTime) for eachTime in time])
poly5Prediction = np.asarray([myPoly5(eachTime) for eachTime in time])
expPrediction = np.asarray([np.exp(myExp(eachTime)) for eachTime in time])

# obtain errors
# get RSS
poly2RSS = (1/2) * np.sum((seriesConfirmed - poly2Prediction)**2)
poly3RSS = (1/2) * np.sum((seriesConfirmed - poly3Prediction)**2)
poly4RSS = (1/2) * np.sum((seriesConfirmed - poly4Prediction)**2)
poly5RSS = (1/2) * np.sum((seriesConfirmed - poly5Prediction)**2)
expRSS = (1/2) * np.sum((seriesConfirmed - expPrediction)**2)
# get AICv
poly2AIC=len(seriesConfirmed)*math.log10(poly2RSS/len(seriesConfirmed))+2*len(coefPoly2)*len(seriesConfirmed)/(len(seriesConfirmed)-len(coefPoly2)-1)
poly3AIC=len(seriesConfirmed)*math.log10(poly3RSS/len(seriesConfirmed))+2*len(coefPoly3)*len(seriesConfirmed)/(len(seriesConfirmed)-len(coefPoly3)-1)
poly4AIC=len(seriesConfirmed)*math.log10(poly4RSS/len(seriesConfirmed))+2*len(coefPoly4)*len(seriesConfirmed)/(len(seriesConfirmed)-len(coefPoly4)-1)
poly5AIC=len(seriesConfirmed)*math.log10(poly5RSS/len(seriesConfirmed))+2*len(coefPoly5)*len(seriesConfirmed)/(len(seriesConfirmed)-len(coefPoly5)-1)
expAIC=len(seriesConfirmed)*math.log10(expRSS/len(seriesConfirmed))+2*len(coefExp)*len(seriesConfirmed)/(len(seriesConfirmed)-len(coefExp)-1)

# plot predictions
plt.subplot(5, 1, 1)
plt.plot(seriesConfirmed, linewidth = 1, color = "tab:blue", label = "real")
plt.plot(poly2Prediction, linestyle = "--", linewidth = 1.5, color = "lightsalmon", label = "2° polynomial")
plt.ylabel("Confirmed cases")
plt.title("AIC: " + str(round(poly2AIC, 4)))
plt.legend()
plt.subplot(5, 1, 2)
plt.plot(seriesConfirmed, linewidth = 1, color = "tab:blue", label = "real")
plt.plot(poly3Prediction, linestyle = "--", linewidth = 1.5, color = "tomato", label = "3° polynomial")
plt.title("AIC: " + str(round(poly3AIC, 4)))
plt.legend()
plt.subplot(5, 1, 3)
plt.plot(seriesConfirmed, linewidth = 1, color = "tab:blue", label = "real")
plt.plot(poly4Prediction, linestyle = "--", linewidth = 1.5, color = "r", label = "4° polynomial")
plt.title("AIC: " + str(round(poly4AIC, 4)))
plt.legend()
plt.subplot(5, 1, 4)
plt.plot(seriesConfirmed, linewidth = 1, color = "tab:blue", label = "real")
plt.plot(poly5Prediction, linestyle = "--", linewidth = 1.5, color = "brown", label = "5° polynomial")
plt.title("AIC: " + str(round(poly5AIC, 4)))
plt.legend()
plt.subplot(5, 1, 5)
plt.plot(seriesConfirmed, linewidth = 1, color = "tab:blue", label = "real")
plt.plot(expPrediction, linestyle = "--", linewidth = 1.5, color = "indigo", label = "exponential")
plt.title("AIC: " + str(round(expAIC, 4)))
plt.legend()
plt.xlabel("Days")
plt.savefig(theCountry + ".pdf")

# start message
print("\n")
print(">>> Finished!")
print("\n\n")

# end ##############################################################################################
####################################################################################################
