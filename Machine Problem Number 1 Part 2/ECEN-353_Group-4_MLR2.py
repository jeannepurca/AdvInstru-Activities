# ECEN 353_Group 4_MLR2
# Hernandez, Daniella, Kim C.
# Purca, Jeanne Mae M.
# Roxas, April T.




# SIMPLE LINEAR REGERESSION TEMPLATE

# To Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To Import the Dataset
dataset = pd.read_csv("50_Startups.csv")

# Preliminary Analysis of the Dataset

# A. To check if there is a missing data
missing_data = dataset.isnull().sum().sort_values(ascending=False)

# B. To check column names and the total records
dataset_count = dataset.count()

# C. To view inforrmation about the dataset
print(dataset.info())

# D. To view the statistical summary of the dataset
statistics = dataset.describe()

# To create a matrix of Independent Variable, X
X = dataset.iloc[:, :1].values

# TO Create a matrix of Dependent Variable, Y
Y = dataset.iloc[:, 4:].values

# To determine the Pearson's Coefficient for Correlation
from scipy.stats import pearsonr

X_axis = dataset["R&D Spend"]
Y_axis = dataset["Profit"]
r, p = pearsonr(X_axis, Y_axis)
print("The Pearson Coefficient of Correlation:", r, "with a P-value of", p)
print(" ")


# To Split the Whole Dataset Into Training Dataset and Testing Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, train_size=0.8, test_size=0.2, random_state=0)

# To Fit the Training Dataset into a Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
simple_linear_regression = LinearRegression()
simple_linear_regression.fit(X_train, Y_train)

# To Predict the Output Using the Testing Dataset
Y_predict = simple_linear_regression.predict(X_test)

# To visualize the Training Dataset and the Simple Linear Regression Model
plt.scatter(X_train, Y_train, color = "red") # To plot the training dataset
Y_predict_train = simple_linear_regression.predict(X_train)
plt.plot(X_train, Y_predict_train, color = "blue")
plt.title("Plot of R&D vs. Profit Using the Training Dataset")
plt.xlabel("R and D")
plt.ylabel("Profit")

import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="red", label="Training Dataset")
blue_patch = mpatches.Patch(color="blue", label="Simple Linear Regression Model")
plt.legend(handles=[red_patch, blue_patch])
plt.show()

#To Determine the Intercept and the Coefficient of th Simple Linear Regression Model

intercept = simple_linear_regression.intercept_
coefficient = simple_linear_regression.coef_
print("For the Simple Linear Regression Model, the Intercept is", intercept, "and the Coefficient is", coefficient)
print (" ")


# To Apply K-Fold Cross-Validation for the Simple Liinear Regression Model
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score

# Try the following Performance Metrics:
    # A. Mean Absolute Error = neg_mean_absolute_error"
    # B. Mean Squared Error = "neg_mean_squared_error"
    # C. R-squared Error = "r2_score"
    # D. Explained Variance Score = "explained_variance"
    
# A. For the Mean Absolute Error (MAE) as Scoring Metric for Cross-Validation
MAE = (cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=k_fold, scoring="neg_mean_absolute_error"))*-1
MAE_mean = MAE.mean()
MAE_deviation = MAE.std()
print("The Mean Absolute Error of K-Folds", MAE)
print(" ")
print("The Average of the Mean Absolute Error of K-Folds:", MAE_mean)
print(" ")
print("The Standard Deviation of the Mean Absolute Error of K-Folds:", MAE_deviation)
print(" ")

# B. For the Mean Squared Error (MSE) as Scoring Metric for Cross-Validation
MSE = (cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=k_fold, scoring="neg_mean_squared_error"))*-1
MSE_mean = MSE.mean()
MSE_deviation = MSE.std()
print("The Mean Squared Error of K-Folds", MSE)
print(" ")
print("The Average of the Mean Squared Error of K-Folds:", MSE_mean)
print(" ")
print("The Standard Deviation of the Mean Squared Error of K-Folds:", MSE_deviation)
print(" ")

from math import sqrt
RMSE = sqrt(MSE_mean)
print("The Average of the Mean Squared Error of K-Folds:", MSE_mean)
print(" ")

# C. For the R-squared Error (R2E) as Scoring Metric for Cross-Validation
R2E = (cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=k_fold, scoring="r2"))
R2E_mean = R2E.mean()
R2E_deviation = R2E.std()
print("The R-Squared Error of K-Folds", R2E)
print(" ")
print("The Average of the R-Squared Error of K-Folds:", R2E_mean)
print(" ")
print("The Standard Deviation of the R-Squared Error of K-Folds:", R2E_deviation)
print(" ")


# D. For the Explained Variance Score as Scoring Metric for Cross-Validation
EVS = (cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=k_fold, scoring="explained_variance"))
EVS_mean = EVS.mean()
EVS_deviation = EVS.std()
print("The Explained Variance Score:", EVS)
print(" ")
print("The Average of the Exaplined Varince Score of K-Folds:", EVS_mean)
print(" ")
print("The Standard Deviation of the Explained Variance Score:", EVS_deviation)
print(" ")

# To Determine the Hold-Out Validation for the Simple Linear Regression Model

# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(Y_test, Y_predict)
print("The Mean Absolute Error: %.4f"
      % MAE)
print(" ")

# B. For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(Y_test, Y_predict)
print("The Mean Squared Error: %.4f"
      % MSE)
print(" ")

# C. For the Root Mean Sqaured Error (RMSE)
RMSE =sqrt(MSE)
print("The Root Mean Sqaured Error: %.4f"
      % RMSE)
print(" ")

# D.For the Explained Variance Score
from sklearn.metrics import explained_variance_score

EVS = explained_variance_score(Y_test, Y_predict)
print("The Explained Variance Score: %.4f"
      % EVS)
print(" ")


# E. For the Coefficient of Determination Regression Score Function (R-Squared)
from sklearn.metrics import r2_score

R2 = r2_score(Y_test, Y_predict)
print("The  Coefficient of Determination Regression Score: %.4f"
      % R2)
print(" ")













