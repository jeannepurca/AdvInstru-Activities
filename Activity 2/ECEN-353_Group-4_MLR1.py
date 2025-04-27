# ECEN 353_Group 4_MLR2
# Hernandez, Daniella, Kim C.
# Purca, Jeanne Mae M.
# Roxas, April T.


# Multiple Linear Regression Template

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
X = dataset.iloc[:, 0:4].values

# TO Create a matrix of Dependent Variable, Y
Y = dataset.iloc[:, 4:].values

# To View the Scatter Plot of the Dataset
import seaborn as sns
sns.pairplot(dataset) # better for many features
plt.show()


sns.boxplot(x="State", y="Profit", data=dataset) #for categorical
plt.show()

from scipy import stats 

group = dataset[["State", "Profit"]]
groupby = group.groupby(["State"])
f_val, p_val = stats.f_oneway(groupby.get_group("California")["Profit"], groupby.get_group("Florida")["Profit"], groupby.get_group("New York")["Profit"])
print("The ANOVA results:", f_val, "with a p-value equal to", p_val)

# To determine the Pearson's Coefficient for Correlation
from scipy.stats import pearsonr

X_axis = dataset["R&D Spend"]
Y_axis = dataset["Profit"]
r, p = pearsonr(X_axis, Y_axis)
print("The Pearson Coefficient of Correlation:", r, "with a P-value of", p)
print(" ")

X_axis = dataset["Administration"]
Y_axis = dataset["Profit"]
r, p = pearsonr(X_axis, Y_axis)
print("The Pearson Coefficient of Correlation:", r, "with a P-value of", p)
print(" ")

X_axis = dataset["Marketing Spend"]
Y_axis = dataset["Profit"]
r, p = pearsonr(X_axis, Y_axis)
print("The Pearson Coefficient of Correlation:", r, "with a P-value of", p)
print(" ")

dataset_correlation = dataset.corr() #Substitute way but without P-value

# To Encode the Categorical Data (State) in the Independent Variable, X, to make it Nominal (Without Ranking)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer([("State", OneHotEncoder(categories="auto"), [3])], remainder="passthrough")
X = column_transformer.fit_transform(X)
X = X.astype(float)  #fromm array of obj to array of floats

# To Avoid the Dummy Variable Trap for the Categorical Data (Sate) in the INdependent Variable, X
    # Note: Remove the Column Index 0 of the Dummy Variable

X_dummytrap = X.copy()
X_dummytrap = X_dummytrap[:, 1:]

# To Split the Whole Dataset Into Training Dataset and Testing Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split (X_dummytrap, Y, train_size=0.8, test_size=0.2, random_state=0)

# TO PERFORM FEATURE SCALE

# or the Standardization Feature Scaling
from sklearn.preprocessing import StandardScaler # For the data that is not normally distributed
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
standard_scaler = StandardScaler()
X_train_standard[:, 2:] = standard_scaler.fit_transform(X_train_standard[:, 2:])
X_test_standard[:, 2:] = standard_scaler.transform(X_test_standard[:, 2:]) 


# To Fit the Training Dataset into a Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_train_standard, Y_train)

# To Predict the Output Using the Testing Dataset
Y_predict = multiple_linear_regression.predict(X_test_standard)

# To Determine the Intercept and the Coefficient of th Simple Linear Regression Model

intercept = multiple_linear_regression.intercept_
coefficient = multiple_linear_regression.coef_
print("For the Multiple Linear Regression Model, the Intercept is", intercept, "and the Coefficients are", coefficient)
print (" ")


# To Apply K-Fold Cross-Validation for the Multiple Linear Regression Model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score


# Try the following Performance Metrics:
    # A. Mean Absolute Error = neg_mean_absolute_error"
    # B. Mean Squared Error = "neg_mean_squared_error"
    # C. R-squared Error = "r2_score"
    # D. Explained Variance Score = "explained_variance"

# To Feature Scale the X_dummytrap Using Standardization Feature Scaling
X_dummytrap_standard = X_dummytrap.copy()
X_dummytrap_standard[:, 2:] = standard_scaler.fit_transform(X_dummytrap_standard[:, 2:])


# A. For the Mean Absolute Error (MAE) as Scoring Metric for Cross-Validation
MAE = (cross_val_score(estimator=multiple_linear_regression, X=X_dummytrap_standard, y=Y, cv=k_fold, scoring="neg_mean_absolute_error"))*-1
MAE_mean = MAE.mean()
MAE_deviation = MAE.std()
print("The Mean Absolute Error of K-Folds", MAE)
print(" ")
print("The Average of the Mean Absolute Error of K-Folds:", MAE_mean)
print(" ")
print("The Standard Deviation of the Mean Absolute Error of K-Folds:", MAE_deviation)
print(" ")

# B. For the Mean Squared Error (MSE) as Scoring Metric for Cross-Validation
MSE = (cross_val_score(estimator=multiple_linear_regression, X=X_dummytrap_standard, y=Y, cv=k_fold, scoring="neg_mean_squared_error"))*-1
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
R2E = (cross_val_score(estimator=multiple_linear_regression, X=X_dummytrap_standard, y=Y, cv=k_fold, scoring="r2"))
R2E_mean = R2E.mean()
R2E_deviation = R2E.std()
print("The R-Squared Error of K-Folds", R2E)
print(" ")
print("The Average of the R-Squared Error of K-Folds:", R2E_mean)
print(" ")
print("The Standard Deviation of the R-Squared Error of K-Folds:", R2E_deviation)
print(" ")


# D. For the Explained Variance Score as Scoring Metric for Cross-Validation
EVS = (cross_val_score(estimator=multiple_linear_regression, X=X_dummytrap_standard, y=Y, cv=k_fold, scoring="explained_variance"))
EVS_mean = EVS.mean()
EVS_deviation = EVS.std()
print("The Explained Variance Score:", EVS)
print(" ")
print("The Average of the Exaplined Varince Score of K-Folds:", EVS_mean)
print(" ")
print("The Standard Deviation of the Explained Variance Score:", EVS_deviation)
print(" ")


# To Determine the Hold-Out Validation for the Multiple Linear Regression Model

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


############## BACKWARD ELIMINATION PROCESS ##################

import statsmodels.regression.linear_model as lm

"""
#index 0 Florida                 x1
#index 1 New York                x2
#index 2 R and D                 x3
#index 3 Administration          x4
#index 4 Marketing               x5
 
Y = b0x0 +b1x1 + b2x2 + be3x3 .....as x0=1
"""
 

X_new = np.append(arr=np.ones((50,1)).astype(int), values = X_dummytrap_standard, axis = 1)


"""
NOTE

index 0 is for XO = column of 1
index 1 is for x1 = dummy of state of Florida
index 2 is for x2 = dummy of state of New York
index 3 is for x3 = R and D
index 4 is for x4 = Admin
index 5 is for x5 = Marketing
"""

X_optimal1 = X_new[:, [0, 1, 2, 3, 4, 5]]
mlr_sm = lm.OLS(exog = X_optimal1, endog = Y).fit()
mlr_sm.summary()

# X2 (Dummy of State of New York) has the highest P-value = 0.99, which is greater the SL = 0.05, thus, it will be removed. (kasi backward elim)

X_optimal2 = X_new[:, [0, 1, 3, 4, 5]]


"""
NOTE

index 0 is for XO = column of 1
index 1 is for x1 = dummy of state of Florida
index 2 is for x2 = R and D
index 3 is for x3 = Admin
index 4 is for x4 = Marketing
"""

mlr_sm = lm.OLS(exog = X_optimal2, endog = Y).fit()
mlr_sm.summary()

# X2 (Dummy of State of California) has the highest P-value = 0.94, which is greater the SL = 0.05, thus, it will be removed. (kasi backward elim)


X_optimal3 = X_new[:, [0, 3, 4, 5]]

"""
NOTE

index 0 is for XO = column of 1
index 1 is for x1 = R and D
index 2 is for x2 = Admin
index 3 is for x3 = Marketing
"""

mlr_sm = lm.OLS(exog = X_optimal3, endog = Y).fit()
mlr_sm.summary()

# X2 (Admin) has the highest P-value = 0.602, which is greater the SL = 0.05, thus, it will be removed. (kasi backward elim)

X_optimal4 = X_new[:, [0, 3, 5]]

"""
NOTE

index 0 is for XO = column of 1
index 1 is for x1 = R and D
index 2 is for x2 = Marketing
"""

mlr_sm = lm.OLS(exog = X_optimal4, endog = Y).fit()
mlr_sm.summary()


# X2 (Marketing) has the highest P-value = 0.06, which is greater the SL = 0.05, thus, it will be removed. (kasi backward elim)

X_optimal5 = X_new[:, [0, 5]]

"""
NOTE

index 0 is for XO = column of 1
index 1 is for x1 = R and D
"""

mlr_sm = lm.OLS(exog = X_optimal5, endog = Y).fit()
mlr_sm.summary()

# FINDINGS: It is only the feature "R and D" that is significant to predict the Profit.

# ADDITIONAL TASK:
    # Create a SLR using R and D as the Feature
    # Create a MLR using " R and D" and marketing as Features
    # Tabulate all results in Excel




















