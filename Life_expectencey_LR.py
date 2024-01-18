Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

enc=OneHotEncoder()
labelencoder=LabelEncoder()

life=pd.read_csv("D:/assignments of data science/19 logistic regression 2 lasso and ridge regression/Life_expectencey_LR.csv")
life
life.head()
life.describe()
life.duplicated().sum()
life.isnull().sum()
life.fillna(life.mean(), inplace=True)

import statsmodels.formula.api as smf 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import pylab

final_ml = smf.ols('Life_expectancy ~ Year + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling', data = life).fit()
final_ml.summary() 

pred = final_ml.predict(life)

res = final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns

sns.residplot(x = pred, y = life.Life_expectancy, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

from sklearn.model_selection import train_test_split
life_train, life_test = train_test_split(life, test_size = 0.2)

model_train = smf.ols('Life_expectancy ~ Year + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling', data = life_train).fit()


test_pred = model_train.predict(life_test)


test_resid = test_pred - life_test

test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

train_pred = model_train.predict(life_train)

train_resid  = train_pred - life_train

train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(life.iloc[:, 1:], life.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(life.iloc[:, 1:])


lasso_reg.score(life.iloc[:, 1:], life.Life_expectancy)

np.sqrt(np.mean((lasso_pred - life.Life_expectancy)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(life.iloc[:, 1:], life.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(life.iloc[:, 1:])


ridge_reg.score(life.iloc[:, 1:], life.Life_expectancy)

np.sqrt(np.mean((ridge_pred - life.Life_expectancy)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(life.iloc[:, 1:], life.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(life.iloc[:, 1:])

enet_reg.score(life.iloc[:, 1:], life.Life_expectancy)

np.sqrt(np.mean((enet_pred - life.Life_expectancy)**2))