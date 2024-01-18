Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd
import numpy as np

toyota=pd.read_csv("D:/assignments of data science/19 logistic regression 2 lasso and ridge regression/ToyotaCorolla.csv",encoding='latin1')

toyota
toyota.head()
toyota.describe()
toyota.duplicated().sum()
toyota.isnull().sum()

toyota = toyota[['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]


import matplotlib.pyplot as plt  
import seaborn as sns

for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.histplot(data=toyota, x=predictor)
    
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.boxplot(data=toyota, x=predictor)     
    
plt.bar(height = toyota["Price"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Age_08_04"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["KM"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["HP"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["cc"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Doors"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Gears"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Quarterly_Tax"], x = np.arange(1, 1437, 1))
plt.bar(height = toyota["Weight"], x = np.arange(1, 1437, 1))

sns.jointplot(x=toyota['Age_08_04'], y=toyota['Price'])
sns.jointplot(x=toyota['KM'], y=toyota['Price'])
sns.jointplot(x=toyota['HP'], y=toyota['Price'])
sns.jointplot(x=toyota['cc'], y=toyota['Price'])
sns.jointplot(x=toyota['Doors'], y=toyota['Price'])
sns.jointplot(x=toyota['Gears'], y=toyota['Price'])
sns.jointplot(x=toyota['Quarterly_Tax'], y=toyota['Price'])
sns.jointplot(x=toyota['Weight'], y=toyota['Price'])    

plt.figure(1, figsize=(16, 10))
for i, predictor in enumerate(toyota):
    plt.figure(i)
    sns.countplot(data=toyota, x=predictor)
    
from scipy import stats
import pylab

stats.probplot(toyota.Price, dist = "norm", plot = pylab)    
plt.show() 

stats.probplot(np.log(toyota['Price']),dist="norm",plot=pylab)

sns.pairplot(toyota.iloc[:, :])    
    
a = toyota.corr() 

import statsmodels.formula.api as smf 

ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit()
    
ml1.summary()

import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

toyota_new = toyota.drop(toyota.index[[80,221,960]])

ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota_new).fit() 

ml_new.summary()

"ml_new =  Final Model"
    
pred = ml_new.predict(toyota)

res = ml_new.resid
sm.qqplot(res)
plt.show()

stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

sns.residplot(x = pred, y = toyota.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml_new)

from sklearn.model_selection import train_test_split
toyota_train, toyota_test = train_test_split(toyota, test_size = 0.2)
    
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight", data = toyota_train).fit()    

test_pred = model_train.predict(toyota_test)

test_resid = test_pred - toyota_test.Price    

test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse     

train_pred = model_train.predict(toyota_train)    

train_resid  = train_pred - toyota_train.Price    

train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse    
    
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()   
    
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(toyota.iloc[:, 1:],toyota.Price)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(toyota.iloc[:, 1:])

lasso_reg.score(toyota.iloc[:, 1:], toyota.Price)

np.sqrt(np.mean((lasso_pred - toyota.Price)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(toyota.iloc[:, 1:], toyota.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(toyota.iloc[:, 1:])

ridge_reg.score(toyota.iloc[:, 1:], toyota.Price)

np.sqrt(np.mean((ridge_pred - toyota.Price)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(toyota.iloc[:, 1:], toyota.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(toyota.iloc[:, 1:])

enet_reg.score(toyota.iloc[:, 1:], toyota.Price)

np.sqrt(np.mean((enet_pred - toyota.Price)**2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    