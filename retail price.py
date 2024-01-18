Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd
import numpy as np

company=pd.read_csv("D:/assignments of data science/19 logistic regression 2 lasso and ridge regression/RetailPrices_data.csv")   

company

company.head()
company.describe()
company.duplicated().sum()
company.isnull().sum()

del company['Unnamed: 0']

company=pd.get_dummies(company, columns=["cd","multi","premium"], drop_first=True)

company.describe()

import matplotlib.pyplot as plt
import seaborn as sns

for i, predictor in enumerate(company):
    plt.figure(i)
    sns.histplot(data=company, x=predictor)
    
for i, predictor in enumerate(company):
    plt.figure(i)
    sns.boxplot(data=company, x=predictor)    

plt.bar(height = company["price"], x = np.arange(1, 6260, 1))
plt.bar(height = company["speed"], x = np.arange(1, 6260, 1))
plt.bar(height = company["hd"], x = np.arange(1, 6260, 1))
plt.bar(height = company["ram"], x = np.arange(1, 6260, 1))
plt.bar(height = company["screen"], x = np.arange(1, 6260, 1))
plt.bar(height = company["ads"], x = np.arange(1, 6260, 1))
plt.bar(height = company["trend"], x = np.arange(1, 6260, 1))
plt.bar(height = company["cd_yes"], x = np.arange(1, 6260, 1))
plt.bar(height = company["multi_yes"], x = np.arange(1, 6260, 1))
plt.bar(height = company["premium_yes"], x = np.arange(1, 6260, 1))

from scipy import stats
import pylab

stats.probplot(company.price, dist='norm',plot=pylab)
plt.show()

sns.pairplot(company.iloc[:, :])

company.corr()
a = company.corr()

import statsmodels.formula.api as smf # for regression model

    
ml1 = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit()

ml1.summary()

import statsmodels.api as sm
company_new = company.drop(company.index[[3783,4477,5960]])

ml_new = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company_new).fit()  

ml_new.summary()

"price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes"

rsq_speed = smf.ols('speed ~ hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared 
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ speed + hd + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared 
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ ram + speed + hd + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared 
vif_screen = 1/(1 - rsq_screen) 

rsq_ads = smf.ols('ads ~ screen + ram + speed + hd + trend + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ ads + screen + ram + speed + hd + cd_yes + multi_yes + premium_yes', data = company).fit().rsquared
vif_trend = 1/(1 - rsq_trend) 

rsq_cd_yes = smf.ols('cd_yes ~ ads + screen + ram + speed + hd + trend + multi_yes + premium_yes', data = company).fit().rsquared 
vif_cd_yes = 1/(1 - rsq_cd_yes) 

rsq_multi_yes = smf.ols('multi_yes ~ cd_yes + ads + screen + ram + speed + hd + trend + premium_yes', data = company).fit().rsquared 
vif_multi_yes = 1/(1 - rsq_multi_yes) 

rsq_premium_yes = smf.ols('premium_yes ~ multi_yes + cd_yes + ads + screen + ram + speed + hd + trend', data = company).fit().rsquared  
vif_premium_yes = 1/(1 - rsq_premium_yes)  

d1 = {'Variables':['speed', 'hd', 'ram', 'screen', 'ads', 'trend', 'cd_yes','multi_yes', 'premium_yes'], 'VIF':[vif_speed, vif_hd, vif_ram, vif_screen,vif_ads,vif_trend,vif_cd_yes,vif_multi_yes,vif_premium_yes]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

final_ml = smf.ols('price ~ speed + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes', data = company).fit()
final_ml.summary()

pred = final_ml.predict(company)

res = final_ml.resid
sm.qqplot(res)
plt.show()

stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

sns.residplot(x = pred, y = company.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

from sklearn.model_selection import train_test_split
company_train, company_test = train_test_split(company, test_size = 0.2) 

model_train = smf.ols("price ~ speed + hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_yes", data = company_train).fit()

test_pred = model_train.predict(company_test)

test_resid = test_pred - company_test.price


test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

train_pred = model_train.predict(company_train)

train_resid  = train_pred - company_train.price

train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(company.iloc[:, 1:], company.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(company.iloc[:, 1:])

lasso_reg.score(company.iloc[:, 1:], company.price)

np.sqrt(np.mean((lasso_pred - company.price)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(company.iloc[:, 1:], company.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(company.iloc[:, 1:])

ridge_reg.score(company.iloc[:, 1:], company.price)

np.sqrt(np.mean((ridge_pred - company.price)**2))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(company.iloc[:, 1:], company.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(company.iloc[:, 1:])

enet_reg.score(company.iloc[:, 1:], company.price)

np.sqrt(np.mean((enet_pred - company.price)**2))




