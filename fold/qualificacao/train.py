
from scipy.io import arff
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew
#Algoritimos Machine Learning
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import os
print(os.listdir("f3"))

#carrega os dados de treinamento
data_train= arff.loadarff('f2/train/sheep.arff')
train = pd.DataFrame(data_train[0])



#carrega os dados de teste
data_test = arff.loadarff('f2/val/sheep.arff')
test = pd.DataFrame(data_test[0])



##dados
print(train.head())
print('**'* 50)
print(test.head())

#atributos e tipos
print(train.info())
print('**'* 50)
print(test.info())


sns.distplot(train['Weight'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Weight'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Weight distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['Weight'], plot=plt)
plt.show() 




plt.scatter(x= 'Area', y='Weight', data = train)

##matrix confusion correlation
plt.figure(figsize=(30,8))
sns.heatmap(train.corr(),cmap='coolwarm',annot = True)
plt.show()

sns.lmplot(x='Minor_axis',y='Weight',data=train) # 1stFlrSF seems very corelated with SalePrice.




plt.figure(figsize=(16,8))
sns.boxplot(x='Minor_axis',y='Weight',data=train)
plt.show()



sns.lmplot(x='Major_axis',y='Weight',data=train)






################outro teste

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Weight', axis=1), train['Weight'], test_size=0.3, random_state=101)

X_train=train.drop('Weight', axis=1)
y_train=train['Weight']


X_test=test.drop('Weight', axis=1)
y_test=test['Weight']

print('Train X')

print(X_train)
print('Xtest')
print(X_test)

print('Ytest')
print(y_train)
print('Ytest')
print(y_test)

# we are going to scale to data

y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)


##criando modelo de regressão linear
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm)


# print the intercept
print(lm.intercept_)

predictions = lm.predict(X_test)
predictions= predictions.reshape(-1,1)

plt.figure(figsize=(15,8))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()



#Predito e testado
plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'predict')
plt.show()


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


########################3Gradient Boosting Regression
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)


clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
acc_gbr = round(clf.score(X_train, y_train) * 100, 2)
print(round(acc_gbr,2,), "%")


plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(clf_pred, label = 'predict')
plt.show()


###################3Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 100)
dtreg.fit(X_train, y_train)

dtr_pred = dtreg.predict(X_test)
dtr_pred= dtr_pred.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))
print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
acc_dtr = round(dtreg.score(X_train, y_train) * 100, 2)
print(round(acc_dtr,2,), "%")

plt.figure(figsize=(15,8))
plt.scatter(y_test,dtr_pred,c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


####################SVM Regression

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('MSE:', metrics.mean_squared_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
acc_svr = round(svr.score(X_train, y_train) * 100, 2)
print(round(acc_svr,2,), "%")

plt.figure(figsize=(15,8))
plt.scatter(y_test,svr_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(svr_pred, label = 'predict')
plt.show()

###############Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 500, random_state = 0)
rfr.fit(X_train, y_train)



rfr_pred= rfr.predict(X_test)
rfr_pred = rfr_pred.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))
print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))
acc_rfr = round(rfr.score(X_train, y_train) * 100, 2)
print(round(acc_rfr,2,), "%")


plt.figure(figsize=(15,8))
plt.scatter(y_test,rfr_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(rfr_pred, label = 'predict')
plt.show()

################Light GBM



import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=3000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X_train,y_train)



lgb_pred = model_lgb.predict(X_test)
lgb_pred = lgb_pred.reshape(-1,1)

print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))
print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))
acc_lgb = round(model_lgb.score(X_train, y_train) * 100, 2)
print(round(acc_lgb,2,), "%")

plt.figure(figsize=(15,8))
plt.scatter(y_test,lgb_pred, c='orange')
plt.xlabel('Y Test')

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(lgb_pred, label = 'predict')
plt.show()

##Perceptron

# Perceptron
print('Perceptron')
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train,y_train)

perceptron_pred = perceptron.predict(X_test)



print('MAE:', metrics.mean_absolute_error(y_test, perceptron_pred))
print('MSE:', metrics.mean_squared_error(y_test, perceptron_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, perceptron_pred)))
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print(round(acc_perceptron,2,), "%")

plt.figure(figsize=(15,8))
plt.scatter(y_test,perceptron_pred, c='orange')
plt.xlabel('Y Test')

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(perceptron_pred, label = 'predict')
plt.show()



# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)


print('MAE:', metrics.mean_absolute_error(y_test, knn_pred))
print('MSE:', metrics.mean_squared_error(y_test, knn_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(round(acc_knn,2,), "%")

plt.figure(figsize=(15,8))
plt.scatter(y_test,knn_pred, c='orange')
plt.xlabel('Y Test')

plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(knn_pred, label = 'predict')
plt.show()







##############melhor modelo https://www.kaggle.com/samukaunt/titanic-passo-a-passo-com-8-modelos-ml-pt-br https://www.kaggle.com/fatmakursun/house-price-some-of-regression-models/output

results = pd.DataFrame({
    'Model': ['Light GBM', 'Support Vector Machines Regression', 'Random Forest Regressor', 
              'Decision Tree Regression', 'Gradient Boosting Regression', 'Perceptron', 
              'KNN'],
    'Score': [acc_lgb, acc_svr, acc_rfr, 
              acc_dtr, acc_gbr, acc_perceptron, 
              acc_knn]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)





