"""
=========================
SKLEARN Model traningMeasure region properties
=========================

This example shows how to measure properties of labelled image regions.

"""


##############melhor modelo https://www.kaggle.com/samukaunt/titanic-passo-a-passo-com-8-modelos-ml-pt-br https://www.kaggle.com/fatmakursun/house-price-some-of-regression-models/output



import time
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
from sklearn.linear_model import LinearRegression,  HuberRegressor, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgboost
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
import os
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.metrics import f1_score 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression

def run(train_file, test_file,fold_name,fold):
            data_train= arff.loadarff(train_file)
            train = pd.DataFrame(data_train[0])

            #carrega os dados de teste            
            data_test = arff.loadarff(test_file)
            test = pd.DataFrame(data_test[0])

            #from sklearn.model_selection import train_test_split
            #x_train, x_test, y_train, y_test = train_test_split(train.drop('Weight', axis=1), train['Weight'], test_size=0.3, random_state=101)
            #columns=['Area','K_0_19','K_20_39','K_40_59','K_60_79','K_80_99','K_100_119','K_120_139','K_140_159','K_160_180','Hu_0','Hu_1','Hu_2','Hu_3','Hu_4','Hu_5','Hu_6','Weight']
            columns=['K_80_99','Extent','Area','K_0_19','K_20_39','K_40_59','K_60_79','Hu_0','Hu_2','Hu_4','Hu_5','Hu_6','Weight']
            
            x_train=train.drop(columns, axis=1).copy()
            y_train=train['Weight'].copy()
            x_test=test.drop(columns, axis=1).copy()
            y_test=test['Weight'].copy()

            # we are going to scale to data - Normalize Data
            print(x_test)
            print(y_test)
            y_train= y_train.values.reshape(-1,1)
            y_test= y_test.values.reshape(-1,1)

        
            sc_X = QuantileTransformer(output_distribution='normal')
            sc_y = QuantileTransformer(output_distribution='normal')
            x_train = sc_X.fit_transform(x_train)
            x_test =  sc_X.fit_transform(x_test)
            y_train = sc_y.fit_transform(y_train)
            y_test = sc_y.fit_transform(y_test)
            
          
            lm = LinearRegression()
            lm.fit(x_train,y_train)
            lm_pred = lm.predict(x_test)
            lm_pred= lm_pred.reshape(-1,1)
            r2 = r2_score(y_test,lm_pred)
            lm_pred=sc_y.inverse_transform(lm_pred)
            y_test=sc_y.inverse_transform(y_test)
            mae=metrics.mean_absolute_error(y_test, lm_pred)
            print('MAE: ', mae)
            mse= metrics.mean_squared_error(y_test, lm_pred)
            print('MSE: ', mse )
            rmse= np.sqrt(metrics.mean_squared_error(y_test,lm_pred))
            print('RMSE: ', rmse)
            print('R2: ', r2)
           
        

if __name__ == "__main__":

    for i in range(8):
       train_file="f"+str(i+1)+"/train/sheep.arff"
       test_file="f"+str(i+1)+"/val/sheep.arff"
       run(train_file, test_file,"Fold_"+str(i+1),i)
 

