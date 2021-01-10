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
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
class Data():

     def __init__(self):
         self.vet_mae=[]
         self.vet_mse=[]
         self.vet_rmse=[]
         self.vet_mape=[]
         self.vet_coeff=[]
         self.vet_r2=[]
         self.vet_pred=[]
         self.vet_test=[]
         self.vet_time=[]
         self.vet_r2_adj=[]
         pass   
     def add_data(self,mae,mse,rmse,mape,coef,r2,pred,test,elapsed_time,r2_adj):
         self.vet_mae.append(mae)
         self.vet_mse.append(mse)
         self.vet_rmse.append(rmse)
         self.vet_mape.append(mape)
         self.vet_coeff.append(coef)
         self.vet_r2.append(r2)
         self.vet_pred.append(pred)
         self.vet_test.append(test)
         self.vet_time.append(elapsed_time)
         self.vet_r2_adj.append(r2_adj)
class MLSklean():
     log_file = None #r ou a
     vet_fold=[]
     sc_X=None
     sc_y=None
     def __init__(self):
            self.vet_fold=[] #lm={},gbr={},dtr={},svr={},rfr={},knn={},gbm={},perc={}
            for i in range(8):
                self.vet_fold.append(Data())
            pass
     def run(self, train_file, test_file,fold_name,fold):
            #print(os.listdir("f3"))
            self.log_file = open('log_experiment.txt', 'a') #r ou a
            self.log_file.write('\n\n\n'+fold_name+' Experiment:\n')
            #carrega os dados de treinamento
            data_train= arff.loadarff(train_file)
            train = pd.DataFrame(data_train[0])

            #carrega os dados de teste            
            data_test = arff.loadarff(test_file)
            test = pd.DataFrame(data_test[0])

            #from sklearn.model_selection import train_test_split
            #x_train, x_test, y_train, y_test = train_test_split(train.drop('Weight', axis=1), train['Weight'], test_size=0.3, random_state=101)

            x_train=train.drop('Weight', axis=1).copy()
            y_train=train['Weight'].copy()
            x_test=test.drop('Weight', axis=1).copy()
            y_test=test['Weight'].copy()

            # we are going to scale to data - Normalize Data
            print(y_test)
            y_train= y_train.values.reshape(-1,1)
            y_test= y_test.values.reshape(-1,1)
        
            self.sc_X = StandardScaler()
            self.sc_y = StandardScaler()
            x_train = self.sc_X.fit_transform(x_train)
            x_test = self.sc_X.fit_transform(x_test)
            y_train = self.sc_y.fit_transform(y_train)
            y_test = self.sc_y.fit_transform(y_test)
            
            ###############################################

            sns.distplot(train['Weight'] , fit=norm);

            # Get the fitted parameters used by the function
            (mu, sigma) = norm.fit(train['Weight'])
            #print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
            self.write_log('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
            #Now plot the distribution
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
            plt.ylabel('Frequency')
            plt.title('Weight distribution')

            #Get also the QQ-plot
            fig = plt.figure()
            res = stats.probplot(train['Weight'], plot=plt)
            #plt.show() 
            plt.savefig(fold_name+'_distribuition.png')

            ##Call Algoritms
            self.processLM(fold_name,x_test,y_test,x_train,y_train,fold)
            self.processGBR(fold_name,x_test,y_test,x_train,y_train,fold)
            self.processXGB(fold_name, x_test,y_test,x_train, y_train,fold)
            self.processSVR(fold_name,x_test, y_test,x_train, y_train,fold)
            self.processRFR(fold_name,x_test, y_test,x_train, y_train,fold)
            self.processKNR(fold_name,x_test, y_test,x_train, y_train,fold)
            self.processLightGBM(fold_name,x_test, y_test,x_train, y_train,fold)
            self.processMPLRegressor(fold_name,x_test, y_test,x_train, y_train,fold)
            #comparation models
            #results = pd.DataFrame({'Model': ['Light GBM', 'Support Vector Machines Regression', 'Random Forest Regressor','Decision Tree Regression','Gradient Boosting Regression', 'MPLRegressor','KNN'],'Score': [acc_lgb, acc_svr, acc_rfr, acc_dtr, acc_gbr, acc_perceptron,acc_knn]})
            #result_df = results.sort_values(by='Score', ascending=False)
            #result_df = result_df.set_index('Score')
            #result_df.head(10)

     # AdaBoostRegressor
     def processLM(self,fold_name,x_test, y_test,x_train, y_train,fold):
           start_time = time.time()
           lm = LinearRegression()
           lm.fit(x_train,y_train)
           lm_pred = lm.predict(x_test)
           lm_pred= lm_pred.reshape(-1,1)
           r2_lm,mae_lm,mse_lm,rmse_lm,mape_lm,coef_lm, r2_adj=self.show_metric(fold_name, 'Linear Regression',lm,lm_pred, y_test,x_train, y_train)
           elapsed_time = time.time() - start_time
           self.add_value_vet(mae_lm,mse_lm,rmse_lm,mape_lm,coef_lm,r2_lm, r2_adj,lm_pred, y_test, elapsed_time,0,fold)


     ########################3Gradient Boosting Regression
     def processGBR(self,fold_name, x_test,y_test,x_train, y_train,fold):
          start_time = time.time()
          params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,              'learning_rate': 0.01, 'loss': 'ls'}
          clf = ensemble.GradientBoostingRegressor(**params)
          clf.fit(x_train, y_train)
          clf_pred=clf.predict(x_test)
          clf_pred= clf_pred.reshape(-1,1)
          r2_clf,mae_clf,mse_clf,rmse_clf,mape_clf,coef_clf, r2_adj=self.show_metric(fold_name, 'Gradient Boosting Regression',clf,clf_pred, y_test,x_train, y_train)
          elapsed_time = time.time() - start_time
          self.add_value_vet(mae_clf,mse_clf,rmse_clf,mape_clf,coef_clf,r2_clf, r2_adj,clf_pred, y_test, elapsed_time,1,fold)

     ############XGBRegressor
     def processXGB(self,fold_name,x_test, y_test,x_train, y_train,fold):
         start_time = time.time()
         xgbreg = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
         xgbreg.fit(x_train, y_train)
         xgb_pred = xgbreg.predict(x_test) 
         xgb_pred= xgb_pred.reshape(-1,1)
         r2_xgb,mae_xgb,mse_xgb,rmse_xgb,mape_xgb,coef_xgb, r2_adj=self.show_metric(fold_name, 'XGB Regressor',xgbreg,xgb_pred, y_test,x_train, y_train)
         elapsed_time = time.time() - start_time
         self.add_value_vet(mae_xgb,mse_xgb,rmse_xgb,mape_xgb,coef_xgb,r2_xgb, r2_adj,xgb_pred, y_test, elapsed_time,2,fold)

     ################SVM Regression
     def processSVR(self,fold_name,x_test, y_test,x_train, y_train,fold):
         start_time = time.time()
         #svr = GridSearchCV(SVR(kernel='poly', gamma=0.1),param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
         svr = SVR(kernel = 'rbf')
         svr.fit(x_train, y_train)
         svr_pred = svr.predict(x_test)
         svr_pred= svr_pred.reshape(-1,1)
         r2_svr,mae_svr,mse_svr,rmse_svr,mape_svr,coef_svr, r2_adj=self.show_metric(fold_name, 'SVM Regression',svr,svr_pred, y_test,x_train, y_train)
         elapsed_time = time.time() - start_time
         self.add_value_vet(mae_svr,mse_svr,rmse_svr,mape_svr,coef_svr,r2_svr, r2_adj,svr_pred, y_test, elapsed_time,3,fold)

     ###############Random Forest Regressor
     def processRFR(self,fold_name,x_test, y_test,x_train, y_train,fold):
         start_time = time.time()
         rfr = RandomForestRegressor(n_estimators = 1000, random_state = 0)
         rfr.fit(x_train, y_train)
         rfr_pred= rfr.predict(x_test)
         rfr_pred = rfr_pred.reshape(-1,1)
         r2_rfr,mae_rfr,mse_rfr,rmse_rfr,mape_rfr,coef_rfr, r2_adj=self.show_metric(fold_name, 'Random Forest Regressor',rfr,rfr_pred, y_test,x_train, y_train)
         elapsed_time = time.time() - start_time
         self.add_value_vet(mae_rfr,mse_rfr,rmse_rfr,mape_rfr,coef_rfr,r2_rfr, r2_adj,rfr_pred, y_test, elapsed_time,4,fold)

     # KNRegressor
     def processKNR(self,fold_name,x_test, y_test,x_train, y_train,fold):
         start_time = time.time()
         knr = KNeighborsRegressor(n_neighbors=5)
         #lab_enc = preprocessing.LabelEncoder()
         #y_train = lab_enc.fit_transform(y_train)
         knr.fit(x_train,y_train)
         knr_pred = knr.predict(x_test)
         r2_knr,mae_knr,mse_knr,rmse_knr,mape_knr,coef_knr, r2_adj=self.show_metric(fold_name, 'KNRegressor',knr,knr_pred, y_test,x_train, y_train)
         elapsed_time = time.time() - start_time 
         self.add_value_vet(mae_knr,mse_knr,rmse_knr,mape_knr,coef_knr,r2_knr, r2_adj,knr_pred, y_test, elapsed_time,5,fold)

     #Light GBM
     def processLightGBM(self,fold_name,x_test, y_test,x_train, y_train,fold):
          start_time = time.time()
          model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                   learning_rate=0.01, n_estimators=1000,
                                   max_bin = 55, bagging_fraction = 0.8,
                                   bagging_freq = 5, feature_fraction = 0.2319,
                                   feature_fraction_seed=9, bagging_seed=9,
                                   min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
          model_lgb.fit(x_train,y_train)
          lgb_pred = model_lgb.predict(x_test)
          lgb_pred = lgb_pred.reshape(-1,1)
          r2_lgb,mae_lgb,mse_lgb,rmse_lgb,mape_lgb,coef_lgb, r2_adj=self.show_metric(fold_name, 'Light GBM',model_lgb,lgb_pred, y_test,x_train, y_train)
          elapsed_time = time.time() - start_time
          self.add_value_vet(mae_lgb,mse_lgb,rmse_lgb,mape_lgb,coef_lgb,r2_lgb, r2_adj,lgb_pred, y_test, elapsed_time,6,fold)

     # MPLRegressor
     def processMPLRegressor(self,fold_name,x_test, y_test,x_train, y_train,fold):
         start_time = time.time()

         perceptron = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                  learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                  random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                  early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
         #lab_enc = preprocessing.LabelEncoder()
         #y_train = perceptron.fit_transform(y_train)
         perceptron.fit(x_train,y_train)
         perceptron_pred = perceptron.predict(x_test)
         r2_perceptron,mae_perceptron,mse_perceptron,rmse_perceptron,mape_perceptron,coef_perceptron, r2_adj=self.show_metric(fold_name, 'MPLRegressor',perceptron,perceptron_pred, y_test,x_train, y_train)
         elapsed_time = time.time() - start_time
         self.add_value_vet(mae_perceptron,mse_perceptron,rmse_perceptron,mape_perceptron,coef_perceptron,r2_perceptron, r2_adj,perceptron_pred, y_test, elapsed_time,7,fold)

     def  add_value_vet(self,mae,mse,rmse,mape,coef,r2, r2_adj,pred, test, time, posi_model,fold):
        pred=self.sc_y.inverse_transform(pred, copy = True)
        test=self.sc_y.inverse_transform(test, copy = True)
        self.vet_fold[posi_model].add_data(mae,mse,rmse,mape,coef,r2,pred, test, time, r2_adj)
    
     def show_metric(self,fold_name, name_model, model,model_pred, y_test,x_train, y_train):
         model_pred=self.sc_y.inverse_transform(model_pred, copy = True)
         y_test=self.sc_y.inverse_transform(y_test, copy = True)
         print(name_model+'\n')
         
         mae=metrics.mean_absolute_error(y_test, model_pred)
         print('MAE: ', mae)
         mse= metrics.mean_squared_error(y_test, model_pred)
         print('MSE: ', mse )
         rmse= np.sqrt(metrics.mean_squared_error(y_test, model_pred))
         print('RMSE: ', rmse)
         #mape= self.mean_absolute_percentage_error(y_test, model_pred)
         mape= (np.mean(np.abs(self.percentage_error(np.array(y_test),np.array(model_pred)))))*100
         print('MAPE: ', mape)
         # The coefficients
         
         #print('Coefficients: ', model.coef_)
         coefficient=0#model.coef_
         # The coefficient of determination: 1 is perfect prediction
         r2 = model.score(x_train, y_train)
         print('R2 Score: %.3f' % r2)
         r2_adj= self.adj_r2_score(r2, y_test,x_train)
         print('R2 Adjust: %.3f' % r2)
         #https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py
         #https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
         #write in log
         self.write_log('\n\n\n#####################################################\n') 
         self.write_log('\n\n'+name_model+'\n') 
         self.write_log('MAE: '+str(mae))
         self.write_log('MSE: '+str(mse))
         self.write_log('RMSE: '+str(rmse))
         self.write_log('MAPE: '+str(mape))
         #self.write_log('Coefficients: '+ ';'.join(str(e) for e in coefficient))
         self.write_log('R2_Score: '+str(r2))
         self.write_log('R2_ Adj: '+str(r2_adj))
         plt.title(name_model)
         plt.figure(figsize=(15,8))
         plt.scatter(y_test,model_pred, c='orange')
         plt.xlabel('Y Test')
         #plt.show()
         plt.savefig(fold_name+'_'+name_model+'_intercept.png')
         plt.title(name_model)
         plt.figure(figsize=(16,8))
         plt.plot(y_test,label ='Test')
         plt.plot(model_pred, label = 'predict')
         #plt.show()
         plt.savefig(fold_name+'_'+name_model+'_prediction_test.png')
         return r2,mae,mse,rmse,mape,coefficient,r2_adj

     def write_log(self,log):
         self.log_file.write(log+'\n')

     def mean_absolute_percentage_error(self,y_true, y_pred): 
         y_true, y_pred = np.array(y_true), np.array(y_pred)
         return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


     def plot_regression_results(self,ax, y_true, y_pred, title, scores, elapsed_time):
         """Scatter plot of the predicted vs true targets."""
         ax.plot([min(y_true)-10, max(y_true)+10],
                 [min(y_true)-10, max(y_true)+10],
                 '--r', linewidth=2)
         ax.scatter(y_true, y_pred, alpha=0.2,color='red')
         import seaborn as sns
         #sns.regplot(x=y_true, y= y_pred.Weight, color='blue', marker='+')
         #ax.scatter(y_pred, y_true, alpha=0.2,color='blue')
         ax.spines['top'].set_visible(False)
         ax.spines['right'].set_visible(False)
         ax.get_xaxis().tick_bottom()
         ax.get_yaxis().tick_left()
         ax.spines['left'].set_position(('outward', 2))
         ax.spines['bottom'].set_position(('outward', 2))
         ax.set_xlim([min(y_true)-5, max(y_true)+5])
         ax.set_ylim([min(y_true)-5, max(y_true)+5])
         ax.set_xlabel('Measured')
         ax.set_ylabel('Predicted')
         extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                               edgecolor='none', linewidth=0)
         ax.legend([extra], [scores], loc='upper left')
         title = title + '\n Evaluation in {:.3f} seconds'.format(elapsed_time)
         ax.set_title(title)
         ax.grid(True)
     def vet_text(self,vet):
        txt_file=""
        for line in vet:
           txt_file+=";"+str(line) 
        return txt_file
     def vet_sum(self,vet):
        sum_vet=0
        for line in vet:
           sum_vet+=line 
        return sum_vet		

     def percentage_error(self,actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            if actual[j] != 0:
               res[j] = ((actual[j] - predicted[j]) / np.abs(actual[j]))
            else:
               res[j] = (predicted[j] / np.mean(actual))
        return res
     def adj_r2_score(self,r2, y_true, X):
        """Adjusted R square — put fitted linear model, y value, estimated y value in order"""
        from sklearn import metrics
        adj = 1 - (1-r2)*(len(y_true)-1)/(len(y_true)-X.shape[1]-1)

        return adj

if __name__ == "__main__":
    ml=MLSklean()
    for i in range(8):
       train_file="f"+str(i+1)+"/train/sheep.arff"
       test_file="f"+str(i+1)+"/val/sheep.arff"
       ml.run(train_file, test_file,"Fold_"+str(i+1),i)
   

    vet_title=["Linear Regressor","Gradient Boosting Regression","XGBoost Regressor","Support Vector Regression ", "Random Forest Regressor","K Neighbors Regressor","Light GBM","MPL Regressor"]
    for i in range(8):
            obj=ml.vet_fold[i]
            ml.write_log("\n\n\n\n #########################################################\n")
            ml.write_log("MODEL: "  + vet_title[i]+" "+ str(i))     
            ml.write_log('MAE: '+ml.vet_text(obj.vet_mae))
            ml.write_log('MSE: '+ml.vet_text(obj.vet_mse))
            ml.write_log('RMSE: '+ml.vet_text(obj.vet_rmse))
            ml.write_log('MAPE: '+ml.vet_text(obj.vet_mape))
            ml.write_log('R2: '+ml.vet_text(obj.vet_r2))
            ml.write_log('R2 Ajust: '+ml.vet_text(obj.vet_r2_adj))
            ml.write_log('TIME: '+ml.vet_text(obj.vet_time))
            			
            vec_pred=[]
            print(obj.vet_pred)
            for j, vec in enumerate(obj.vet_pred):
                for jj,vec2 in enumerate(vec):
                    vec_pred.append(vec2.copy())
                
            vec_test=[]
            for f, vec in enumerate(obj.vet_test):
                for ff,vec2 in enumerate(vec):
                    vec_test.append(vec2.copy())
                
            print(obj.vet_test)
            ml.write_log('Pred:('+str(len(vec_pred))+') '+ml.vet_text(vec_pred))
            ml.write_log('Test:('+str(len(vec_test))+') '+ml.vet_text(vec_test))
            ml.write_log('MAE Média: '+str(ml.vet_sum(obj.vet_mae.copy())/8)+' +- '+str(np.std(obj.vet_mae.copy())))
            ml.write_log('MSE Média: '+str(ml.vet_sum(obj.vet_mse.copy())/8)+' +- '+str(np.std(obj.vet_mse.copy())))
            ml.write_log('RMSE Média: '+str(ml.vet_sum(obj.vet_rmse.copy())/8)+' +- '+str(np.std(obj.vet_rmse.copy())))
            ml.write_log('MAPE Média: '+str((ml.vet_sum(obj.vet_mape.copy())/8))+' +- '+str(np.std(obj.vet_mape.copy())))
            ml.write_log('R2 Score Média: '+str(ml.vet_sum(obj.vet_r2)/8)+' +- '+str(np.std(obj.vet_r2.copy())))
            ml.write_log('R2 Ajust Média: '+str(ml.vet_sum(obj.vet_r2_adj)/8)+' +- '+str(np.std(obj.vet_r2_adj.copy())))
            ml.write_log('Tempo médio das dobras: '+str(np.mean(obj.vet_time.copy()))+' +- '+str(np.std(obj.vet_time.copy())))
            ml.write_log('Explained Varience Score Total: '+str(explained_variance_score(vec_test, vec_pred)))
            plt.title(vet_title[i])
            plt.figure(figsize=(16,8))
            plt.plot(vec_test,label ='Test')
            plt.plot(vec_pred, label = 'predict')
            #plt.show()
            plt.savefig(vet_title[i]+'_total_prediction_test.png')
            #plot		
            fig, axs = plt.subplots(1, 1, figsize=(10, 8))			
            axs = np.ravel(axs)
            ml.plot_regression_results(
                   axs[0], vec_test.copy(), vec_pred.copy(), 
                   vet_title[i],
                   (r'$R^2={:.3f} (\pm {:.4f}$)' + '\n' + r'$MAE={:.3f} (\pm {:.4f}$)')
                   .format(np.mean(obj.vet_r2.copy()),
                   np.std(obj.vet_r2.copy()),
                   np.mean(obj.vet_mae.copy()),
                   np.std(obj.vet_mae.copy())),
                   ml.vet_sum(obj.vet_time.copy()))		
            plt.suptitle('Single predictors versus stacked predictors')
            plt.tight_layout()
            plt.subplots_adjust(top=0.80)
            plt.savefig(vet_title[i]+'_predictvsmesured.png')	 	   
			
