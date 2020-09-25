import math  
from scipy.io import arff
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import os
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from time import time
import matplotlib.pyplot as plt 
import seaborn as sns 
class PrintData():
    ''' Init with parameter file of log open '''   
    def __init__(self,log_file):
       self.log_file=log_file
    def print_double_line(self):
        self.log_file.write('***************************************************************************************************\n')
        self.log_file.write('***************************************************************************************************\n\n\n')	
    '''Vector with same size when fist vector have names of models and second values of R2 of models 
       Param: vector list_name
       Param: vector list_result
    '''
    def print_all_result_r2(self,list_name,list_result):
       for i, model in enumerate(list_name):
          self.log_file.write('All R2 for '+str(model)+': '+str(list_result[i])+'\n') 
    def print_fold_result(self, regressor_name,i,train_time,test_time,mae,mape,mse,rmse,r,r2, explain, maxe, mea):
       self.log_file.write(regressor_name+' \n')
       self.log_file.write('Traning time: '+str(train_time["{0}".format(i)])+'\n')
       self.log_file.write('Test time: '+str(test_time["{0}".format(i)])+'\n')
       self.log_file.write('MAE: '+str(mae["{0}".format(i)])+'\n')
       self.log_file.write('MAPE: '+str(mape["{0}".format(i)])+'\n')
       self.log_file.write('MSE: '+str(mse["{0}".format(i)] )+'\n')
       self.log_file.write('RMSE: '+str(rmse["{0}".format(i)] )+'\n')
       self.log_file.write('R: '+str( r["{0}".format(i)])+'\n')
       self.log_file.write('R^2: '+str(r2["{0}".format(i)] )+'%'+'\n')
       self.log_file.write('Explained variance regression score: '+str(explain["{0}".format(i)] )+'\n')
       self.log_file.write('Max error:  '+str(maxe["{0}".format(i)] )+'\n')
       self.log_file.write('Median absolute error: '+str(mea["{0}".format(i)] )+'\n\n\n')
       self.log_file.write('***************************************************************************************************\n\n\n')
    	
    def print_average_total(self,qtd_fold,train_time_sum,test_time_sum,mae_sum,mape_sum,mse_sum,rmse_sum,r_sum,r2_sum,all_r2_random,explain_sum,maxe_sum,mea_sum):
       self.log_file.write('Average Traning time: '+str(train_time_sum/qtd_fold)+'\n')
       self.log_file.write('Average Test time: '+str(test_time_sum/qtd_fold)+'\n')    
       self.log_file.write('Average MAE: '+str(mae_sum/qtd_fold)+'\n')
       self.log_file.write('Average MAPE: '+str(mape_sum/qtd_fold)+'\n')
       self.log_file.write('Average MSE: '+str(mse_sum/qtd_fold)+'\n')
       self.log_file.write('Average RMSE: '+str(rmse_sum/qtd_fold)+'\n')
       self.log_file.write('Average R: '+str(r_sum/qtd_fold)+'\n')
       self.log_file.write('Average R2: '+str(r2_sum/qtd_fold)+'\n')
       self.log_file.write('Mean: '+str(np.mean(all_r2_random))+'\n')
       self.log_file.write('Standard Deviation: '+str(np.std(all_r2_random))+'\n')
       self.log_file.write('Average Explained variance regression score: '+str(explain_sum/qtd_fold)+'\n')
       self.log_file.write('Average Max error: '+str(maxe_sum/qtd_fold)+'\n')
       self.log_file.write('Average Median absolute error: '+str(mea_sum/qtd_fold)+'\n')
       self.log_file.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'+'\n\n')
       	
class Util():
    ''' Init with parameter file of log open '''   
           
    def calc_metric(self,model,scaler_y,i,x_test,y_test,y_pred,mae,mape,mse,rmse,acc,r,r2,explain,maxe,mea):
        y_pred_real= scaler_y["{0}".format(i)].inverse_transform(y_pred["{0}".format(i)],copy = True) 
        y_test_real= scaler_y["{0}".format(i)].inverse_transform(y_test["{0}".format(i)],copy = True) 		
        mae["{0}".format(i)] =metrics.mean_absolute_error(y_test_real, y_pred_real)
        mape["{0}".format(i)] =(np.mean(np.abs(percentage_error(y_test_real,y_pred_real))))
        mse["{0}".format(i)] =metrics.mean_squared_error(y_test_real, y_pred_real)
        rmse["{0}".format(i)] =np.sqrt(metrics.mean_squared_error(y_test_real, y_pred_real))
        acc["{0}".format(i)]= round(model["{0}".format(i)].score(x_test["{0}".format(i)], y_test["{0}".format(i)]) * 100, 2)
        if acc["{0}".format(i)] <=0:acc["{0}".format(i)]=-acc["{0}".format(i)]
        r["{0}".format(i)] =np.sqrt(acc["{0}".format(i)])
        r2["{0}".format(i)] =round(acc["{0}".format(i)],2)
        explain["{0}".format(i)] =metrics.explained_variance_score(y_test_real,y_pred_real)
        maxe["{0}".format(i)] =metrics.max_error(y_test_real,y_pred_real)
        mea["{0}".format(i)] = metrics.median_absolute_error(y_test_real,y_pred_real)    
    
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
QTD_FOLD=4    
t1='resultsMAPE.txt'

if os.path.exists(t1):
    os.remove(t1)

 
arq1 = open(t1,'a') 
'''Declaration of classes'''
print_data=PrintData(arq1)
util=Util()

data_train={}
train={}
data_test={}
test={}

X_t={}
y_tes={}
X_tes={}
y_t={}
X_train={}
y_train={}
X_test={}
y_test={}
sc_X={}
sc_y={}
X_train_MAPE={}
y_train_MAPE={}
X_test_MAPE={}
y_test_MAPE={}

    
dtreg={}
dtr_pred={}
acc_dtr={}
mae_dtr={}
mape_dtr={}
mse_dtr={}
rmse_dtr={}
r_dtr={}
r2_dtr={}
explain_dtr={}
maxe_dtr={}
mea_dtr={} 
dtr_t0={}
dtr_train_time={}
dtr_t1={}
dtr_test_time={}
dtr_train_time_sum={}    
dtr_test_time_sum={}

    
random_forest={}
random_forest_pred={}
acc_random={}
mae_random={}
mape_random={}
mse_random={}
rmse_random={}
r_random={}
r2_random={}
explain_random={}
maxe_random={}
mea_random={} 
random_t0={}
random_train_time={}
random_t1={}
random_test_time={}
random_train_time_sum={}    
random_test_time_sum={}
    
    
    
knn={}
knn_pred={}
acc_knn={}
mae_knn={}
mape_knn={}
mse_knn={}
rmse_knn={}
r_knn={}
r2_knn={}
explain_knn={}
maxe_knn={}
mea_knn={} 
knn_t0={}
knn_train_time={}
knn_t1={}
knn_test_time={}
knn_train_time_sum={}    
knn_test_time_sum={}



reg={}
mlp_pred={}
mlp_acc={}
mae_mlp={}
mape_mlp={}
mse_mlp={}
rmse_mlp={}
r_mlp={}
r2_mlp={}
explain_mlp={}
maxe_mlp={}
mea_mlp={}    
mlp_t0={}
mlp_train_time={}
mlp_t1={}
mlp_test_time={}
mlp_train_time_sum={}    
mlp_test_time_sum={}

    
svm_={}
svm_pred={}
acc_svm={}
mae_svm={}
mape_svm={}
mse_svm={}
rmse_svm={}
r_svm={}
r2_svm={}
explain_svm={}
maxe_svm={}
mea_svm={}
svm_t0={}
svm_train_time={}
svm_t1={}
svm_test_time={}
svm_train_time_sum={}    
svm_test_time_sum={}

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = ((actual[j] - predicted[j]) / np.abs(actual[j]))
        else:
            res[j] = (predicted[j] / np.mean(actual))
    return res

for i in range(QTD_FOLD):

    arq1.write('************************************ FOLD NUMBER: %s'%(i+1)+' ****************************************\n')
    arq1.write('***************************************************************************************************\n\n\n')

    #carrega os dados de treinamento
    data_train["{0}".format(i)]= arff.loadarff('f%s'%(i+1)+'/train/sheep.arff') #training.arff
    train["{0}".format(i)] = pd.DataFrame(data_train["{0}".format(i)][0])

    #carrega os dados de teste
    data_test["{0}".format(i)] = arff.loadarff('f%s'%(i+1)+'/val/sheep.arff')# test.arff
    test["{0}".format(i)] = pd.DataFrame(data_test["{0}".format(i)][0])

    #sns.distplot(test["{0}".format(i)]['weight'] , fit=norm);
    # sns.distplot(train["{0}".format(i)] , fit=norm);
    # plt.show()
    
    #arq1.write(str(train["{0}".format(i)]))

    #*************************************************************************
    # we are going to scale to data

    X_t["{0}".format(i)] = train["{0}".format(i)].drop("Weight", axis=1)
    y_t["{0}".format(i)] = train["{0}".format(i)]["Weight"]#*100 converter para int
    X_tes["{0}".format(i)] = test["{0}".format(i)].drop('Weight', axis=1)
    y_tes["{0}".format(i)] = test["{0}".format(i)]['Weight'] #*100 converter para int
    y_t["{0}".format(i)]= y_t["{0}".format(i)].values.reshape(-1,1)
    y_tes["{0}".format(i)]= y_tes["{0}".format(i)].values.reshape(-1,1)
    

        
    sc_X["{0}".format(i)] = StandardScaler()
    sc_y["{0}".format(i)] = sc_X["{0}".format(i)] 
    X_train["{0}".format(i)] = sc_X["{0}".format(i)].fit_transform(X_t["{0}".format(i)])
    X_test["{0}".format(i)] = sc_X["{0}".format(i)].fit_transform(X_tes["{0}".format(i)])
    y_train["{0}".format(i)] = sc_y["{0}".format(i)].fit_transform(y_t["{0}".format(i)])
    y_test["{0}".format(i)] = sc_y["{0}".format(i)].fit_transform(y_tes["{0}".format(i)])
    
    ##Converter para int e remover .00
    # y_train["{0}".format(i)]=y_train["{0}".format(i)].astype('int') 
    # y_test["{0}".format(i)]=y_test["{0}".format(i)].astype('int')



    # sns.distplot(X_train["{0}".format(i)], fit=norm, color='green', fit_kws={"color":"blue"} );
    # plt.show()
    # sns.distplot(X_train["{0}".format(i)] , fit=norm,color='green', fit_kws={"color":"blue"} );
    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(X_train["{0}".format(i)])
    # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    # #Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                # loc='best')
    # plt.ylabel('Frequency')
    # plt.title('Dataset distribution')
    # #Get also the QQ-plot
    # fig = plt.figure()
    # #res = stats.probplot(X_train, plot=plt)
    # plt.show() 

    # X_train_MAPE["{0}".format(i)] = train["{0}".format(i)].drop("weight", axis=1)
    # y_train_MAPE["{0}".format(i)] = train["{0}".format(i)]["weight"]*100 #*100 converter para int
    # X_test_MAPE["{0}".format(i)] = test["{0}".format(i)].drop('weight', axis=1)
    # y_test_MAPE["{0}".format(i)] = test["{0}".format(i)]['weight']*100 #*100 converter para int


    

    #*************************************************************************
    ###################Decision Tree Regression
    
    dtreg["{0}".format(i)] = DecisionTreeRegressor(random_state = 0)
    dtr_t0["{0}".format(i)]  = time()
    dtreg["{0}".format(i)].fit(X_train["{0}".format(i)], y_train["{0}".format(i)])
    dtr_train_time["{0}".format(i)]  = time() - dtr_t0["{0}".format(i)] 
    dtr_t1["{0}".format(i)]  = time()
    dtr_pred["{0}".format(i)] = dtreg["{0}".format(i)].predict(X_test["{0}".format(i)])
    dtr_test_time["{0}".format(i)]  = time() - dtr_t1["{0}".format(i)] 
   
    util.calc_metric(dtreg,sc_y,i,X_test,y_test,dtr_pred,mae_dtr,mape_dtr,mse_dtr,rmse_dtr,acc_dtr,r_dtr,r2_dtr,explain_dtr,maxe_dtr,mea_dtr)
    
    print_data.print_fold_result("Decision Tree Regression",i,dtr_train_time,dtr_test_time,mae_dtr,mape_dtr,mse_dtr,rmse_dtr,r_dtr,r2_dtr, explain_dtr, maxe_dtr, mea_dtr)
	

    #*************************************************************************
    #################### Random Forest

    
    random_forest["{0}".format(i)] = RandomForestRegressor(n_estimators=100)
    random_t0["{0}".format(i)]  = time()
    random_forest["{0}".format(i)].fit(X_train["{0}".format(i)], y_train["{0}".format(i)].ravel())
    random_train_time["{0}".format(i)]  = time() - random_t0["{0}".format(i)] 
    random_t1["{0}".format(i)]  = time()
    random_forest_pred["{0}".format(i)] = random_forest["{0}".format(i)].predict(X_test["{0}".format(i)])
    random_test_time["{0}".format(i)]  = time() - random_t1["{0}".format(i)] 
    
    util.calc_metric( random_forest,sc_y,i,X_test,y_test,random_forest_pred,mae_random,mape_random,mse_random,rmse_random,acc_random,r_random,r2_random,explain_random,maxe_random,mea_random)


    print_data.print_fold_result("Random Forest",i,random_train_time,random_test_time,mae_random,mape_random,mse_random,rmse_random,r_random,r2_random, explain_random, maxe_random, mea_random)


    #*************************************************************************
    ##################### KNN
    
    knn["{0}".format(i)] = KNeighborsRegressor(n_neighbors = 3)
    knn_t0["{0}".format(i)]  = time()
    knn["{0}".format(i)].fit(X_train["{0}".format(i)], y_train["{0}".format(i)].ravel())
    knn_train_time["{0}".format(i)]  = time() - knn_t0["{0}".format(i)] 
    knn_t1["{0}".format(i)]  = time()
    knn_pred["{0}".format(i)] = knn["{0}".format(i)].predict(X_test["{0}".format(i)])
    knn_test_time["{0}".format(i)]  = time() - knn_t1["{0}".format(i)] 
    
    util.calc_metric( knn,sc_y,i,X_test,y_test,knn_pred,mae_knn,mape_knn,mse_knn,rmse_knn,acc_knn,r_knn,r2_knn,explain_knn,maxe_knn,mea_knn)
	 
    print_data.print_fold_result("KNN",i,knn_train_time,knn_test_time,mae_knn,mape_knn,mse_knn,rmse_knn,r_knn,r2_knn, explain_knn, maxe_knn, mea_knn)	


    #*************************************************************************
    #####################  MLPRegressor
    
    reg["{0}".format(i)] = MLPRegressor(solver='lbfgs',
                           alpha=1e-5,
                           hidden_layer_sizes=(37, 37),
                           random_state=1,
                           activation="tanh",
                           max_iter=1000)
    mlp_t0["{0}".format(i)]  = time()                      
    reg["{0}".format(i)].fit(X_train["{0}".format(i)], y_train["{0}".format(i)].ravel())
    mlp_train_time["{0}".format(i)]  = time() - mlp_t0["{0}".format(i)] 
    mlp_t1["{0}".format(i)]  = time()
    mlp_pred["{0}".format(i)] = reg["{0}".format(i)].predict(X_test["{0}".format(i)])
    mlp_test_time["{0}".format(i)]  = time() - mlp_t1["{0}".format(i)]  
    
    util.calc_metric( reg,sc_y,i,X_test,y_test,mlp_pred,mae_mlp,mape_mlp,mse_mlp,rmse_mlp,mlp_acc,r_mlp,r2_mlp,explain_mlp,maxe_mlp,mea_mlp)
    
    print_data.print_fold_result("MLP Regressor",i,mlp_train_time,mlp_test_time,mae_mlp,mape_mlp,mse_mlp,rmse_mlp,r_mlp,r2_mlp, explain_mlp, maxe_mlp, mea_mlp)	


    #*************************************************************************
    #####################   Suporte Vector Regression
    
    svm_["{0}".format(i)] = svm.SVR()#max_iter=3000)#abaixo de 3000 dá warning
    svm_t0["{0}".format(i)]  = time()
    svm_["{0}".format(i)].fit(X_train["{0}".format(i)], y_train["{0}".format(i)].ravel())
    svm_train_time["{0}".format(i)]  = time() - svm_t0["{0}".format(i)] 
    svm_t1["{0}".format(i)]  = time()
    svm_pred["{0}".format(i)] = svm_["{0}".format(i)].predict(X_test["{0}".format(i)])
    svm_test_time["{0}".format(i)]  = time() - svm_t1["{0}".format(i)] 
    
    util.calc_metric(svm_,sc_y,i,X_test,y_test,svm_pred,mae_svm,mape_svm,mse_svm,rmse_svm,acc_svm,r_svm,r2_svm,explain_svm,maxe_svm,mea_svm)

    print_data.print_fold_result("Suporte Vector Regression",i,svm_train_time,svm_test_time,mae_svm,mape_svm,mse_svm,rmse_svm,r_svm,r2_svm, explain_svm, maxe_svm, mea_svm)	
        
    print_data.print_double_line()
    
    #print(mape_dtr,'\n')
    
    
    
 
arq1.write('***************************************************************************************************\n')
arq1.write('***************************************************************************************************\n')
arq1.write('***************************************************************************************************\n')
arq1.write('***************************************************************************************************\n\n\n')
arq1.write('                        - SUMMARY - Average Values after Cross-Validation \n')   

mape_dtr_sum,mape_random_sum,mape_knn_sum,mape_mlp_sum,mape_svm_sum=(0,)*5

dtr_train_time_sum,dtr_test_time_sum,random_train_time_sum,random_test_time_sum,knn_train_time_sum,knn_test_time_sum,mlp_train_time_sum,mlp_test_time_sum,svm_train_time_sum,svm_test_time_sum=(0,)*10

all_r2_dtr=[]
all_r2_random=[]
all_r2_knn=[]
all_r2_mlp=[]
all_r2_svm=[]

arq1.write('---------------------- Decision Tree Regression \n')
mae_dtr_sum,mse_dtr_sum,rmse_dtr_sum,r_dtr_sum,r2_dtr_sum,explain_dtr_sum,maxe_dtr_sum,mea_dtr_sum = (0,)*8
for i in range(QTD_FOLD):
    dtr_train_time_sum=dtr_train_time_sum+dtr_train_time["{0}".format(i)]
    dtr_test_time_sum=dtr_test_time_sum+dtr_test_time["{0}".format(i)]
    mae_dtr_sum=mae_dtr_sum+mae_dtr["{0}".format(i)]
    mape_dtr_sum=mape_dtr_sum+mape_dtr["{0}".format(i)]
    mse_dtr_sum=mse_dtr_sum+mse_dtr["{0}".format(i)] 
    rmse_dtr_sum=rmse_dtr_sum+rmse_dtr["{0}".format(i)] 
    r_dtr_sum=r_dtr_sum+r_dtr["{0}".format(i)] 
    r2_dtr_sum=r2_dtr_sum+r2_dtr["{0}".format(i)]
    all_r2_dtr.append(float(r2_dtr["{0}".format(i)]))
    explain_dtr_sum=explain_dtr_sum+explain_dtr["{0}".format(i)] 
    maxe_dtr_sum=maxe_dtr_sum+maxe_dtr["{0}".format(i)]
    mea_dtr_sum=mea_dtr_sum+mea_dtr["{0}".format(i)]

print_data.print_average_total(QTD_FOLD,dtr_train_time_sum,dtr_test_time_sum,mae_dtr_sum,mape_dtr_sum,mse_dtr_sum,rmse_dtr_sum,r_dtr_sum,r2_dtr_sum,all_r2_dtr,explain_dtr_sum,maxe_dtr_sum,mea_dtr_sum)     





arq1.write('---------------------- Random Forest \n')
mae_random_sum,mse_random_sum,rmse_random_sum,r_random_sum,r2_random_sum,explain_random_sum,maxe_random_sum,mea_random_sum = (0,)*8

for i in range(QTD_FOLD):
    random_train_time_sum=random_train_time_sum+random_train_time["{0}".format(i)]
    random_test_time_sum=random_test_time_sum+random_test_time["{0}".format(i)]
    mae_random_sum=mae_random_sum+mae_random["{0}".format(i)]
    mape_random_sum=mape_random_sum+mape_random["{0}".format(i)]
    mse_random_sum=mse_random_sum+mse_random["{0}".format(i)] 
    rmse_random_sum=rmse_random_sum+rmse_random["{0}".format(i)] 
    r_random_sum=r_random_sum+r_random["{0}".format(i)] 
    r2_random_sum=r2_random_sum+r2_random["{0}".format(i)]
    all_r2_random.append(float(r2_random["{0}".format(i)]))
    explain_random_sum=explain_random_sum+explain_random["{0}".format(i)] 
    maxe_random_sum=maxe_random_sum+maxe_random["{0}".format(i)]
    mea_random_sum=mea_random_sum+mea_random["{0}".format(i)]

print_data.print_average_total(QTD_FOLD,random_train_time_sum,random_test_time_sum,mae_random_sum,mape_random_sum,mse_random_sum,rmse_random_sum,r_random_sum,r2_random_sum,all_r2_random,explain_random_sum,maxe_random_sum,mea_random_sum)     



arq1.write('---------------------- KNN \n')
mae_knn_sum,mse_knn_sum,rmse_knn_sum,r_knn_sum,r2_knn_sum,explain_knn_sum,maxe_knn_sum,mea_knn_sum = (0,)*8

for i in range(QTD_FOLD):
    knn_train_time_sum=knn_train_time_sum+knn_train_time["{0}".format(i)]
    knn_test_time_sum=knn_test_time_sum+knn_test_time["{0}".format(i)]
    mae_knn_sum=mae_knn_sum+mae_knn["{0}".format(i)]
    mape_knn_sum=mape_knn_sum+mape_knn["{0}".format(i)]
    mse_knn_sum=mse_knn_sum+mse_knn["{0}".format(i)] 
    rmse_knn_sum=rmse_knn_sum+rmse_knn["{0}".format(i)] 
    r_knn_sum=r_knn_sum+r_knn["{0}".format(i)] 
    r2_knn_sum=r2_knn_sum+r2_knn["{0}".format(i)]
    all_r2_knn.append(float(r2_knn["{0}".format(i)]))
    explain_knn_sum=explain_knn_sum+explain_knn["{0}".format(i)] 
    maxe_knn_sum=maxe_knn_sum+maxe_knn["{0}".format(i)]
    mea_knn_sum=mea_knn_sum+mea_knn["{0}".format(i)]

print_data.print_average_total(QTD_FOLD,knn_train_time_sum,knn_test_time_sum,mae_knn_sum,mape_knn_sum,mse_knn_sum,rmse_knn_sum,r_knn_sum,r2_knn_sum,all_r2_knn,explain_knn_sum,maxe_knn_sum,mea_knn_sum)    


arq1.write('---------------------- MLPRegressor \n')
mae_mlp_sum,mse_mlp_sum,rmse_mlp_sum,r_mlp_sum,r2_mlp_sum,explain_mlp_sum,maxe_mlp_sum,mea_mlp_sum = (0,)*8

for i in range(QTD_FOLD):
    mlp_train_time_sum=mlp_train_time_sum+mlp_train_time["{0}".format(i)]
    mlp_test_time_sum=mlp_test_time_sum+mlp_test_time["{0}".format(i)]
    mae_mlp_sum=mae_mlp_sum+mae_mlp["{0}".format(i)]
    mape_mlp_sum=mape_mlp_sum+mape_mlp["{0}".format(i)]
    mse_mlp_sum=mse_mlp_sum+mse_mlp["{0}".format(i)] 
    rmse_mlp_sum=rmse_mlp_sum+rmse_mlp["{0}".format(i)] 
    r_mlp_sum=r_mlp_sum+r_mlp["{0}".format(i)] 
    r2_mlp_sum=r2_mlp_sum+r2_mlp["{0}".format(i)]
    all_r2_mlp.append(float(r2_mlp["{0}".format(i)]))
    explain_mlp_sum=explain_mlp_sum+explain_mlp["{0}".format(i)] 
    maxe_mlp_sum=maxe_mlp_sum+maxe_mlp["{0}".format(i)]
    mea_mlp_sum=mea_mlp_sum+mea_mlp["{0}".format(i)]

print_data.print_average_total(QTD_FOLD,mlp_train_time_sum,mlp_test_time_sum,mae_mlp_sum,mape_mlp_sum,mse_mlp_sum,rmse_mlp_sum,r_mlp_sum,r2_mlp_sum,all_r2_mlp,explain_mlp_sum,maxe_mlp_sum,mea_mlp_sum)

arq1.write('----------------------  Suporte Vector Regression \n')
mae_svm_sum,mse_svm_sum,rmse_svm_sum,r_svm_sum,r2_svm_sum,explain_svm_sum,maxe_svm_sum,mea_svm_sum = (0,)*8

for i in range(QTD_FOLD):
    svm_train_time_sum=svm_train_time_sum+svm_train_time["{0}".format(i)]
    svm_test_time_sum=svm_test_time_sum+svm_test_time["{0}".format(i)]
    mae_svm_sum=mae_svm_sum+mae_svm["{0}".format(i)]
    mape_svm_sum=mape_svm_sum+mape_svm["{0}".format(i)]
    mse_svm_sum=mse_svm_sum+mse_svm["{0}".format(i)] 
    rmse_svm_sum=rmse_svm_sum+rmse_svm["{0}".format(i)] 
    r_svm_sum=r_svm_sum+r_svm["{0}".format(i)] 
    r2_svm_sum=r2_svm_sum+r2_svm["{0}".format(i)]
    all_r2_svm.append(float(r2_svm["{0}".format(i)]))
    explain_svm_sum=explain_svm_sum+explain_svm["{0}".format(i)] 
    maxe_svm_sum=maxe_svm_sum+maxe_svm["{0}".format(i)]
    mea_svm_sum=mea_svm_sum+mea_svm["{0}".format(i)]

'''Print all average of all techniques'''
print_data.print_average_total(QTD_FOLD,svm_train_time_sum,svm_test_time_sum,mae_svm_sum,mape_svm_sum,mse_svm_sum,rmse_svm_sum,r_svm_sum,r2_svm_sum,all_r2_svm,explain_svm_sum,maxe_svm_sum,mea_svm_sum)



'''Print all total r2 of all techniques'''
print_data.print_all_result_r2(['DTR','RF','KNN','MLP','SVR'],[all_r2_dtr,all_r2_random,all_r2_knn,all_r2_mlp,all_r2_svm])

'''Print all total r2 of all techniques charts'''
all_r2_vec=[all_r2_dtr,all_r2_random,all_r2_knn,all_r2_mlp,all_r2_svm]
all_test_time=[dtr_test_time_sum,random_test_time_sum,knn_test_time_sum,mlp_test_time_sum,svm_test_time_sum]
all_pred=[dtr_pred.copy(),random_forest.copy(),knn_pred.copy(),mlp_pred.copy(),svm_pred.copy()]
all_test=y_test.copy()
for i, title in enumerate(['DTR','RF','KNN','MLP','SVR']):
            vec_pred=[]
            mae_svm["{0}".format(i)]
            for j in range(QTD_FOLD):
                vec_pred.append(sc_y["{0}".format(j)].inverse_transform((all_pred[i])["{0}".format(j)],copy = True)  )      
                       
            vec_test=[]        
            for f in range(QTD_FOLD):
                vec_test.append(sc_y["{0}".format(i)].inverse_transform((all_test["{0}".format(f)]),copy = True) 	)	
                

            #plt.title(title)
            #plt.figure(figsize=(16,8))
            #plt.plot(vec_test,label ='Test')
            #plt.plot(vec_pred, label = 'predict')
            print(vec_test)
            print(vec_pred)
            #plt.show()
            #plt.savefig('Marcio_'+title+'_total_prediction_test.png')
			
            #fig, axs = plt.subplots(1, 1, figsize=(10, 8))			
            #axs = np.ravel(axs)
            #util.plot_regression_results(
            #       axs[0], vec_pred.copy(), vec_pred.copy(), 
            #       title,
            #       ('$R^2={:.3f} (\pm {:.4f}$)' + '\n' )
            #       .format(np.mean(all_r2_vec[i].copy()),
            #       np.std(all_r2_vec[i].copy()), all_test_time[i])		
            #plt.suptitle('Single predictors versus stacked predictors')
            #plt.tight_layout()
            #plt.subplots_adjust(top=0.80)
            #plt.savefig('Marcio_Alg_'+vet_title[i]+'_predictvsmesured.png')	

 
arq1.close()
