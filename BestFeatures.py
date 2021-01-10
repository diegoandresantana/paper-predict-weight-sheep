from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgboost

from sklearn.svm import SVR
from scipy.io import arff
import pandas as pd 


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def adjust_r2(r2, num_examples, num_features):
    coef = (num_examples - 1) / (num_examples - num_features - 1) 
    return 1 - (1 - r2) * coef
if __name__ == "__main__":
    log_file = open('selection_attributes.txt', 'a') #r ou a
    log_file.write('Experiment:\n')
            
    QTD_FOLDS=4

    location_arff="sheepcustomizado.arff"
    dataset= arff.loadarff(location_arff)
    data = pd.DataFrame(dataset[0])
    data.head()



    x=data.drop('Weight', axis=1).copy()
    y=data['Weight'].copy()
    lr = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=0,colsample_bytree=0.3,
                 gamma=10,                 
                 learning_rate=0.1,
                 alpha=10,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=100,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=300) 

    #lr = xgboost.XGBRegressor(colsample_bytree=0.3,
    #             gamma=10,                 
    #             learning_rate=0.1,
    #             alpha=10,
    #             max_depth=20,
    #             min_child_weight=1.5,
    #             n_estimators=1000,                                                                    
    #             reg_alpha=0.75,
    #             reg_lambda=0.45,
    #             subsample=0.6,
    #             seed=300) 

    efs = EFS(lr, n_jobs=12,
          min_features=10,
          max_features=20,
          scoring= 'r2' ,
          cv=4,print_progress=True)

    efs.fit(x, y)
    log_file.write('Best MSE score: %.2f' % efs.best_score_ * (-1))
    log_file.write('\n')
    log_file.write('Best subset:' + str( efs.best_idx_))
    log_file.write('\n')
    log_file.write('Best subset (corresponding names):'+ str( efs.best_feature_names_))
    log_file.write('\n')
    print('Best MSE score: %.2f' % efs.best_score_ * (-1))
    print('Best subset:', efs.best_idx_)
    print('Best subset (corresponding names):', efs.best_feature_names_)

    
    df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
    df.sort_values('avg_score', inplace=True, ascending=False)
    print(df)

    log_file.write(str(df))
    log_file.write('\n')
    log_file.write(str(efs.subsets_))   
    log_file.write('\n#########################################################3\n')
    for i in efs.subsets_:
        efs.subsets_[i]['adjusted_avg_score'] = (
            adjust_r2(r2=efs.subsets_[i]['avg_score'],
                  num_examples=x.shape[0]/10,
                  num_features=len(efs.subsets_[i]['feature_idx']))
        )
 


    for i in efs.subsets_:
      score = efs.subsets_[i]['adjusted_avg_score']
      if ( efs.subsets_[i]['adjusted_avg_score'] == score and
          len(efs.subsets_[i]['feature_idx']) < len(efs.best_idx_) )\
        or efs.subsets_[i]['adjusted_avg_score'] > score:
           efs.best_idx_ = efs.subsets_[i]['feature_idx']

    score = -99e10
    print('Best adjusted R2 score: %.2f' % efs.best_score_ * (-1))
    print('Best subset:', efs.best_idx_)
    log_file.write('\n#########################################################3\n')
    log_file.write('Best adjusted R2 score: ' + str(efs.best_score_ * (-1)))   
    log_file.write('Best subset:'+ str(efs.best_idx_))  
    log_file.write('\n#########################################################3\n')
    import matplotlib.pyplot as plt

    metric_dict = efs.get_metric_dict()

    fig = plt.figure()
    k_feat = sorted(metric_dict.keys())

    avg = [metric_dict[k]['avg_score'] for k in k_feat]

    upper, lower = [], []
    for k in k_feat:
        upper.append(metric_dict[k]['avg_score'] +
                 metric_dict[k]['std_dev'])
        lower.append(metric_dict[k]['avg_score'] -
                 metric_dict[k]['std_dev'])

    plt.fill_between(k_feat,
                 upper,
                 lower,
                 alpha=0.2,
                 color='blue',
                 lw=1)

    plt.plot(k_feat, avg, color='blue', marker='o')
    plt.ylabel('R-square +/- Standard Deviation')
    plt.xlabel('Number of Features')
    feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
    feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
    plt.xticks(k_feat, 
           [str(metric_dict[k]['feature_names']) for k in k_feat], 
           rotation=90)
    plt.show()
