
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
data_train= arff.loadarff('sheepcustomizado.arff')
train = pd.DataFrame(data_train[0])




sns.distplot(train['Weight'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Weight'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Weight distribution')
plt.show()



plt.scatter(x= 'Area', y='Weight', data = train)

##matrix confusion correlation
plt.figure(figsize=(30,8))
sns.set(font_scale=0.8)
sns.heatmap(train.corr(), cbar=True, annot=True, square=True, fmt='.2f')
plt.show()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

train.Weight.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Weight', size=24)
plt.xlabel('Weight (Kilograms)', size=18)
plt.ylabel('Quantity', size=18);
plt.show()
