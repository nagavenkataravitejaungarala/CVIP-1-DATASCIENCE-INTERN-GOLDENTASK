#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('diabetes.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.describe


# In[7]:


data.shape


# In[8]:


data.index


# In[10]:


data.columns


# In[11]:


data.dtypes


# In[12]:


data.info


# In[13]:


#data cleaning


# In[14]:


data=data.drop_duplicates()


# In[15]:


data


# In[16]:


data.isnull().sum()


# In[17]:


print(data[data['BloodPressure']==0].shape[0])
print(data[data['Glucose']==0].shape[0])
print(data[data['SkinThickness']==0].shape[0])
print(data[data['Insulin']==0].shape[0])
print(data[data['BMI']==0].shape[0])


# In[18]:



data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].median())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].median())
data['BMI']=data['BMI'].replace(0,data['BMI'].median())


# In[19]:


data['Glucose']


# In[20]:


data['BloodPressure']


# In[21]:


data['SkinThickness']


# In[22]:


data['Insulin']


# In[23]:


data['BMI']


# In[24]:


#data visualization


# In[25]:


sns.countplot('Outcome',data=data)


# In[26]:


#histogram for each  feature
data.hist(bins=10,figsize=(10,10))
plt.show()


# In[27]:


plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x='Glucose',data=data)
plt.subplot(3,3,2)
sns.boxplot(x='BloodPressure',data=data)
plt.subplot(3,3,3)
sns.boxplot(x='Insulin',data=data)
plt.subplot(3,3,4)
sns.boxplot(x='BMI',data=data)
plt.subplot(3,3,5)
sns.boxplot(x='Age',data=data)
plt.subplot(3,3,6)
sns.boxplot(x='SkinThickness',data=data)
plt.subplot(3,3,7)
sns.boxplot(x='Pregnancies',data=data)
plt.subplot(3,3,8)
sns.boxplot(x='DiabetesPedigreeFunction',data=data)


# In[29]:


from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(20,20));


# In[30]:


#feature selection


# In[31]:


corrmat=data.corr()
sns.heatmap(corrmat, annot=True)


# In[32]:


dat_selected=data.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction'],axis='columns')


# In[33]:


dat_selected


# In[34]:


#handling outliers


# In[35]:


from sklearn.preprocessing import QuantileTransformer
x=dat_selected
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
dat_new=quantile.transform(X)
dat_new=pd.DataFrame(X)
dat_new.columns =['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']
dat_new.head()


# In[36]:


plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x=dat_new['Glucose'],data=dat_new)
plt.subplot(3,3,2)
sns.boxplot(x=dat_new['BMI'],data=dat_new)
plt.subplot(3,3,3)
sns.boxplot(x=dat_new['Pregnancies'],data=dat_new)
plt.subplot(3,3,4)
sns.boxplot(x=dat_new['Age'],data=dat_new)
plt.subplot(3,3,5)
sns.boxplot(x=dat_new['SkinThickness'],data=dat_new)


# In[37]:


#Split the Data Frame into X and y


# In[38]:


target_name='Outcome'
y= dat_new[target_name]
X=dat_new.drop(target_name,axis=1)


# In[39]:


X.head()


# In[40]:


y.head() 


# In[41]:


#train test split


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)


# In[43]:


X_train.shape,y_train.shape


# In[44]:


X_test.shape,y_test.shape


# In[45]:


#machine learning algorithm


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


# In[47]:


#List Hyperparameters to tune
knn= KNeighborsClassifier()
n_neighbors = list(range(15,25))
p=[1,2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
#convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p,weights=weights,metric=metric)
#Making model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1',error_score=0)


# In[48]:


best_model = grid_search.fit(X_train,y_train)


# In[49]:


print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[50]:


knn_pred = best_model.predict(X_test)


# In[51]:


print("Classification Report is:\n",classification_report(y_test,knn_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,knn_pred))


# In[52]:


#navie bayes


# In[53]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

param_grid_nb = {
    'var_smoothing': np.logspace(0,-2, num=100)
}
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)


# In[54]:


best_model= nbModel_grid.fit(X_train, y_train)


# In[55]:


nb_pred=best_model.predict(X_test)


# In[56]:


print("Classification Report is:\n",classification_report(y_test,nb_pred))
print("\n F1:\n",f1_score(y_test,nb_pred))
print("\n Precision score is:\n",precision_score(y_test,nb_pred))
print("\n Recall score is:\n",recall_score(y_test,nb_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,nb_pred))


# In[57]:


#support vector machine 


# In[58]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


# In[59]:


model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']


# In[60]:


grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0)


# In[61]:


grid_result = grid_search.fit(X, y)
svm_pred=grid_result.predict(X_test)


# In[62]:


grid_result


# In[63]:


svm_pred


# In[64]:


print("Classification Report is:\n",classification_report(y_test,svm_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,svm_pred))


# In[65]:


#decision tree


# In[68]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
data = DecisionTreeClassifier(random_state=42)


# In[70]:


params = {
    'max_depth': [5, 10, 20,25],
    'min_samples_leaf': [10, 20, 50, 100,120],
    'criterion': ["gini", "entropy"]
}


# In[71]:


grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[72]:


best_model=grid_search.fit(X_train, y_train)


# In[73]:


dt_pred=best_model.predict(X_test)


# In[74]:


print("Classification Report is:\n",classification_report(y_test,dt_pred))
print("\n F1:\n",f1_score(y_test,dt_pred))
print("\n Precision score is:\n",precision_score(y_test,dt_pred))
print("\n Recall score is:\n",recall_score(y_test,dt_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,dt_pred))


# In[75]:


#random forest


# In[76]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[77]:


model = RandomForestClassifier()
n_estimators = [1800]
max_features = ['sqrt', 'log2']


# In[78]:


grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)


# In[79]:


best_model = grid_search.fit(X_train, y_train)


# In[80]:


rf_pred=best_model.predict(X_test)


# In[81]:


print("Classification Report is:\n",classification_report(y_test,rf_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,rf_pred))


# In[82]:


#logistic regression


# In[83]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score


# In[84]:


reg = LogisticRegression()
reg.fit(X_train,y_train)  


# In[85]:


lr_pred=reg.predict(X_test)


# In[86]:


print("Classification Report is:\n",classification_report(y_test,lr_pred))
print("\n F1:\n",f1_score(y_test,lr_pred))
print("\n Precision score is:\n",precision_score(y_test,lr_pred))
print("\n Recall score is:\n",recall_score(y_test,lr_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,lr_pred))


# In[ ]:




