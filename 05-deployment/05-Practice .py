
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm


c=1


data=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df=pd.DataFrame(data)


df.columns=df.columns.str.lower().str.replace(' ','_')
categorical_columns=list(df.dtypes[df.dtypes=='object'].index)
for c in categorical_columns:
    df[c]=df[c].str.lower().str.replace(' ','_')
df.totalcharges=pd.to_numeric(df.totalcharges,errors='coerce')
df.totalcharges=df.totalcharges.fillna(0)
df.churn=(df.churn=='yes').astype('int')


numerical=['tenure','monthlycharges','totalcharges']
categorical_columns=['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod' ]
df_train_full,df_test=train_test_split(df,test_size=0.2,random_state=1)
y_test=df_test.churn


def train(df,y,c=1.0):
    dictt=df[categorical_columns+numerical].to_dict(orient='record')
    dv=DictVectorizer(sparse=False)
    x_train=dv.fit_transform(dictt)
    model=LogisticRegression(C=c,max_iter=1000)
    model.fit(x_train,y)
    return dv,model


def predict(df,model,dv):
    dictt=df[categorical_columns+numerical].to_dict(orient='record')
    x_val=dv.fit_transform(dictt)
    y_pred=model.predict_proba(x_val)[:,1]
    return y_pred




kfold=KFold(n_splits=5,random_state=1,shuffle=True)
scores=[]
for train_indx,val_indx in tqdm(kfold.split(df_train_full)):
    df_train=df_train_full.iloc[train_indx]
    df_val=df_train_full.iloc[val_indx]
    y_train=df_train.churn.values
    y_val=df_val.churn.values
    dv,model=train(df_train,y_train,c)
    y_pred=predict(df_val,model,dv)
    auc=roc_auc_score(y_val,y_pred)
    scores.append(auc)
scores=np.array(scores)
scores.mean()
print('%s   %.3f   %.3f' %(c,scores.mean(),scores.std()))




dv,model=train(df_train_full,df_train_full.churn.values,1.0)
y_pred=predict(df_test,model,dv)
auc=roc_auc_score(y_test,y_pred)



import pickle
#output_file='model_C=%s.bin'%c
output_file=f'model_C={c}.bin'


# In[ ]:


output_file


# In[ ]:


f_out=open(output_file,'wb')
pickle.dump((dv,model),f_out)
f_out.close()


# In[ ]:


# after out of with keyword,file is automatically closed
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)


# In[ ]:


#load the model


# In[1]:


import pickle


# In[2]:


input_file=f'model_C=1.0.bin'
with open(input_file,'rb') as f_in:
    (dv,model)=pickle.load(f_in)
     


# In[16]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[17]:


X=dv.transform(customer)


# In[18]:


X


# In[19]:


Y_pred=model.predict_proba(X)[:,1]


# In[20]:


Y_pred

