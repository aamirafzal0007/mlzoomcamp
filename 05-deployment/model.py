
# coding: utf-8
# mport sys
# print(sys.executable)
# print(sys.path)
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm


c = 1.0


data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.DataFrame(data)

print('Preparing Data')
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for cc in categorical_columns:
    df[cc] = df[cc].str.lower().str.replace(' ', '_')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype('int')


numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical_columns = ['gender', 'seniorcitizen', 'partner', 'dependents',
                       'phoneservice', 'multiplelines', 'internetservice',
                       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
                       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
                       'paymentmethod']
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test.churn.values


def train(df, y, c=1.0):
    dictt = df[categorical_columns+numerical].to_dict('records')
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dictt)
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x_train, y)
    return dv, model


def predict(df, model, dv):
    dictt = df[categorical_columns+numerical].to_dict('records')
    x_val = dv.fit_transform(dictt)
    y_pred = model.predict_proba(x_val)[:, 1]
    return y_pred


print('Data Modelling ')
kfold = KFold(n_splits=5, random_state=1, shuffle=True)
scores = []
fold = 1
for train_indx, val_indx in tqdm(kfold.split(df_train_full)):
    df_train = df_train_full.iloc[train_indx]
    df_val = df_train_full.iloc[val_indx]
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    dv, model = train(df_train, y_train, 1.0)
    y_pred = predict(df_val, model, dv)
    auc = roc_auc_score(y_val, y_pred)
    print(f'auc on fold{fold} is {auc}')
    fold = fold+1
    scores.append(auc)

scores = np.array(scores)
scores.mean()
print('Mean and standard Deviation of all fold auc')
print('%s   %.3f   %.3f' % (c, scores.mean(), scores.std()))


dv, model = train(df_train_full, df_train_full.churn.values, 1.0)
y_pred = predict(df_test, model, dv)
auc = roc_auc_score(y_test, y_pred)

# output_file='model_C=%s.bin'%c
output_file = f'model_C={c}.bin'


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# after out of with keyword,file is automatically closed
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print("succesfull train model")
