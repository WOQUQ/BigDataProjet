"""
    In this algo, we choose xgboost to be the model to predict
    Because it's the best model we have trained
"""
# load packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df_train = pd.read_csv('./train.csv')
df_predict = pd.read_csv('./predict.csv')

# Pre-processing to train dataset
cols_2_delete = missing_value_info(df_train)[missing_value_info(df_train) > 50].index.to_list()
df_train = df_train.drop(cols_2_delete, axis=1)
df_train = df_train.drop("InsuredInfo_7", axis=1) # Gender column
le = LabelEncoder()
df_train['Product_Info_2_en'] = le.fit_transform(df_train['Product_Info_2'])
df_train.drop(axis=1, labels=['Product_Info_2'], inplace=True)
cols_with_null = missing_value_info(df_train)
for index in cols_with_null.index:
    median = df_train[index].median()
    df_train[index] = df_train[index].fillna(median)


# Pre-processing to predict dataset
cols_2_delete = missing_value_info(df_predict)[missing_value_info(df_predict) > 50].index.to_list()
df_predict = df_predict.drop(cols_2_delete, axis=1)
df_predict = df_predict.drop("InsuredInfo_7", axis=1) # Gender column
le = LabelEncoder()
df_predict['Product_Info_2_en'] = le.fit_transform(df_predict['Product_Info_2'])
df_predict.drop(axis=1, labels=['Product_Info_2'], inplace=True)
cols_with_null = missing_value_info(df_predict)
for index in cols_with_null.index:
    median = df_predict[index].median()
    df_predict[index] = df_predict[index].fillna(median)

xgb=XGBClassifier(n_estimators=900, learning_rate=0.1, random_state=0)
X = df_train.drop('Response', axis=1)
y = df_train['Response']
xgb.fit(X,y)
df_predict['Response'] = xgb.predict(df_predict)
df_predict.to_csv('result.csv')