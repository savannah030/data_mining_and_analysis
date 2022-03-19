#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
#import seaborn as sns; sns.set()
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer
import os
import re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings

def titanic_survivor_predictor():
    data_train = pd.read_csv("./train.csv")
    data_test = pd.read_csv("./test.csv")
    data_train
    
    vars_ = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X_train = pd.get_dummies(data_train[vars_], columns = ['Sex', 'Embarked'])
    y_train = data_train['Survived']
    X_test = pd.get_dummies(data_test[vars_], columns = ['Sex', 'Embarked'])
    X_train

    vectorizer = CountVectorizer(binary = True, max_features = 3)
    text_train = vectorizer.fit_transform(data_train['Name'])
    text_test = vectorizer.transform(data_test['Name'])
    #print(vectorizer.get_feature_names())
    text_train.toarray()

    X_train = pd.concat([X_train, pd.DataFrame(text_train.toarray())]
                    , axis = 1, ignore_index = True)
    X_test = pd.concat([X_test, pd.DataFrame(text_test.toarray())],
                   axis = 1, ignore_index = True)
    #filling in the nans with the nearest neighbor
    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    #run model
    model = RandomForestClassifier(n_estimators = 10, random_state = 0)
    grid = GridSearchCV(estimator = model, param_grid = {'max_depth': range(2, 6)}, cv = 10)
    grid.fit(X_train, y_train)

    predict = grid.predict(X_test)
    output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predict})
    output = output.astype({'Survived': 'int32'})
    output.to_csv('submission.csv', index=False)
    print("\n<<<<<<<<<<submission.csv has been made>>>>>>>>>>\n")
    return grid.best_score_

def market_basket_analyzer(min):
    dataset = pd.read_csv("./Market_Basket_Optimisation.csv", header=None)
    trans = []
    for i in range(0, 7501):
        trans.append([str(dataset.values[i,j]) for j in range(0, 20)])
    trans = np.array(trans)
    
    te = TransactionEncoder()
    dataset = te.fit_transform(trans)
    dataset = pd.DataFrame(dataset, columns = te.columns_,dtype=int)
    dataset.drop('nan',axis=1,inplace=True)
    warnings.filterwarnings('ignore')
    #dataset = dataset.loc[:, list(y.index)]
    
    frequent_itemsets = fpgrowth(dataset, min_support = min, use_colnames=True)
    #frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print(frequent_itemsets)
     
menu = 0

while True:
    print("[ Student ID: 1713661 ]")
    print("[ Name: 한승윤 ]\n")

    print("1. Titanic Survivor Predictor")
    print("2. Market Basket Analyzer")
    print("3. Quit")

    menu = int(input("Select Menu: "))

    if menu == 1:
        score = titanic_survivor_predictor()*100
        #print("Accuracy: {0:0.2f} %".format(score))
    elif menu == 2:
        print("2")
        min = float(input("Enter the minimum support: "))
        market_basket_analyzer(min)
    elif menu == 3:
        break
    else:
        print("\n<<<<<<<<<< wrong number >>>>>>>>>>\n")













# In[1]:




