import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import xgboost
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle


if __name__ == '__main__':
    adult = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(adult.data)
    y_true = (adult.target == '>50K') * 1

    scaler = sklearn.preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y_true)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('classifier', RandomForestClassifier())])

    # Create space of candidate learning algorithms and their hyperparameters
    search_space = [{'classifier': [LogisticRegression()],
                     'classifier__C': np.logspace(0, 4, 10)},
                    {'classifier': [RandomForestClassifier()],
                     'classifier__max_features': [1, 2, 3]},
                    {'classifier': [xgboost.XGBClassifier()],
                     'classifier__n_estimators': [10, 20, 50]}]

    # Create grid search
    gs = GridSearchCV(pipe, search_space)
    gs.fit(X_train, y_train)
    pickle.dump(gs.best_estimator_, open('models/model1_income_model.pkl', 'wb'))

    print(gs.best_estimator_.score(X_test, y_test))
    print(gs.best_estimator_)

    print('success')
