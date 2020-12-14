import sqlite3
import os
from sklearn.datasets import fetch_openml
import datetime
import numpy as np
import pandas as pd

connection = sqlite3.connect('database/model_information.db')

FOLDER = 'database/ddl'

for file in os.listdir(FOLDER):
    if file.endswith('sql'):
        with open(f'{FOLDER}/{file}') as f:
            connection.executescript(f.read())
# load adult dataset
adult = fetch_openml(data_id=1590, as_frame=True)
# insert data to dim_models

dim_models = pd.DataFrame([[1,
               'predict_income_model',
               datetime.datetime.now(),
               0,
               'over_50k',
               'fct_income_model',
               f"Model to predict whether one makes more than 50k a year based on the features: {', '.join(list(adult.data.columns))}"
                            ]],
               columns=['model_id', 'model_name', 'create_time', 'is_uploaded_by_user', 'target_variable',
                        'data_table', 'model_description'])

dim_models.to_sql('dim_models', con=connection, if_exists='append', index=False)

# insert data to fct_model_data
df = pd.concat([pd.get_dummies(adult.data), pd.get_dummies(adult.target)], axis=1).rename(columns={'>50K': 'over_50k'})
df = df.drop('<=50K', axis=1)
df.to_sql('fct_income_model', con=connection, if_exists='append', index=False)

df.describe().to_sql('fct_income_model_describe', con=connection, if_exists='append')

connection.close()
