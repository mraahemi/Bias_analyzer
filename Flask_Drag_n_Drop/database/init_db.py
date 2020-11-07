import sqlite3
import os
from sklearn.datasets import fetch_openml
import datetime
import pandas as pd

connection = sqlite3.connect('model_information.db')

FOLDER = 'ddl'

for file in os.listdir(FOLDER):
    if file.endswith('sql'):
        with open(f'{FOLDER}/{file}') as f:
            connection.executescript(f.read())
# load adult dataset
adult = fetch_openml(data_id=1590, as_frame=True)
df = pd.melt(adult.data, value_vars=adult.data.columns).rename(columns={'variable': 'column_name',
                                                                        'value': 'column_value'})
# insert data to dim_models
dim_models = pd.DataFrame([[1,
               'predict_income_model',
               datetime.datetime.now(),
               0,
               f"Model to predict whether one makes more than 50k a year based on the features: {', '.join(list(adult.data.columns))}"]],
               columns=['model_id', 'model_name', 'create_time', 'is_uploaded_by_user', 'model_description'])

dim_models.to_sql('dim_models', con=connection, if_exists='append', index=False)

# insert data to fct_model_data
df.loc[:, 'model_id'] = 1
df.loc[:, 'is_target_variable'] = 0
df.loc[:, 'is_donated'] = 0

target = pd.DataFrame(adult.target).rename(columns={'class': 'column_value'})
target.loc[:, 'model_id'] = 1
target.loc[:, 'column_name'] = 'income'
target.loc[:, 'is_target_variable'] = 1
target.loc[:, 'is_donated'] = 0

insert_df = df.append(target)
insert_df.to_sql('fct_model_data', con=connection, if_exists='append', index=False)

connection.close()
