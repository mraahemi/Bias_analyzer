import pandas as pd
import numpy as np
import pathlib
import os
import pickle

MODEL_PATH = pathlib.Path('models')

def load_data_for_model_overview(id, engine):
    model_information=pd.read_sql_query(f'select * from dim_models where model_id = {id}', con=engine)
    target_table = model_information['data_table'].values[0]
    target_variable = model_information['target_variable'].values[0]
    df = pd.read_sql_query(f'select * from {target_table}', con=engine)
    df = df.drop(target_variable, axis=1)
#    age_hist = create_age_hist_plot(df)

    data_cols = list(set([col.split('_')[0] for col in df.columns]))

    explainer = load_shap_explainer_by_id(id)
    # todo: display shap values of the complete dataset. Due to performance reasons this should be precalculated
    #  and saved in a table, here we should only load the data.
    shap_values = list(np.ravel(explainer.shap_values(df.iloc[[20], :])))
    column_names = list(np.ravel(df.columns))
    column_names = [x for _,x in sorted(zip(shap_values,column_names))]
    shap_values = sorted(shap_values)
    return model_information, shap_values, column_names, data_cols

def load_column_names_by_id(id, engine):
    model_information=pd.read_sql_query(f'select * from dim_models where model_id = {id}', con=engine)
    target_table = model_information['data_table'].values[0]
    df = pd.read_sql_query(f'select * from {target_table} where 1 = 0', con=engine)
    return df

def explain_observation(shap_exp_id, user_features):
    explainer = load_shap_explainer_by_id(shap_exp_id)
    # todo: display shap values of the complete dataset. Due to performance reasons this should be precalculated
    #  and saved in a table, here we should only load the data.
    user_features.to_csv('testcsv.csv', index=False)

    shap_values = list(np.ravel(explainer.shap_values(user_features)))
    column_names = list(np.ravel(user_features.columns))
    column_names = [x for _,x in sorted(zip(shap_values,column_names))]
    shap_values = sorted(shap_values)
    return shap_values, column_names

def load_model_by_id(id):
    model = load_pickle_obj_by_id(id, 'model')
    return model

def calculate_percentage(model_id, user_feature):
    model = load_model_by_id(model_id)
    prediction = model.predict_proba(user_feature)[0][1]
    return prediction

def load_shap_explainer_by_id(id):
    explainer = load_pickle_obj_by_id(id, 'explainer')
    return explainer

def load_fct_table_by_id(id, engine):
    model_information=pd.read_sql_query(f'select * from dim_models where model_id = {id}', con=engine)
    target_table = model_information['data_table'].values[0]
    return pd.read_sql_query(f'select * from {target_table}', con=engine)


def load_pickle_obj_by_id(id, type):
    filelist = os.listdir(MODEL_PATH)
    if type == 'model':
        obj_list = [el for el in filelist if el.startswith(f'model{id}') and not el.endswith('explainer.pkl')]
    elif type == 'explainer':
        obj_list = [el for el in filelist if el.startswith(f'model{id}_explainer.pkl')]
    else:
        raise ValueError(f"Type {type} not specified. Choose one of 'model' or 'explainer'.")
    assert len(obj_list) == 1
    with open(MODEL_PATH / obj_list[0], 'rb') as f:
        obj = pickle.load(f)
    return obj

def column_names_to_options_dict(column_names):
    result_dct = {}
    continouos_cols = []
    for el in column_names:
        if '_' in el:
            k, v = el.split('_')
            if k not in result_dct:
                result_dct[k] = [v]
            else:
                result_dct[k].append(v)
        else:
            continouos_cols.append(el)
    return result_dct, continouos_cols