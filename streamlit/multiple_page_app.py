import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sqlalchemy import create_engine

# TODO: implement the option that we can select a random entry form the database as a gamification feature:
#  This should be easy. First add the length of the table to the metadata info
#  add a button which basically takes a random number between 0 and length_of_table -1 and select * from {target_table} where rownum = random_number
#  then display this data

from helper import load_data_for_model_overview, column_names_to_options_dict, explain_observation, \
    load_column_names_by_id, calculate_percentage, load_fct_table_by_id
engine = create_engine('sqlite:///database/model_information.db')

model_mapping = {'model 1': 1, 'model 2': 2}

@st.cache
def load_data_table(model_id, engine):
    return load_fct_table_by_id(model_id, engine)

option = st.sidebar.selectbox(
    'What do you want to check out?',
    ['Overview', 'model 1'])
if option == 'Overview':
    st.title('Multiple page app')

    st.write('This page wants to give an overview over different finance scoring models. Check them out and '
             'see how these models score you. If your interest it would be great if you could donate your personal '
             'data to us.')
else:
    st.title('Income Model')
    st.write('This is a model predicting the income based on features.')
    model_id = model_mapping[option]
    # todo: cache this
    df_data = load_fct_table_by_id(model_id, engine)

    num_feat = st.sidebar.slider('#most important features', 0, 105, 20)
    column_names = load_column_names_by_id(model_id, engine)
    # todo: make this generic. We need to drop the target_feature form the meta-information. Write
    #  an extra method to get this meta information.
    column_names = column_names.drop('over_50k', axis=1).columns
    col_options, continouos_cols = column_names_to_options_dict(column_names)


    def inverse_get_dummies_single_row(df, dummy_cols):
        s = df[dummy_cols]
        s = s[s != 0].dropna(axis=1)
        l = s.stack().to_frame().reset_index()
        t = l.assign(value=[col.split('_')[1] for col in l['level_1'].values],
                     col_name=[col.split('_')[0] for col in l['level_1'].values],
                     row_number=l['level_0'].values).drop(0, axis=1)
        return pd.pivot_table(t, columns='col_name', values='value', index='row_number', aggfunc='first')

    user_features = {}
    if st.sidebar.button("Show random Person from Dataset"):
        user_features = {}
        randint = np.random.randint(0, len(df_data))
        rand_data = df_data.iloc[[randint], :]
        rand_data.to_csv('rand_data.csv', index=False)
        # the following two lines reverse the pd.get_dummies operation
        dummy_cols = [col for col in df_data.columns if '_' in col and col != 'over_50k']
        user_features = inverse_get_dummies_single_row(rand_data.drop('over_50k', axis=1),
                                                       dummy_cols).iloc[0, :].to_dict()
        user_features.update(rand_data[continouos_cols].iloc[0, :].to_dict())
    for feature, option_list in col_options.items():
        feat_from_dict = user_features.get(feature)
        #if feature in user_features and feature in :
        if feature in user_features:
             default_ind = option_list.index(feat_from_dict)
        else:
            default_ind = 0
        user_features[feature] = st.sidebar.selectbox(
            f'What is your {feature}?', option_list, index=default_ind)
    # todo: would be better if this is read dynamically. So we need another metadata table for each model which
    #  includes df.describe() like data and read like the mean or median as default value
    user_feat_df = pd.get_dummies(pd.DataFrame([user_features]))

    user_feat_df['age'] = st.sidebar.slider('age', 10, 99, int(user_features.get('age', 25)))
    user_feat_df['fnlwgt'] = st.sidebar.slider('fnlwgt', 10000, 1490400, int(user_features.get('fnlwgt', 50000)))
    user_feat_df['education-num'] = st.sidebar.slider('education-num', 1, 16, int(user_features.get('education-num', 13)))
    user_feat_df['capital-gain'] = st.sidebar.slider('capital-gain', 0, 99999, int(user_features.get('capital-gain', 50000)))
    user_feat_df['capital-loss'] = st.sidebar.slider('capital-loss', 0, 4356, int(user_features.get('capital-loss', 2000)))
    user_feat_df['hours-per-week'] = st.sidebar.slider('hours-per-week', 1, 99, int(user_features.get('hours-per-week', 38)))

    for col in column_names:
        if col not in user_feat_df.columns:
            user_feat_df[col] = 0
    user_feat_df = user_feat_df[column_names]
    shap_values, column_names = explain_observation(model_id, user_feat_df)

#    model_metadata, shap_values, column_names, data_cols = load_data_for_model_overview(model_mapping['model 1'],
#                                                                                        engine)
    df = pd.DataFrame(np.array([shap_values, column_names]).T, columns=['shap_value', 'attribute'])

    df['shap_value'] = df['shap_value'].astype(float)
    df['abs_value'] = np.abs(df['shap_value'])
    # filter df
    df = df.sort_values('abs_value', ascending=False).iloc[:num_feat, :]
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y('attribute', sort='-x'),
        color='attribute',
        x='shap_value',
        tooltip=['attribute', 'shap_value']
    ).interactive()

    percentage = calculate_percentage(model_id, user_feat_df)
    per = str(round(percentage, 2))
    st.write(f'The percentage to earn more than 50k with the current configuration is {per}')

    st.altair_chart(bars, use_container_width=True)
