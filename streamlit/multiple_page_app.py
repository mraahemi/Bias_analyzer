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

describe = pd.read_sql_table('fct_income_model_describe', con=engine)
describe = describe.set_index(describe['index'])

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

    num_feat = st.sidebar.slider('#most important features', 0, 105, 20)
    column_names = load_column_names_by_id(model_id, engine)
    # todo: make this generic. We need to drop the target_feature form the meta-information. Write
    #  an extra method to get this meta information.
    column_names = column_names.drop('over_50k', axis=1).columns
    col_options, continuous_cols = column_names_to_options_dict(column_names)


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
        randint = np.random.randint(0, 40000)
        rand_data = pd.read_sql_query(f'select * from fct_income_model where rowid = {randint}', con=engine)
        # the following two lines reverse the pd.get_dummies operation
        dummy_cols = [col for col in describe.columns if '_' in col and col != 'over_50k']
        user_features = inverse_get_dummies_single_row(rand_data.drop('over_50k', axis=1),
                                                       dummy_cols).iloc[0, :].to_dict()
        user_features.update(rand_data[continuous_cols].iloc[0, :].to_dict())
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

    for col in continuous_cols:
        user_feat_df[col] = st.sidebar.slider(col, int(describe.loc['min', col]), int(describe.loc['max', col]),
                                              int(user_features.get(col, int(describe.loc['50%', col]))))

    for col in column_names:
        if col not in user_feat_df.columns:
            user_feat_df[col] = 0
    user_feat_df = user_feat_df[column_names]
    shap_values, column_names = explain_observation(model_id, user_feat_df)

#    model_metadata, shap_values, column_names, data_cols = load_data_for_model_overview(model_mapping['model 1'],
#                                                                                        engine)
    user_feat_df_norm = 100*(user_feat_df / describe[user_feat_df.columns].loc['max', :])
    shap_values_norm = 100*(shap_values / np.max(shap_values))
    df = pd.DataFrame(np.array([shap_values_norm,
                                shap_values,
                                user_feat_df_norm[column_names].values[0],
                                user_feat_df[column_names].values[0]]).T, columns=['shap_value_norm',
                                                                                   'shap_value',
                                                                                   'feature_value_norm',
                                                                                   'feature_value'],
                      index=column_names)

    convert_cols = ['shap_value_norm', 'shap_value', 'feature_value_norm', 'feature_value']
    df[convert_cols] = df[convert_cols].astype(float)
    df['abs_value'] = np.abs(df['shap_value_norm'])
    # filter df
    df = df.sort_values('abs_value', ascending=False).iloc[:num_feat, :]
    df_stacked = df[['shap_value_norm', 'feature_value_norm']].stack().reset_index().rename(columns={'level_1': 'value_type', 0:'value',
                                                   'level_0': 'attribute'})
    df = pd.merge(df_stacked, df[['shap_value', 'feature_value']], left_on='attribute', right_on=df.index)

    bars = alt.Chart(df).mark_bar().encode(
        # y=alt.Y('attribute:O', sort='-x'),
        y=alt.Y('value_type:O', axis=alt.Axis(title=None, labels=False, ticks=False)),
        color='value_type:N',
        x='value:Q',
        row=alt.Row('attribute:N', header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=['attribute', 'shap_value', 'feature_value']
    ).interactive()


    percentage = calculate_percentage(model_id, user_feat_df)
    per = str(round(percentage, 2))
    st.write(f'The percentage to earn more than 50k with the current configuration is {per}')

    st.altair_chart(bars, use_container_width=True)
