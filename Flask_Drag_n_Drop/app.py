# Flask Packages
from flask import Flask,render_template,request,url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet, configure_uploads, IMAGES, DATA, ALL
from flask_sqlalchemy import SQLAlchemy
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from werkzeug.utils import secure_filename
import os
import datetime
import time
import pickle


# EDA Packages
import pandas as pd 
import numpy as np 

# ML Packages
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import own code
from project_utils.plot_utils import create_histogram

app = Flask(__name__, static_url_path='/static')
Bootstrap(app)
db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'database'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/model_information.db'

MODEL_PATH = pathlib.Path('models')

# Saving Data To Database Storage
class FileContents(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(300))
    modeldata = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)

# class FctModelData(db.Model):
#     model_id = db.Column(db.Integer)
#     column_name = db.Column(db.String(300))
#     column_value = db.Column(db.Float)
#     is_target_variable = db.Column(db.Integer)
#     is_donated = db.Column(db.Integer)
#     index_column = db.Column(db.Integer)

class DimModels(db.Model):
    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50))
    create_time = db.Column(db.DateTime)
    is_uploaded_by_user = db.Column(db.Integer)
    model_description = db.Column(db.String(2000))

@app.route('/')
def index():
    models = DimModels.query.all()
    return render_template('index.html', models=models)


@app.route('/file-upload')
def fileupload():
    # models = DimModels.query.all()
    return render_template('file_upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def create_age_hist_plot(df):
    hist = np.histogram(df['age'], 100)

    df_hist = pd.DataFrame(hist, index=['counts', 'age']).T
    df_hist = df_hist[df_hist['counts'] != 0]

    ax = sns.relplot(
        data=df_hist, kind="line",
        x="age", y="counts"
    )
    max_val = 1.03 * df_hist['counts'].max()
    # todo: this must be inserted data, don't use "25" here
    ax = plt.vlines(25, df_hist['counts'].min(), max_val, colors='red')
    url = "/static/new_plot.png"
    plt.savefig(f'templates/{url}')
    return url

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

def load_model_by_id(id):
    model = load_pickle_obj_by_id(id, 'model')
    return model

def load_shap_explainer_by_id(id):
    explainer = load_pickle_obj_by_id(id, 'explainer')
    return explainer

def load_data_for_model_overview(id):
    model_metadata = DimModels.query.filter_by(model_id=id).all()

    model_information=pd.read_sql_query(f'select * from dim_models where model_id = {id}', con=db.engine)
    target_table = model_information['data_table'].values[0]
    target_variable = model_information['target_variable'].values[0]
    df = pd.read_sql_query(f'select * from {target_table}', con=db.engine)
    df = df.drop(target_variable, axis=1)
    age_hist = create_age_hist_plot(df)

    data_cols = list(set([col.split('_')[0] for col in df.columns]))

    explainer = load_shap_explainer_by_id(id)
    # todo: display shap values of the complete dataset. Due to performance reasons this should be precalculated
    #  and saved in a table, here we should only load the data.
    shap_values = list(np.ravel(explainer.shap_values(df.iloc[[20], :])))
    column_names = list(np.ravel(df.columns))
    column_names = [x for _,x in sorted(zip(shap_values,column_names))]
    shap_values = sorted(shap_values)
    return model_metadata, age_hist, shap_values, column_names, data_cols

@app.route('/models/<int:id>', methods=['GET', 'POST'])
def show_model(id):
    model_metadata, age_hist, shap_values, feature_cols, data_cols = load_data_for_model_overview(id)
    threshold = 0.01
    json_cols_data = {col_name: shap_value for col_name, shap_value in zip(feature_cols, shap_values) if
                      abs(shap_value) > threshold}
    return render_template('model_overview.html', model_overview=model_metadata, age_hist=age_hist,
                           shap_values=shap_values, feature_cols=feature_cols, json_cols_data=json_cols_data,
                           chartID='test_id', data_cols=data_cols)

@app.route('/models_adjusted/<int:id>', methods=['GET', 'POST'])
def show_model_own_data(id):
    model_metadata, age_hist, shap_values, feature_cols, data_cols = load_data_for_model_overview(id)
    for col in data_cols:
        first_name = request.form['age']
    return render_template('model_overview_adjusted.html', model_overview=model_metadata, age_hist=age_hist,
                           shap_values=shap_values, feature_cols=feature_cols,
                           chartID='test_id', first_name=first_name, data_cols=data_cols)

# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        filename = secure_filename(file.filename)
        # os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
        file.save(os.path.join('database',filename))
        fullfile = os.path.join('database',filename)

        # For Time
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA function
        df = pd.read_csv(os.path.join('database',filename))
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_targetname = df[df.columns[-1]].name
        df_featurenames = df_columns[0:-1] # select all columns till last column
        df_Xfeatures = df.iloc[:,0:-1]
        df_Ylabels = df[df.columns[-1]] # Select the last column as target
        # same as above df_Ylabels = df.iloc[:,-1]


        # Model Building
        X = df_Xfeatures
        Y = df_Ylabels
        seed = 7
        # prepare models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn


        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names

        # Saving Results of Uploaded Files to Sqlite DB
        newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
        db.session.add(newfile)
        db.session.commit()

    return render_template('details.html',filename=filename,date=date,
        df_size=df_size,
        df_shape=df_shape,
        df_columns =df_columns,
        df_targetname =df_targetname,
        model_results = allmodels,
        model_names = names,
        fullfile = fullfile,
        dfplot = df
        )

if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=True)





# Coded by mraahemi 2020