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

from werkzeug.utils import secure_filename
import os
import datetime
import time


# EDA Packages
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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


@app.route('/models/<int:id>')
def show_model(id):
    model_metadata = DimModels.query.filter_by(model_id=id).all()

    model_information=pd.read_sql_query(f'select * from dim_models where model_id = {id}', con=db.engine)
    target_table = model_information['data_table'].values[0]
    df = pd.read_sql_query(f'select * from {target_table}', con=db.engine)
    age_hist = create_age_hist_plot(df)

    return render_template('model_overview.html', model_overview=model_metadata, age_hist=age_hist)

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