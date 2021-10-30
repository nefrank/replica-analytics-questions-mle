import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import catboost as cb
import os

app = Flask(__name__)

model_list = os.listdir("./models/")

def encode_df(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    encoder = ce.OrdinalEncoder(cols=X.iloc[:, np.where(X.dtypes == object)[0]].columns, return_df=True)
    encoder = encoder.fit(X)
    X = encoder.transform(X)
    pickle.dump(encoder,open("./data/enc.dat", "wb"))

    return pd.concat((X, pd.Series(y).rename('income')), axis=1)

def load_dataset(data_file):
    df = pd.read_csv('data/us census data.csv')
    df = df.replace("?", "Other")
    df = encode_df(df)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test

def evaluate_model(model,X_train,y_train,X_test,y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return accuracy_score(y_train,y_train_pred), accuracy_score(y_test,y_test_pred)

def get_persistent_variables():
    context = {}
    context['model_list'] = os.listdir("./models/")
    context['workclass_list'] = ['State-gov','Self-emp-not-inc','Private','Federal-gov','Local-gov','Other','Self-emp-inc']
    context['education_list'] = ['Bachelors','HS-grad','11th','Masters','9th','Some-college','Assoc-acdm','Assoc-voc','7th-8th','Doctorate', 'Prof-school','5th-6th','10th','1st-4th','Preschool','12th']
    context['marital_list'] = ['Never-married','Married-civ-spouse','Divorced','Married-spouse-absent','Separated','Married-AF-spouse','Widowed']
    context['occupation_list'] = ['Adm-clerical','Exec-managerial','Handlers-cleaners','Prof-specialty','Other-service','Sales','Craft-repair','Transport-moving','Farming-fishing','Machine-op-inspct','Tech-support','Other','Protective-serv','Armed-Forces','Priv-house-serv']
    context['relationship_list'] = ['Not-in-family','Husband','Wife','Own-child','Unmarried','Other-relative']
    context['race_list'] = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    context['sex_list'] = ['Male', 'Female']
    context['country_list'] = ['United-States','Cuba','Jamaica','India','Other','Mexico','South','Puerto-Rico','Honduras','England','Canada','Germany','Iran','Philippines','Italy','Poland','Columbia','Cambodia','Thailand','Ecuador','Laos','Taiwan','Haiti','Portugal','Dominican-Republic','El-Salvador','France','Guatemala','China','Japan','Yugoslavia','Peru','Outlying-US(Guam-USVI-etc)','Scotland','Trinadad&Tobago','Greece','Nicaragua', 'Vietnam','Hong','Ireland','Hungary','Holand-Netherlands']

    return context



@app.route('/', methods=['GET'])
def hello_world():
    option_list = ["One", "Two", "Three"]
    return render_template('index.html', **get_persistent_variables())


@app.route('/', methods=['POST'])
def predict():
    if request.form['action'] == "Train":
        dataset = request.files['dataset']
        dataset_path = "./data/" + dataset.filename
        dataset.save(dataset_path)
        X_train, X_test, y_train, y_test = load_dataset(dataset_path)

        if request.form['model'] == 'gbc':
            gbc = GradientBoostingClassifier(learning_rate=0.2,max_depth=3,n_estimators=100)
            gbc.fit(X_train,y_train)
            if request.form['model-name'] == "":
                pickle.dump(gbc, open("./models/gbc.dat", "wb"))
            else:
                pickle.dump(gbc,open("./models/"+request.form['model-name']+".dat", "wb"))

            train_acc, test_acc = evaluate_model(gbc,X_train,y_train,X_test,y_test)

        if request.form['model'] == 'cbc':
            cat_features_indices = np.where(X_train.dtypes == 'object')[0]
            train_dataset = cb.Pool(X_train,y_train,cat_features=cat_features_indices)
            test_dataset = cb.Pool(X_test,y_test,cat_features=cat_features_indices)

            cbc = cb.CatBoostClassifier(cat_features=cat_features_indices,
                                        depth=2,
                                        iterations=500,
                                        eval_metric='Accuracy',
                                        allow_writing_files=False,
                                        logging_level='Silent'
                                        )
            cbc.fit(train_dataset,eval_set=test_dataset,plot=False)
            if request.form['model-name'] == "":
                pickle.dump(cbc, open("./models/cbc.dat", "wb"))
            else:
                pickle.dump(cbc,open("./models/"+request.form['model-name']+".dat", "wb"))

            train_acc, test_acc = evaluate_model(cbc,X_train,y_train,X_test,y_test)

        return render_template('index.html',tr_acc=round(train_acc*100,2),ts_acc=round(test_acc*100,2),metrics=True, **get_persistent_variables())

    if request.form['action'] == "Predict":
        model = pickle.load(open("./models/"+request.form['model-choice'], "rb"))
        encoder = pickle.load(open("./data/enc.dat", "rb"))

        input = [int(request.form['age']),
                 request.form['workclass'],
                 int(request.form['fnlwgt']),
                 request.form['education'],
                 request.form['marital-status'],
                 request.form['occupation'],
                 request.form['relationship'],
                 request.form['race'],
                 request.form['sex'],
                 int(request.form['hours-per-week']),
                 request.form['native-country'],
                 int(request.form['capital'])
                 ]

        new = pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education', 'marital-status',
                                    'occupation', 'relationship', 'race', 'sex', 'hours-per-week',
                                    'native-country', 'capital'])
        new.loc[0,:] = input
        input = encoder.transform(new)

        pred = model.predict(input)[0]
        if pred == 0:
            prediction = "<=50k"
        else:
            prediction = ">50k"
        proba = model.predict_proba(input)[0][0]

        return render_template('index.html',prediction=prediction,proba=np.round(proba*100,2), **get_persistent_variables())


if __name__ == '__main__':
    app.run(port=3000, debug=True)
