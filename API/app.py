import pickle
import string
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import catboost as cb
import os
from io import StringIO
import io

app = Flask(__name__)

US_CENSUS_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex', 'hours-per-week',
                  'native-country', 'capital', 'income']
DATASET_PATH = "./data/data.csv"


def encode_df(df):
    # Returns DataFrame with 'object' type columns encoded using Ordinal Encoding

    X = df.iloc[:, :-1]  # Construct X from features
    y = df.iloc[:, -1]  # Construct y from label
    y = LabelEncoder().fit_transform(y)  # Encode label
    encoder = ce.OrdinalEncoder(cols=X.iloc[:, np.where(X.dtypes == object)[0]].columns, return_df=True)
    encoder = encoder.fit(X)
    X = encoder.transform(X)  # Encode categorical features
    pickle.dump(encoder, open("../data/enc.pkl", "wb"))  # Save feature encoder for encoding new inputs

    return pd.concat((X, pd.Series(y).rename('income')), axis=1)


def load_dataset():
    # Reads in uploaded data and returns train and test sets for X and y

    df = pd.read_csv(DATASET_PATH)
    df = df.replace("?", "Other")
    df = encode_df(df)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Returns train and test accuracy of model

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)


def get_persistent_variables():
    # Retrieves variables used in templates to populate drop down lists and views

    context = {'model_list': os.listdir("./models/"),
               'trained': "data.csv" in os.listdir("./data/") and len(os.listdir("./models/")) != 0,
               'uploaded': "data.csv" in os.listdir("./data/")}

    if context['trained']:
        df = pd.read_csv(DATASET_PATH)
        df = df.replace("?", "Other")

        context['workclass_list'] = list(df['workclass'].unique())
        context['education_list'] = list(df['education'].unique())
        context['marital_list'] = list(df['marital-status'].unique())
        context['occupation_list'] = list(df['occupation'].unique())
        context['relationship_list'] = list(df['relationship'].unique())
        context['race_list'] = list(df['race'].unique())
        context['sex_list'] = list(df['sex'].unique())
        context['country_list'] = list(df['native-country'].unique())

    return context


def validate_inputs(request):
    # Check data in request form, if invalid raises Exception to display in view

    # Validation when uploading new data
    if request.form['action'] == "Upload" or request.form['action'] == "Change":
        if request.files['dataset'].mimetype != "application/vnd.ms-excel":
            raise Exception("Data empty or not in .csv or .xls format")

        # TODO: Validating if data is same format as US Census is causing errors
        # dataset = request.files['dataset']
        # df = pd.read_csv(io.BytesIO(dataset.read()),sep=",")
        # try:
        #     if any(US_CENSUS_COLS != df.columns):
        #         raise Exception()
        # except:
        #     raise Exception("Please upload valid US Census data with the following features:", US_CENSUS_COLS)

    # Validation on model parameters when training models
    elif request.form['action'] == "Train":
        safechars = string.ascii_letters + string.digits + "~ -_."
        try:
            if float(request.form['learning_rate']) <= 0:
                raise Exception()
        except:
            raise Exception("Learning rate must be of type 'float' and > 0")
        try:
            if int(request.form['depth']) <= 0:
                raise Exception()
        except:
            raise Exception("Max Depth must be of type 'int' and > 0")
        try:
            if int(request.form['n_trees']) <= 0:
                raise Exception()
        except:
            raise Exception("Number of Trees must be of type 'int' and > 0")
        if type(request.form['model_name']) != str:
            raise Exception()
        elif any(c not in safechars for c in request.form['model_name']):
            raise Exception("Model name contains illegal characters")

    # Validation on prediction parameters when predicting
    elif request.form['action'] == "Predict":
        try:
            int(request.form['age'])
        except:
            raise Exception("Age must be of type 'int'")
        try:
            int(request.form['fnlwgt'])
        except:
            raise Exception("Final Weight must be of type 'int'")
        try:
            int(request.form['hours_per_week'])
        except:
            raise Exception("Hours Per Week must be of type 'int'")
        try:
            int(request.form['capital'])
        except:
            raise Exception("Capital must be of type 'int'")


@app.route('/', methods=['GET'])
def homepage():
    # GET method renders template with persistent variables
    return render_template('index.html', **get_persistent_variables())


@app.route('/', methods=['POST'])
def predict():
    # POST method renders template depending on action performed
    try:
        validate_inputs(request)
    except Exception as e:
        return render_template('index.html', error=e, **get_persistent_variables())

    if request.form['action'] == "Upload":
        dataset = request.files['dataset']
        dataset.save(DATASET_PATH)
        return render_template('index.html', **get_persistent_variables())

    if request.form['action'] == "Change":
        for f in os.listdir("./data/"):
            os.remove(os.path.join("./data/", f))
        for f in os.listdir("./models/"):
            os.remove(os.path.join("./models/", f))
        dataset = request.files['dataset']
        dataset.save(DATASET_PATH)
        return render_template('index.html', **get_persistent_variables())

    if request.form['action'] == "Train":

        lr = float(request.form['learning_rate'])
        dep = int(request.form['depth'])
        n_trees = int(request.form['n_trees'])

        X_train, X_test, y_train, y_test = load_dataset()

        if request.form['model'] == 'gbc':
            gbc = GradientBoostingClassifier(learning_rate=lr,
                                             max_depth=dep,
                                             n_estimators=n_trees)
            gbc.fit(X_train, y_train)
            if request.form['model_name'] == "":
                pickle.dump(gbc, open("./models/gbc.pkl", "wb"))
            else:
                pickle.dump(gbc, open("./models/" + request.form['model_name'] + ".pkl", "wb"))

            train_acc, test_acc = evaluate_model(gbc, X_train, y_train, X_test, y_test)

        if request.form['model'] == 'cbc':
            cat_features_indices = np.where(X_train.dtypes == 'object')[0]
            train_dataset = cb.Pool(X_train, y_train, cat_features=cat_features_indices)
            test_dataset = cb.Pool(X_test, y_test, cat_features=cat_features_indices)

            cbc = cb.CatBoostClassifier(cat_features=cat_features_indices,
                                        learning_rate=lr,
                                        depth=dep,
                                        iterations=n_trees,
                                        eval_metric='Accuracy',
                                        allow_writing_files=False,
                                        logging_level='Silent'
                                        )
            cbc.fit(train_dataset, eval_set=test_dataset, plot=False)
            if request.form['model_name'] == "":
                pickle.dump(cbc, open("./models/cbc.pkl", "wb"))
            else:
                pickle.dump(cbc, open("./models/" + request.form['model_name'] + ".pkl", "wb"))

            train_acc, test_acc = evaluate_model(cbc, X_train, y_train, X_test, y_test)

        return render_template('index.html', tr_acc=round(train_acc * 100, 2), ts_acc=round(test_acc * 100, 2),
                               metrics=True,
                               **get_persistent_variables())

    if request.form['action'] == "Predict":
        model = pickle.load(open("./models/" + request.form['model_choice'], "rb"))
        encoder = pickle.load(open("../data/enc.pkl", "rb"))

        input_X = [int(request.form['age']),
                   request.form['workclass'],
                   int(request.form['fnlwgt']),
                   request.form['education'],
                   request.form['marital_status'],
                   request.form['occupation'],
                   request.form['relationship'],
                   request.form['race'],
                   request.form['sex'],
                   int(request.form['hours_per_week']),
                   request.form['native_country'],
                   int(request.form['capital'])]

        df_X = pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'hours-per-week',
                                     'native-country', 'capital'])
        df_X.loc[0, :] = input_X
        input_X = encoder.transform(df_X)


        pred = model.predict(input_X)[0]
        if pred == 0:
            prediction = "<=50k"
        else:
            prediction = ">50k"
        proba = model.predict_proba(input_X)[0][0]


        return render_template('index.html', prediction=prediction, proba=np.round(proba * 100, 2),
                               **get_persistent_variables())


if __name__ == '__main__':
    app.run(port=3000, debug=True)
