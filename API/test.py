import os

from app import app
import unittest
from werkzeug.datastructures import FileStorage
from shutil import copyfile


unittest.TestLoader.sortTestMethodsUsing = None
PREDICT_CAT_PARAMS = {'workclass':"State-gov",
                      'education':"Bachelors",
                      'marital_status':"Never-married",
                      'occupation':"Adm-clerical",
                      'relationship':"Not-in-family",
                      'race':"White",
                      'sex':"Male",
                      'native_country':'United-States'}


def get_dataset():
    my_file_name = os.path.join("../data/us census data.csv")
    my_file = FileStorage(
        stream=open(my_file_name, "rb"),
        filename="us census data.csv"
    )
    return my_file

def get_model():
    copyfile("../models/gbc.pkl","./models/gbc.pkl")
    copyfile("../data/enc.pkl","./data/enc.pkl")


def clean_after_tests():
    for f in os.listdir("./data/"):
        os.remove(os.path.join("./data/", f))
    for f in os.listdir("./models/"):
        os.remove(os.path.join("./models/", f))


class FlaskTestCase(unittest.TestCase):

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html_test')
        self.assertEqual(response.status_code, 200)

    def test_upload(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(dataset=dataset, action="Upload")
        )
        self.assertTrue(b'Upload new data.' in response.data)
        self.assertTrue("data.csv" in os.listdir("./data/"))

    def test_change(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(dataset=dataset, action="Change")
        )
        self.assertTrue(b'Upload new data.' in response.data)
        self.assertTrue(b'Train a model before prediction.' in response.data)
        self.assertTrue("data.csv" in os.listdir("./data/"))

    def test_train_valid(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(learning_rate="0.1", depth="3", n_trees="100",model="gbc", model_name="test_name", action="Train")
        )
        self.assertTrue(b'Train accuracy' in response.data)
        self.assertTrue(b'Test accuracy' in response.data)
        self.assertTrue(b'Train a model before prediction.' not in response.data)
        self.assertTrue("test_name.pkl" in os.listdir("./models/"))

    def test_train_invalid_lr(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(learning_rate="-4", depth="3", n_trees="100",model="gbc", model_name="test_name", action="Train")
        )
        self.assertTrue(b'Train accuracy' not in response.data)
        self.assertTrue(b'Test accuracy' not in response.data)
        self.assertTrue(b"Learning rate must be" in response.data)
        self.assertTrue("test_name.pkl" not in os.listdir("./models/"))


    def test_train_invalid_depth(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(learning_rate="0.1", depth="0", n_trees="100",model="gbc", model_name="test_name", action="Train")
        )
        self.assertTrue(b'Train accuracy' not in response.data)
        self.assertTrue(b'Test accuracy' not in response.data)
        self.assertTrue(b"Max Depth must be" in response.data)
        self.assertTrue("test_name.pkl" not in os.listdir("./models/"))

    def test_train_invalid_ntrees(self):
        tester = app.test_client(self)
        dataset=get_dataset()
        response = tester.post(
            '/',
            data=dict(learning_rate="0.1", depth="3", n_trees="0",model="gbc", model_name="test_name", action="Train")
        )
        self.assertTrue(b'Train accuracy' not in response.data)
        self.assertTrue(b'Test accuracy' not in response.data)
        self.assertTrue(b"Number of Trees must be" in response.data)
        self.assertTrue("test_name.pkl" not in os.listdir("./models/"))

    def test_predict_valid(self):
        tester = app.test_client(self)
        get_model()
        response = tester.post(
            '/',
            data=dict(model_choice="gbc.pkl",
                      age="39",
                      fnlwgt="77516",
                      hours_per_week="40",
                      capital="2174",
                      workclass="State-gov",
                      education="Bachelors",
                      marital_status="Never-married",
                      occupation="Adm-clerical",
                      relationship="Not-in-family",
                      race="White",
                      sex="Male",
                      native_country="United-States",
                      action="Predict")
        )
        self.assertTrue(b'Predicted income is' in response.data)

    def test_predict_invalid_age(self):
        tester = app.test_client(self)
        get_model()
        response = tester.post(
            '/',
            data=dict(model_choice="gbc.pkl",
                      age="twelve",
                      fnlwgt="77516",
                      hours_per_week="40",
                      capital="2174",
                      workclass="State-gov",
                      education="Bachelors",
                      marital_status="Never-married",
                      occupation="Adm-clerical",
                      relationship="Not-in-family",
                      race="White",
                      sex="Male",
                      native_country="United-States",
                      action="Predict")
        )
        self.assertTrue(b'Age must be' in response.data)
        self.assertTrue(b'Predicted income is' not in response.data)


    def test_predict_invalid_fnlwgt(self):
            tester = app.test_client(self)
            get_model()
            response = tester.post(
                '/',
                data=dict(model_choice="gbc.pkl",
                          age="39",
                          fnlwgt="invalid",
                          hours_per_week="40",
                          capital="2174",
                          workclass="State-gov",
                          education="Bachelors",
                          marital_status="Never-married",
                          occupation="Adm-clerical",
                          relationship="Not-in-family",
                          race="White",
                          sex="Male",
                          native_country="United-States",
                          action="Predict")
            )
            self.assertTrue(b'Final Weight must be' in response.data)
            self.assertTrue(b'Predicted income is' not in response.data)


    def test_predict_invalid_hours(self):
            tester = app.test_client(self)
            get_model()
            response = tester.post(
                '/',
                data=dict(model_choice="gbc.pkl",
                          age="39",
                          fnlwgt="77516",
                          hours_per_week="fourty",
                          capital="2174",
                          workclass="State-gov",
                          education="Bachelors",
                          marital_status="Never-married",
                          occupation="Adm-clerical",
                          relationship="Not-in-family",
                          race="White",
                          sex="Male",
                          native_country="United-States",
                          action="Predict")
            )
            self.assertTrue(b'Hours Per Week must be' in response.data)
            self.assertTrue(b'Predicted income is' not in response.data)

    def test_predict_invalid_capital(self):
            tester = app.test_client(self)
            get_model()
            response = tester.post(
                '/',
                data=dict(model_choice="gbc.pkl",
                          age="39",
                          fnlwgt="77516",
                          hours_per_week="40",
                          capital="million",
                          workclass="State-gov",
                          education="Bachelors",
                          marital_status="Never-married",
                          occupation="Adm-clerical",
                          relationship="Not-in-family",
                          race="White",
                          sex="Male",
                          native_country="United-States",
                          action="Predict")
            )
            self.assertTrue(b'Capital must be' in response.data)
            self.assertTrue(b'Predicted income is' not in response.data)


    @classmethod
    def tearDownClass(cls):
        clean_after_tests()


if __name__ == "__main__":
    unittest.main()