import os
from shutil import copyfile

from locust import HttpUser, task, between
from werkzeug.datastructures import FileStorage


def get_dataset():
    my_file_name = os.path.join("../data/us census data.csv")
    my_file = FileStorage(
        stream=open(my_file_name, "rb"),
        filename="us census data.csv"
    )
    return my_file


class WebsiteTestUser(HttpUser):
    wait_time = between(1, 1)

    def on_start(self):
        copyfile("../models/gbc.pkl", "./models/gbc.pkl")
        copyfile("../data/us census data.csv", "./data/data.csv")
        pass

    def on_stop(self):
        for f in os.listdir("./data/"):
            os.remove(os.path.join("./data/", f))
        for f in os.listdir("./models/"):
            os.remove(os.path.join("./models/", f))
        pass

    @task(1)
    def trainer(self):
        get_dataset()
        self.client.post("http://127.0.0.1:8080/",
                         data=dict(learning_rate="0.1", depth="3", n_trees="100", model="gbc", model_name="gbc",
                                   action="Train"))
