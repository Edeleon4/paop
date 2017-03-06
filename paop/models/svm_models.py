import numpy as np
from random import shuffle
import json
from sklearn import svm
from paop.models.data import Data
from paop.models.abstract_model import AbstractModel
#TODO move from json to pickle
class AbstractSVM(AbstractModel):
    name = "SVM"
    #__init__ properties data
    def setup(self):
        self.model = svm.SVC()

    def train(self):
        print(len(self.data.train_x))
        self.model.fit(
            self.data.train_x,
            self.data.train_y
        )

    def evaluate(self):
        score = self.model.score(self.data.dev_x, self.data.dev_y)
        return score

class SVM(AbstractSVM):
    def __init__(self, train_path, dev_path, test_path, debug=False) :
        super().__init__(train_path, dev_path, test_path, debug)

    def get_data(self, train_path, dev_path, test_path):
        train_x = None
        train_y = None
        dev_x = None
        dev_y = None
        test_x = None
        test_y = None
        with open(train_path, 'r') as file:
            data = json.loads(file.read())
            print(len(data['x']))
            train_x = np.array(data['x'])
            train_y = np.array(data['y'])

        with open(dev_path, 'r') as file:
            data = json.loads(file.read())
            dev_x = np.array(data['x'])
            dev_y = np.array(data['y'])

        with open(test_path, 'r') as file:
            data = json.loads(file.read())
            test_x = np.array(data['x'])
            test_y = np.array(data['y'])
        return Data(train_x, train_y, dev_x, dev_y, test_x, test_y), {}

