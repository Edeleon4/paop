print('abs model loaded')
from paop.models.data import Data
"""
    Model interface base
        Exposed API:
            train: Trains the model on inputted Training set
            evaluate: Evaluates the model on the Dev set
            summary: Returns summary data about the model
            save(path): Store the model at the given path

            stati:
                load
"""
class AbstractModel():
    name = "AbstractModel"
    def load():
        raise NotImplementedError()

    def __init__(self, train_path, dev_path, test_path,
                 debug=False):
        self.debug = debug
        self.data, self.props = self._get_data(train_path, dev_path, test_path)
        self.model = self._get_model()

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def _get_data(self, train_path, dev_path, test_path):
        raise NotImplementedError()

    def _get_model(self):
        raise NotImplementedError()

