import pickle as p
from sklearn import svm
from paop.models.data import Data
from paop.models.abstract_model import AbstractModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Merge, LSTM
from keras.optimizers import Adam
class LSTMMergeClassify(AbstractModel):
    name = "LSTM_merge_classify"
    def __init__(self, train_path, dev_path, test_path,
                 debug=False):
        super().__init__(train_path, dev_path, test_path, debug)

    def train(self):
        self.model.fit(
            self.data.train_x,
            self.data.train_y,
            batch_size = self.props["batch_size"],
            nb_epoch = self.props["nb_epoch"],
            validation_data=(self.data.dev_x, self.data.dev_y)
        )
    def evaluate(self):
        score = self.model.evaluate(self.data.test_x, self.data.test_y,
                                    batch_size=128)
        return score

    def _get_data(self, train_path, dev_path, test_path):
        train_x = None
        train_y = None
        dev_x = None
        dev_y = None
        test_x = None
        test_y = None
        props = {}
        with open(train_path,'rb') as file:
            data = p.load(file)
            train_x = data['x']
            train_y = data['y']
            props["timesteps"] = data["timesteps"]
            props["data_dim"] = data["data_dim"]
            props["embeds"] = data["embeds"]
            props["batch_size"]= 64
            props["nb_epoch"]= 20

        with open(dev_path,'rb') as file:
            data = p.load(file)
            dev_x = data['x']
            dev_y = data['y']

        with open(test_path,'rb') as file:
            data = p.load(file)
            test_x = data['x']
            test_y = data['y']
        return Data(train_x, train_y, dev_x, dev_y, test_x, test_y), props


class LSTMMergeClassifyEmbeds(LSTMMergeClassify):
    name = "LSTM_bow_merge_classify_embeds"
    #__init__ properties data
    def _get_model(self):
        layers = []
        encoder_outcome = Sequential()
        encoder_outcome.add(Embedding(self.props["data_dim"]+1,
                                      200,
                                      input_length=self.props["timesteps"],
                                      weights=[self.props["embeds"]],
                                      dropout=.2,
                                      trainable=True,
                                      mask_zero=True))
        encoder_outcome.add(Dropout(0.2))
        encoder_outcome.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
        encoder_outcome.add(Dropout(0.2))

        encoder_selection = Sequential()
        encoder_selection.add(Embedding(self.props["data_dim"]+1,
                                        200,
                                        input_length=self.props["timesteps"],
                                        weights=[self.props["embeds"]],
                                        dropout=.2,
                                        trainable=True,
                                        mask_zero=True))
        encoder_selection.add(Dropout(0.2))
        encoder_selection.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
        encoder_selection.add(Dropout(0.2))

        decoder = Sequential()
        decoder.add(Merge([encoder_outcome, encoder_selection], mode='concat'))
        decoder.add(Dense(output_dim=32, activation='relu'))
        decoder.add(Dense(output_dim=3,
                        input_dim=32, activation="sigmoid"))

        decoder.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                        metrics=['accuracy'])
        return decoder

class LSTMMergeClassifyBOW(LSTMMergeClassify):
    name = "LSTM_bow_merge_classify_bow"
    #__init__ properties data
    def _get_model(self):
        layers = []
        print(self.props["embeds"][1])
        print(self.props["embeds"][2])
        encoder_outcome = Sequential()
        encoder_outcome.add(Embedding(self.props["data_dim"]+1,
                                      200,
                                      input_length=self.props["timesteps"],
                                      dropout=.2,
                                      trainable=True,
                                      mask_zero=True))
        encoder_outcome.add(Dropout(0.2))
        encoder_outcome.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
        encoder_outcome.add(Dropout(0.2))

        encoder_selection = Sequential()
        encoder_selection.add(Embedding(self.props["data_dim"]+1,
                                        200,
                                        input_length=self.props["timesteps"],
                                        dropout=.2,
                                        trainable=True,
                                        mask_zero=True))
        encoder_selection.add(Dropout(0.2))
        encoder_selection.add(LSTM(200, dropout_W=0.2, dropout_U=0.2))
        encoder_selection.add(Dropout(0.2))

        decoder = Sequential()
        decoder.add(Merge([encoder_outcome, encoder_selection], mode='concat'))
        decoder.add(Dense(output_dim=32, activation='relu'))
        decoder.add(Dense(output_dim=3,
                        input_dim=32, activation="sigmoid"))

        decoder.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                        metrics=['accuracy'])
        return decoder

