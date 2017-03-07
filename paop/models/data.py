import numpy as np
class Data:
    """
    Data representation:
        separates train, dev, and test sets
        holds meta data about data properties
    """
    def __init__(self, train_x, train_y,
                 dev_x, dev_y, test_x,
                 test_y, timesteps=None,
                 data_dim=200, #Initialized to gensim word2vec dim
                 vocab_size=0):

        self.train_x = train_x
        self.train_y = train_y

        self.dev_x = dev_x
        self.dev_y = dev_y

        self.test_x = test_x
        self.test_y = test_y

        self.timesteps = timesteps
        self.data_dim = data_dim
        self.vocab_size = vocab_size

