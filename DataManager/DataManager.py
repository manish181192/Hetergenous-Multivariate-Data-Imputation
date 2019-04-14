import config
import pandas as pd
import numpy as np

class DataManager(object):

    def __init__(self):
        self.df = pd.read_csv(config.DATA_PATH)
        self.data_size = len(list(self.df))

        if config.features!=None:
            self.df.drop(self.df.columns[config.features], axis=1, inplace=True)
        self.feature_naIndices_dict = self.get_na_indices()
        self.feature_testIndices_dict = self.get_test_indices()

    def get_na_indices(self):
        '''
        create a dict of na indices
        :return:
        '''
        pass

    def get_test_indices(self):
        '''
        returns the test indices
        :return:  dict {feature:[B x #rep_dims]} - B is the config.batch_size
        '''
        pass

    def init_train_batcher(self):
        '''
        Rese the batcher
        :return:
        '''
        self.current_train_index = 0
        self.train_sequence = np.random.shuffle(np.arange(self.data_size))

    def get_train_batch(self):
        '''
        generate the batches given batch_size
        :return: dict {feature:[B x #rep_dims]} - B is the config.batch_size
        '''
        pass