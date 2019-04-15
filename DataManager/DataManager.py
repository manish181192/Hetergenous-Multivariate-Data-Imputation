import config
import pandas as pd
import numpy as np
import json

class DataManager(object):

    def __init__(self):
        self.df = pd.read_csv(config.DATA_PATH)

        # load specified features from JSON
        self.fields = json.load(open(config.FEATURE_JSON))
        config.num_features = len(self.fields)
        self.data_size = self.df.shape[0]

        # Drop other features
        self.filter_data()

        # get observation indices
        self.observed_indices = self.get_observed_matrix()

        # generate test indices
        self.test_indices_list = self.get_test_indices()

        #convert raw data to feature representation,
        self.process_features()

    def filter_data(self):
        feature_names = []
        for feature in self.fields:
            feature_names.append(feature["name"])
        self.df = self.df[feature_names]

    def get_observed_matrix(self):
        return self.df.notna().as_matrix()

    def get_test_indices(self):
        '''
        returns the test indices
        :return:  dict {feature:[B x #rep_dims]} - B is the config.batch_size
        '''
        test_indices_list = []
        for feature_id in range(config.num_features):
            feature_data = self.df.iloc[:, feature_id]
            observed_indices = np.where(self.observed_indices[:, feature_id]==True)[0]
            test_indices = np.random.choice(observed_indices, size= int(0.3*len(observed_indices)))
            test_indices_list.append(test_indices)

        return test_indices_list

    def process_features(self):

        features_list = []
        feature_vocab = []

        for feature_id, feature in enumerate(self.fields):

            if feature["type"] == "categorical":
                feature_representation = self.process_categorical_feature(feature)
                features_list.append(feature_representation)

            elif feature["type"] == "real":
                pass

            elif feature["type"] == "ordinal":
                pass

    def process_categorical_feature(self, feature):

        feature_vocab = np.load(config.vocab_dir + feature["name"] + ".npy").item()
        total_features = feature["ndims"]

        feature_numpy = np.cast(self.df[feature["name"]].map(feature_vocab).as_matrix(), int)

        onehot_feature = np.zeros((self.data_size, total_features))
        onehot_feature[np.arange(self.data_size), feature_numpy] =1
        return onehot_feature

    def process_ordinal_feature(self, feature_name):
        feature_vocab = np.load(config.vocab_dir + feature_name + ".npy")
        self.df[feature_name].map(feature_vocab)


    ###### BATCHER #####

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
        train_indices = self.train_sequence[self.current_train_index:self.current_train_index+config.batch_size]

        train_data = []
        for feature in self.fields:
            pass

if __name__=="__main__":
    dm = DataManager()
