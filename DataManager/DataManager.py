from numpy.core.multiarray import dtype

from config import Config as config
import pandas as pd
import numpy as np
import json

class DataManager(object):

    def __init__(self, features_json=None):
        self.df = pd.read_csv(config.DATA_PATH)


        # load specified features from JSON
        if features_json == None:
            features_json = config.FEATURE_JSON
        self.fields = json.load(open(features_json))
        self.data_size = self.df.shape[0]

        # Drop other features
        self.filter_data()
        config.num_features = len(self.df.columns)

        #preprocessDF
        self.df_preprocess()

        # get observation indices
        self.observed_indices = self.get_observed_matrix()

        # generate test indices
        self.test_indices_list = self.get_test_indices()

        #convert raw data to feature representation,
        self.features_list = self.process_features()

    def filter_data(self):
        print(" ---- Features included ---- ")
        feature_names = []
        total_dims = 0
        for feature_id in range(len(self.fields)):
            if self.fields[feature_id]["type"]=="nlp" and config.nlp==False:
                del self.fields[feature_id]
                continue
            print(self.fields[feature_id]["name"])
            feature_names.append(self.fields[feature_id]["name"])
            total_dims+=self.fields[feature_id]["ndims"]

        self.df = self.df[feature_names]
        self.total_dims = total_dims
        print("------------------------------\n\n")

    def get_observed_matrix(self):
        return self.df.notna().values.astype(int)

    def df_preprocess(self):

        # age field preprocessing
        col = []
        if len(col)>0:
            self.df[col] = self.df[col].fillna(-1)
            self.df[col] = self.df[col].astype(int)
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].replace('-1', np.nan)

        # # Participant_ID field preprocessing
        # col = "Participant_ID"
        # self.df[col] = self.df[col].fillna(-1)
        # self.df[col] = self.df[col].astype(int)
        # self.df[col] = self.df[col].astype(str)
        # self.df[col] = self.df[col].replace('-1', np.nan)


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
            elif feature["type"] == "multilabel":
                multilabel_data = np.load(config.MULTILABEL_DIR+feature["name"]+".npy")
                features_list.append(multilabel_data)

            elif feature["type"] == "real":
                feature_representation = self.process_real_feature(feature)
                features_list.append(feature_representation)

            elif feature["type"] == "nlp":
                nlp_ip_seq = np.load(config.NLP_PREPROCESSED_DATA+feature["name"]+".npy")
                features_list.append(nlp_ip_seq)
        return features_list

    def process_categorical_feature(self, feature):

        feature_vocab = np.load(config.vocab_dir + feature["name"] + ".npy").item()
        total_features = feature["ndims"]
        feature_numpy = self.df[feature["name"]].map(feature_vocab).values.astype(int)

        onehot_feature = np.zeros((self.data_size, total_features), dtype=int)
        for i, id in enumerate(feature_numpy):
            if id>=0:
                onehot_feature[i, id] =1
        return onehot_feature

    def process_real_feature(self, feature):

        feature_numpy = self.df[feature["name"]].fillna(0.).values
        feature_numpy = np.expand_dims(feature_numpy, axis=-1)
        return feature_numpy

    def process_ordinal_feature(self, feature_name):
        feature_vocab = np.load(config.vocab_dir + feature_name + ".npy")
        self.df[feature_name].map(feature_vocab)


    ###### BATCHER #####

    def init_train_batcher(self, shuffle=False):
        '''
        Rese the batcher
        :return:
        '''
        self.current_train_index = 0
        self.train_sequence = np.arange(self.data_size)
        if shuffle: np.random.shuffle(self.train_sequence)

    def get_train_batch(self, batch_size=None):
        '''
        generate the batches given batch_size
        :return: dict {feature:[B x #re p_dims]} - B is the config.batch_size
        '''
        if batch_size==None:
            batch_size=config.batch_size
        if batch_size == None:
            return self.features_list, self.observed_indices
        else:

            if self.current_train_index+batch_size>self.data_size:
                return None
            train_indices = self.train_sequence[self.current_train_index:self.current_train_index+config.batch_size]

            # list of train data per feature
            train_data_list = []
            for feature_id, feature in enumerate(self.fields):
                if feature["type"] == "categorical":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
                elif feature["type"] == "multilabel":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
                elif feature["type"] == "real":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
                elif feature["type"] == "posReal":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
                elif feature["type"] == "ordinal":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
                elif feature["type"] == "nlp":
                    train_data_list.append(self.retreive_batch(feature_id, train_indices))
            observation_matrix_batch = self.observed_indices[train_indices, :]
            return train_data_list, observation_matrix_batch

    def retreive_batch(self, feature_id, train_indices):
        return self.features_list[feature_id][train_indices, :]



if __name__=="__main__":
    dm = DataManager()
