import tensorflow as tf
from Model.Model import Model
from DataManager.DataManager import DataManager
from config import Config as config
import os
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":

    data_manager = DataManager()
    hivae_model = Model(data_manager)
    session = tf.Session()

    print("Generating Feature covariance matrix")

    saver = tf.train.Saver(max_to_keep=5)
    latest_checkpoint = tf.train.latest_checkpoint(config.SAVE_DIR)
    if latest_checkpoint:
        print(" --- #### Restoring Session ###  ")
        saver.restore(session, latest_checkpoint)
    else:
        print("No saved model to restore..!!")
        exit(0)

    data_manager.init_train_batcher()
    data_list, obs_matrix = data_manager.get_train_batch()
    # create feed dict
    fd = {}
    for feature_id, featureData in enumerate(data_list):
        fd[hivae_model.feature_placeholders[feature_id]] = data_list[feature_id]
    fd[hivae_model.observed_matrix] = obs_matrix
    fd[hivae_model.tau] = 10.0
    fd[hivae_model.num_samples] = 10
    grouped_y = session.run([
                            hivae_model.grouped_y
                            ], feed_dict= fd)
    grouped_y = grouped_y[0]
    def compute_covariance(grouped_y):

        cov = np.zeros(shape=[len(grouped_y), len(grouped_y)], dtype=float)
        for i in range(len(grouped_y)):
            for j in range(len(grouped_y)):
                cov[i, j] = np.dot(grouped_y[i], grouped_y[j])
                cov[j, i] = cov[i, j]
        return cov

    mean_grouped_y = []
    for y in grouped_y:
        mean_grouped_y.append(np.mean(y, axis=0))
    print(compute_covariance(mean_grouped_y))