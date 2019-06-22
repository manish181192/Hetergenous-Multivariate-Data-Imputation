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

    data_manager = DataManager(features_json="features_backup2.json")
    hivae_model = Model(data_manager)
    rnum = np.random.randint(0, 1000, 1)
    print("########  GENERATION TASK ID : {}  ########".format(rnum))
    print("find your generated features in file: Generated_samples/GenTask_{}".format(rnum))
    session = tf.Session()

    # globalStep = tf.Variable(initial_value=0)
    # optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    # train_step = optimizer.minimize(hivae_model.elbo, global_step=globalStep)

    saver = tf.train.Saver(max_to_keep=5)
    latest_checkpoint = tf.train.latest_checkpoint(config.SAVE_DIR)
    if config.restore_model and latest_checkpoint:
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
    loss, pred_params, likelihood, outputs, kl_s, kl_z = session.run([
                                                        hivae_model.elbo,
                                                        hivae_model.parameter_estimates,
                                                        tf.reduce_mean(hivae_model.likelihood_sum),
                                                        hivae_model.outputs,
                                                        tf.reduce_sum(hivae_model.kl_s),
                                                        tf.reduce_sum(hivae_model.kl_z)
                                                        ], feed_dict= fd)
    if config.save_gen_samples:
        def combine_samples(outputs):
            print("Samples \n")
            samples_dict = dict()
            for feature_id, feature in enumerate(data_manager.fields):
                samples_dict[feature["name"]] = outputs[feature_id]["samples"]
                print(feature["name"])
                print(outputs[feature_id]["samples"])
            return samples_dict
        def display_samples(samples_dict):
            for feature_name in samplesDict.keys():
                print("{}:{}".format(feature_name, samples_dict[feature_name]))

        file_path = "Generated_Samples/GenTask_"+str(rnum)+"_sampleDict.npy"
        if os.path.exists(file_path):
            os.remove(file_path)
        samplesDict = combine_samples(outputs)
        np.save(file_path, samplesDict)

    print("Numpy file saved ....!!!")
    print("Thank you")