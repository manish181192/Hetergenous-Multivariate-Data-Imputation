import tensorflow as tf
from Model.Model import HI_VAE
from DataManager.DataManager import DataManager
import config
import os

if __name__=="__main__":

    data_manager = DataManager()
    hivae_model = HI_VAE(data_manager)

    session = tf.Session()

    globalStep = tf.constant(0)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_step = optimizer.minimize(hivae_model.elbo, global_step=globalStep)

    saver = tf.train.Saver(max_to_keep=5)
    latest_checkpoint = tf.train.latest_checkpoint(config.SAVE_DIR)
    if latest_checkpoint:
        saver.restore(session, latest_checkpoint)

    max_accuracy = -1
    for epoch in config.num_epochs:

        data_list, obs_matrix = data_manager.get_train_batch()
        # create feed dict
        fd = {}
        for feature_id, featureData in enumerate(data_list):
            fd[hivae_model.feature_placeholders[feature_id]] = data_list[feature_id]
        fd[hivae_model.observed_matrix] = obs_matrix

        _, loss = session.run([train_step, hivae_model.elbo], feed_dict= fd)
        print("Epoch: {} - Loss: {}".format(epoch, loss))
        # Evaluate test data
        accuracy = -1
        #todo Compute accuracy of the test data


        if accuracy> max_accuracy:
            print("### --- Saving model --- ###")
            model_path = os.path.join(config.SAVE_DIR, config.model_name)
            saver.save(session, model_path, globalStep)
            print("# Model Saved to path: {}".format(model_path))