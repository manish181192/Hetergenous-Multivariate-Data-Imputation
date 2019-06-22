import tensorflow as tf
from Model.Model import Model
from DataManager.DataManager import DataManager
from config import Config as config
import os
import numpy as np
from evaluation import calculate_accuracy, get_bleu_score
import matplotlib.pyplot as plt
from utils import get_generated_sentence


import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":

    #Define our model
    # data_manager = DataManager(features_json="features_backup2.json")



    data_manager = DataManager()
    hivae_model = Model(data_manager)

    session = tf.Session()
    globalStep = tf.Variable(initial_value=0)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_step = optimizer.minimize(hivae_model.elbo, global_step=globalStep)

    saver = tf.train.Saver(max_to_keep=5)
    latest_checkpoint = tf.train.latest_checkpoint(config.SAVE_DIR)

    # Restore checkpoints - continue training
    if config.restore_model and latest_checkpoint:
        print(" --- #### Restoring Session ###  ")
        saver.restore(session, latest_checkpoint)
    else:
        print(" ---- ### Initializing fresh parameters ###")
        session.run(tf.global_variables_initializer())

    print("Training started ....")

    #Gather data for plots
    plots_likelihoods = []
    plots_loss = []
    plots_train_accuracy = []
    plots_test_accuracy = []
    plots_epochs = []

    # Start Training
    max_accuracy = -1
    for epoch in range(config.num_epochs):

        #Initiate training batch
        data_manager.init_train_batcher()
        data_list, obs_matrix = data_manager.get_train_batch()

        # Prepare inputs
        fd = {}
        if config.nlp:
            nlp_feature_id =0
        for feature_id, featureData in enumerate(data_list):
            if data_manager.fields[feature_id]["type"]!="nlp":
                fd[hivae_model.feature_placeholders[feature_id]] = data_list[feature_id]
            else:
                if config.nlp:
                    fd[hivae_model.nlp_texts_ids[nlp_feature_id]] = data_list[feature_id]
                    nlp_feature_id+=1

        fd[hivae_model.observed_matrix] = obs_matrix
        fd[hivae_model.tau] = 10.0
        fd[hivae_model.num_samples] = config.num_samples
        _, loss, pred_params, likelihood, outputs, kl_s, kl_z = session.run([train_step,
                                                            hivae_model.elbo,
                                                            hivae_model.parameter_estimates,
                                                            tf.reduce_mean(hivae_model.likelihood_sum),
                                                            hivae_model.outputs,
                                                            tf.reduce_sum(hivae_model.kl_s),
                                                            tf.reduce_sum(hivae_model.kl_z)
                                                            ], feed_dict= fd)


        # sample sentence using the logits
        if config.nlp:
            nlp_logits = outputs[-1]["params"]
            nlp_samples = get_generated_sentence(nlp_logits)
            avg_bleu_score = get_bleu_score(reference=data_list[-1],
                                            predictions=nlp_samples)
            print("BLEU Score : {}".format(avg_bleu_score))
        # Evaluate test data
        feature_train_accuracy, feature_test_accuracy, avg_train_accuracy, avg_test_accuracy = \
            calculate_accuracy(data_list=data_list,
                               obs_matrix=obs_matrix,
                               parameter_estimates=pred_params,
                               data_manager=data_manager)

        # Gather data for plots
        plots_likelihoods.append(likelihood)
        plots_loss.append(loss)
        plots_train_accuracy.append(avg_train_accuracy)
        plots_test_accuracy.append(avg_test_accuracy)
        plots_epochs.append(epoch)

        # Print output ( Turn verbose_output flag on for more detailed output)
        if epoch%config.test_epochs==0:
            print("\n=================================================================")
            print("Epoch: {} - AvgTrainAcc:{} - AvgTestAcc:{} - Loss(-elbo): {} - lik: {}".format(epoch,
                                                                                avg_train_accuracy,
                                                                                avg_test_accuracy,
                                                                                loss,
                                                                                likelihood,
                                                                                kl_s,
                                                                                kl_z))
            if config.verbose_ouput:
                print("---------------------------------------------------------")
                print("feature-wise outputs")
                for feature_id, feature in enumerate(data_manager.fields):
                    print("name: {} - negloglik: {} - acc(train::test): {}::{}".format(feature["name"],
                                                                            outputs[feature_id]["likelihood"],
                                                                            feature_train_accuracy[feature_id],
                                                                            feature_test_accuracy[feature_id]))

            # Save the best model till now
            if config.save_model and avg_test_accuracy> max_accuracy:
                print("### --- Saving model --- ###")
                model_path = os.path.join(config.SAVE_DIR, config.model_name)
                saver.save(session, model_path, globalStep)
                print("# Model Saved to path: {}\n\n".format(model_path))
                max_accuracy = avg_test_accuracy

            # Generate Samples for the features
            if config.save_gen_samples:
                print("Generating samples for Iteration {}".format(epoch))
                def combine_samples(outputs):
                    samples_dict = dict()
                    for feature_id, feature in enumerate(data_manager.fields):
                        if feature["type"]!="nlp":
                            samples_dict[feature["name"]] = outputs[feature_id]["samples"]
                    return samples_dict


                def display_samples(samples_dict):
                    for feature_id, feature in enumerate(data_manager.fields):
                        feature_name = feature["name"]
                        samples_raw = samples_dict[feature_name][0][:config.num_samples]
                        if feature['type'] == 'categorical' or feature['type'] == 'multilabel':
                            samples = []
                            vocab = np.load(config.reverse_vocab_dir + feature["name"] + ".npy").item()
                            sample_ids = np.argmax(samples_raw, axis=-1)
                            for id in sample_ids:
                                samples.append(vocab[id])
                            print("{}:{}".format(feature_name, samples))

                        else:
                            print("{}:{}".format(feature_name, samples_raw))

                file_path = "Generated_Samples/Iteration_"+str(epoch)+"_sampleDict.npy"
                if os.path.exists(file_path):
                    os.remove(file_path)
                samplesDict = combine_samples(outputs)
                np.save(file_path, samplesDict)
                if config.display_samples: display_samples(samplesDict)

    #plot gathered data
    plt.plot(plots_epochs, plots_likelihoods, label="likelihood")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

    plt.plot(plots_epochs, plots_loss, label="loss")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

    plt.plot(plots_epochs, plots_train_accuracy, label="train accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

    plt.plot(plots_epochs, plots_test_accuracy, label="test accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

