import numpy as np
from utils import softmax, sigmoid
from nltk.translate.bleu_score import sentence_bleu

def get_bleu_score(reference, predictions):
    bleu_sum =0.
    for id, predicted_sentence in enumerate(predictions):
        bleu_sum += sentence_bleu([reference[id, :].tolist()], predicted_sentence.tolist(), weights=[0.25, 0.25, 0.25, 0.25])
    return float(bleu_sum)/(reference.shape[0])

def calculate_accuracy(data_list, obs_matrix, parameter_estimates, data_manager):


    feature_train_accuracies = []
    feature_test_accuracies = []

    for feature_id, feature in enumerate(data_manager.fields):

        total_train_correct = 0
        total_train_count = 0
        total_test_correct = 0
        total_test_count = 0

        if feature["type"] == "real":

            mean_est, logvar_est = parameter_estimates[feature_id]
            std_est = np.sqrt(np.exp(logvar_est))

            data = data_list[feature_id]
            obs_list = obs_matrix[:, feature_id]
            test_list = data_manager.test_indices_list[feature_id]

            for data_index, is_obs in enumerate(obs_list):

                if is_obs ==1:
                    data_point = data[data_index][0]
                    mean_est_point = mean_est[data_index][0]
                    std_est_point = std_est[data_index][0]
                    epsilon = std_est_point
                    # epsilon = 0.01*std_est_point
                    if data_index not in test_list:
                        # data point is in trainset
                        if data_point>=(mean_est_point-epsilon) and data_point<=(mean_est_point+epsilon):
                            total_train_correct+=1
                        total_train_count+=1
                    else:
                        # data point in test set
                        if data_point>=(mean_est_point-std_est_point) and data_point<=(mean_est_point+std_est_point):
                            total_test_correct+=1
                        total_test_count+=1

            if total_train_count>0:
                feature_train_accuracies.append(float(total_train_correct)/total_train_count)
            else:
                feature_train_accuracies.append(0.)
            if total_test_count>0:
                feature_test_accuracies.append(float(total_test_correct)/total_test_count)
            else:
                feature_test_accuracies.append(0.)
        elif feature["type"] == "categorical":
            probs = softmax(parameter_estimates[feature_id])
            predictions = np.argmax(probs, axis=-1)

            data = np.argmax(data_list[feature_id], axis=-1)
            obs_list = obs_matrix[:, feature_id]
            test_list = data_manager.test_indices_list[feature_id]

            for data_index, is_obs in enumerate(obs_list):
                if is_obs == 1:
                    data_point = data[data_index]
                    pred = predictions[data_index]
                    if data_index not in test_list:
                        # datapoint is from train set
                        if pred == data_point:
                            total_train_correct+=1
                        total_train_count+=1
                    else:
                        #datapoint is from test set
                        if pred == data_point:
                            total_test_correct +=1
                        total_test_count+=1
            if total_train_count>0:
                feature_train_accuracies.append(float(total_train_correct)/total_train_count)
            else:
                feature_train_accuracies.append(0.)
            if total_test_count>0:
                feature_test_accuracies.append(float(total_test_correct)/total_test_count)
            else:
                feature_test_accuracies.append(0.)

        elif feature["type"] == "multilabel":
            probs = sigmoid(parameter_estimates[feature_id])
            super_threshold_indices = probs < 0.5
            probs[super_threshold_indices] = 0.
            super_threshold_indices = probs > 0.5
            probs[super_threshold_indices] = 1.

            # predictions = np.argwhere(probs == 1)[0]

            # data = np.argmax(data_list[feature_id], axis=-1)
            # data = np.argwhere(data_list[feature_id]==1)[0]
            prob_mul_data = np.multiply(probs, data_list[feature_id])
            pmd_sum = np.sum(prob_mul_data, axis=-1)
            actual_sum = np.sum(data_list[feature_id], axis=-1)
            obs_list = obs_matrix[:, feature_id]
            test_list = data_manager.test_indices_list[feature_id]

            for data_index, is_obs in enumerate(obs_list):
                if is_obs == 1:
                    if data_index not in test_list:
                        # datapoint is from train set
                        total_train_correct += pmd_sum[data_index]
                        total_train_count += actual_sum[data_index]
                    else:
                        #datapoint is from test set
                        total_test_correct += pmd_sum[data_index]
                        total_test_count += actual_sum[data_index]
            if total_train_count>0:
                feature_train_accuracies.append(float(total_train_correct)/total_train_count)
            else:
                feature_train_accuracies.append(0.)
            if total_test_count>0:
                feature_test_accuracies.append(float(total_test_correct)/total_test_count)
            else:
                feature_test_accuracies.append(0.)


    avg_train_accuracy = float(sum(feature_train_accuracies))/len(feature_train_accuracies)
    avg_test_accuracy = float(sum(feature_test_accuracies)) / len(feature_test_accuracies)
    return feature_train_accuracies, feature_test_accuracies, avg_train_accuracy, avg_test_accuracy
