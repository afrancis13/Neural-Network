import csv
import numpy as np
import scipy.io
import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from network import NeuralNetwork
from utils import cross_validate, benchmark, shuffle_in_unison_inplace, plot_image

import pickle

# Configure these variables to run specific parts of the script.
RUN_XOR = False
RUN_CROSS_VALIDATION = True
RUN_MSE = False
RUN_CROSS_ENTROPY = True
RUN_KAGGLE = True
PLOTS = False

##########################################
#                  XOR                   #
##########################################

if RUN_XOR:
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    data = np.array([np.array(x) for x in data])

    labels = np.array([0, 1, 1, 0]).reshape(4, 1)

    nn = NeuralNetwork(data, labels, hidden_layer_size=2, learning_rate=0.01, decreasing_rate=False)

    V_xor, W_xor, xor_training_error_array, xor_training_loss_array = nn.train()
    l = nn.predict(data, V_xor, W_xor)
    ll = np.argmax(l)
    print(l)

##########################################
#         Process Digits Dataset         #
##########################################

train_images = scipy.io.loadmat(file_name='../dataset/train.mat')

train_matrix_images = train_images['train_images']
train_labels_images = train_images['train_labels'].flatten()

test_images_file_dump = scipy.io.loadmat(file_name='../dataset/test.mat')
test_images_matrix = test_images_file_dump['test_images']
test_plot_images = np.reshape(test_images_matrix, (10000, 28, 28))
test_images = np.reshape(test_images_matrix, (10000, 28 * 28))

# Test the test_images matrix.
if PLOTS:
    for i in range(15):
        plot_image(test_plot_images[i])

train_images_matrix = np.swapaxes(np.swapaxes(train_matrix_images, 0, 1), 0, 2)
train_plot_images_matrix = np.reshape(train_images_matrix, (train_images_matrix.shape[0], 28, 28))
reshaped_images_matrix = np.reshape(train_images_matrix, (train_images_matrix.shape[0], 28 * 28))

# Test the reshaped_images_matrix
if PLOTS:
    for i in range(5):
        plot_image(train_plot_images_matrix[48000 + i], train_labels_images[48000 + i])

################################################
# Cross-Validation & Plots, Mean Squared Error #
################################################

if RUN_CROSS_VALIDATION:
    k = 6

    learning_rates = [0.01]
    training_error_rates_mse = []
    validation_error_rates_mse = []
    training_error_rates_cross_entropy = []
    validation_error_rates_cross_entropy = []

    file_number = 0

    for epsilon in learning_rates:
        cv_sets = cross_validate(k, reshaped_images_matrix, train_labels_images.reshape(60000, 1))
        errors_epsilon_train_mse = np.array([])
        errors_epsilon_validation_mse = np.array([])
        errors_epsilon_train_cross_entropy = np.array([])
        errors_epsilon_validation_cross_entropy = np.array([])
        for cv_set in cv_sets:
            training_X, training_y, validation_X, validation_y = cv_set

            if RUN_MSE:
                print("Training Neural Network with MSE loss function for k-fold Cross-Validation. Size of image set is %d."
                      % training_X.shape[0])

                classifier_mse = NeuralNetwork(
                    training_X,
                    training_y,
                    validation_data=validation_X,
                    validation_labels=validation_y,
                    hidden_layer_size=200,
                    learning_rate=epsilon,
                    decreasing_rate=True
                )

                pre_training_time_cross_validation_mse = time.time()
                trained_V_mse, trained_W_mse, train_error_array_mse, train_loss_array_mse = classifier_mse.train()
                post_training_time_cross_validation_mse = time.time()
                training_time_cross_validation_mse = post_training_time_cross_validation_mse - pre_training_time_cross_validation_mse

                # Save state.
                file_V_mse_name = "V_matrix_mse%d.txt" % file_number
                file_W_mse_name = "W_matrix_mse%d.txt" % file_number
                pickle.dump(trained_V_mse, open(file_V_mse_name, "wb"))
                pickle.dump(trained_W_mse, open(file_W_mse_name, "wb"))

                print("Saved matrices to files on the local machine.")

                if PLOTS:
                    plt.plot(range(len(train_error_array_mse)), train_error_array_mse)
                    plt.title("Training Error vs. Number of Iterations")
                    plt.ylabel("Classification Error (%)")
                    plt.xlabel("Number of Iterations")
                    plt.show()

                    plt.plot(range(len(train_loss_array_mse)), train_loss_array_mse)
                    plt.title("Mean Squared Loss vs. Number of Iterations")
                    plt.ylabel("Mean Squared Loss")
                    plt.xlabel("Number of Iterations")
                    plt.show()

                print("Finished training Neural Network with MSE loss function. Training time was %.4f."
                      % training_time_cross_validation_mse)

                training_predictions_mse = classifier_mse.predict(training_X, trained_V_mse, trained_W_mse)
                training_error_mse, indices_training_mse = benchmark(training_predictions_mse, training_y)
                errors_epsilon_train_mse = np.append(np.array([training_error_mse]), errors_epsilon_train_mse)

                validation_predictions_mse = classifier_mse.predict(validation_X, trained_V_mse, trained_W_mse)
                validation_error_mse, indices_validation_mse = benchmark(validation_predictions_mse, validation_y)
                errors_epsilon_validation_mse = np.append(np.array([validation_error_mse]), errors_epsilon_validation_mse)

                print("The error rate on the validation set with MSE loss function is %.4f." % validation_error_mse)

            if RUN_CROSS_ENTROPY:
                print("Training Neural Network with cross-entropy loss function for k-fold Cross-Validation. Size of image set is %d."
                      % training_X.shape[0])

                classifier_cross_entropy = NeuralNetwork(
                    training_X,
                    training_y,
                    validation_data=validation_X,
                    validation_labels=validation_y,
                    hidden_layer_size=200,
                    loss_function="cross-entropy",
                    learning_rate=epsilon,
                    decreasing_rate=True
                )
                pre_training_time_cross_validation_cross_entropy = time.time()
                trained_V_cross_entropy, trained_W_cross_entropy, train_error_array_cross_entropy, train_loss_array_cross_entropy = classifier_cross_entropy.train()
                post_training_time_cross_validation_cross_entropy = time.time()
                training_time_cross_validation_cross_entropy = post_training_time_cross_validation_cross_entropy - pre_training_time_cross_validation_cross_entropy

                # Save state.
                pickle.dump(trained_V_cross_entropy, open("V_matrix_cross_entropy%d.txt" % file_number, "wb"))
                pickle.dump(trained_W_cross_entropy, open("W_matrix_cross_entropy%d.txt" % file_number, "wb"))

                print("Saved matrices to files on the local machine.")
                if PLOTS:
                    plt.plot(range(len(train_error_array_cross_entropy)), train_error_array_cross_entropy)
                    plt.title("Training Error vs. Number of Iterations")
                    plt.ylabel("Classification Error (%)")
                    plt.xlabel("Number of Iterations")
                    plt.show()

                    plt.plot(range(len(train_loss_array_cross_entropy)), train_loss_array_cross_entropy)
                    plt.title("Cross Entropy Loss vs. Number of Iterations")
                    plt.ylabel("Cross Entropy Loss")
                    plt.xlabel("Number of Iterations")
                    plt.show()

                print("Finished training Neural Network with cross-entropy loss function. Training time was %.4f."
                      % training_time_cross_validation_cross_entropy)

                training_predictions_cross_entropy = classifier_cross_entropy.predict(training_X, trained_V_cross_entropy, trained_W_cross_entropy)
                training_error_cross_entropy, indices_training_cross_entropy = benchmark(training_predictions_cross_entropy, training_y)
                errors_epsilon_train_cross_entropy = np.append(np.array([training_error_cross_entropy]), errors_epsilon_train_cross_entropy)

                validation_predictions_cross_entropy = classifier_cross_entropy.predict(validation_X, trained_V_cross_entropy, trained_W_cross_entropy)
                validation_error_cross_entropy, indices_validation_cross_entropy = benchmark(validation_predictions_cross_entropy, validation_y)
                errors_epsilon_validation_cross_entropy = np.append(np.array([validation_error_cross_entropy]), errors_epsilon_validation_cross_entropy)

                print("The error rate on the validation set with cross-entropy loss function is %.4f." % validation_error_cross_entropy)

            file_number += 1

        if RUN_MSE:
            average_error_rate_training_mse = np.mean(errors_epsilon_train_mse)
            training_error_rates_mse.append(average_error_rate_training_mse)

            average_error_rate_validation_mse = np.mean(errors_epsilon_validation_mse)
            validation_error_rates_mse.append(average_error_rate_training_mse)

            print("Finished cross validation for parameter epsilon = %.2f with MSE loss function." % epsilon)
            print("The average error rate on the training set for parameter epsilon = %.2f with MSE loss function is %.2f."
                  % (epsilon, average_error_rate_training_mse))
            print("The average error rate on the validation set for parameter epsilon = %.2f with MSE loss function is %.2f."
                  % (epsilon, average_error_rate_validation_mse))

        if RUN_CROSS_ENTROPY:
            average_error_rate_training_cross_entropy = np.mean(errors_epsilon_train_cross_entropy)
            training_error_rates_cross_entropy.append(average_error_rate_training_cross_entropy)

            average_error_rate_validation_cross_entropy = np.mean(errors_epsilon_train_cross_entropy)
            validation_error_rates_cross_entropy.append(average_error_rate_validation_cross_entropy)

            print("Finished cross validation for parameter epsilon = %.2f with cross-entropy loss function." % epsilon)
            print("The average error rate on the training set for parameter epsilon = %.2f with cross-entropy loss function is %.2f."
                  % (epsilon, average_error_rate_training_cross_entropy))
            print("The average error rate on the validation set for parameter epsilon = %.2f with cross-entropy loss function is %.2f."
                  % (epsilon, average_error_rate_validation_cross_entropy))

    if RUN_MSE:
        best_epsilon_training_mse = learning_rates[np.argmax(np.array(training_error_rates_mse))]
        print("The best learning rate for the training set using the MSE loss function is %.2f." % best_epsilon_training_mse)

        best_epsilon_validation_mse = learning_rates[np.argmax(np.array(validation_error_rates_mse))]
        print("The best learning rate for the validation set using the MSE loss function is %.2f." % best_epsilon_validation_mse)

    if RUN_CROSS_ENTROPY:
        best_epsilon_training_cross_entropy = learning_rates[np.argmax(np.array(training_error_rates_cross_entropy))]
        print("The best learning rate for the training set using the cross-entropy loss function is %.2f." % best_epsilon_training_cross_entropy)

        best_epsilon_validation_cross_entropy = learning_rates[np.argmax(np.array(validation_error_rates_cross_entropy))]
        print("The best learning rate for the validation set using the cross-entropy loss function is %.2f." % best_epsilon_validation_cross_entropy)

##########################################
#         Kaggle Predictions             #
##########################################

if RUN_KAGGLE:
    shuffled_image_matrix, shuffled_train_labels = shuffle_in_unison_inplace(
        reshaped_images_matrix, train_labels_images.reshape(60000, 1)
    )

    test_classifier = NeuralNetwork(
        shuffled_image_matrix,
        shuffled_train_labels,
        hidden_layer_size=200,
        learning_rate=0.01,
        decreasing_rate=False
    )

    # The important benchmark.
    pre_training_time = time.time()
    V, W, training_error_array, training_loss_array = test_classifier.train()
    post_training_time = time.time()

    V = pickle.load(open("V_matrix_mse0.txt", "rb"))
    W = pickle.load(open("W_matrix_mse0.txt", "rb"))

    print("The total training time for the Neural Network classifier is %.4f" % post_training_time)

    # Save state.
    file_V_test_name = "V_matrix_test.txt"
    file_W_test_name = "W_matrix_test.txt"

    pickle.dump(V, open(file_V_test_name, "wb"))
    pickle.dump(W, open(file_W_test_name, "wb"))

    pickle.dump(training_error_array, open(file_V_test_name, "wb"))
    pickle.dump(training_loss_array, open(file_W_test_name, "wb"))

    test_predictions = test_classifier.predict(scale(test_images), V, W)
    test_predictions = test_predictions.reshape((test_predictions.shape[0],))

    with open("../kaggle/kaggle_digits.csv", "w") as csvfile:
        digit_writer = csv.writer(csvfile)
        digit_writer.writerow(['Id', 'Category'])
        for i in range(len(test_predictions)):
            digit_writer.writerow([i + 1, test_predictions[i]])

    print("Drum Roll please...\nThe image classifications for the test set are:\n%s" % str(test_predictions))
