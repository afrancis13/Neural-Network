import numpy as np
import math

from sklearn.preprocessing import scale

from utils import compute_sigmoid, benchmark, cross_entropy_loss, mean_squared_error


class NeuralNetwork(object):

    def __init__(self, data, labels, validation_data=None, validation_labels=None,
                 hidden_layer_size=0, loss_function="mean-squared-error",
                 learning_rate=1.0, decreasing_rate=False):
        self.input_layer_size = data.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = len(np.unique(labels))

        if loss_function not in ("mean-squared-error", "cross-entropy"):
            raise ValueError("Loss function must be 'mean-squared-error' or 'cross-entropy'.")

        self.loss_function = loss_function

        self.data = scale(data)
        self.labels = labels

        self.Y = np.zeros((data.shape[0], self.output_layer_size))
        for i in range(data.shape[0]):
            label_i = labels[i]
            self.Y[i][label_i] = 1

        self.learning_rate = learning_rate
        self.decreasing_rate = decreasing_rate

        if validation_data is not None:
            self.validation_data = scale(validation_data)
        else:
            self.validation_data = None

        self.validation_labels = validation_labels

    def train(self):
        weights_v = 0.01 * np.random.randn(self.hidden_layer_size, self.input_layer_size + 1)
        weights_w = 0.01 * np.random.randn(self.output_layer_size, self.hidden_layer_size + 1)

        # The specification recommends computing the post-prediction cost or the
        # magnitude of the gradient to determine the stopping condition. This is
        # carried out in the stop_sgd function in the utils section of this class.
        num_iter = 1
        gradient_v, gradient_w = None, None

        training_error_array = []
        training_loss_array = []
        converged, training_error, training_loss = self.converged_gradient(
            num_iter, self.data, weights_v, weights_w
        )

        epoch_size = self.data.shape[0]
        epoch_sample_num = 0

        while not converged:
            if training_error is not None and training_loss is not None:
                training_error_array.append(1 - training_error[0])
                training_loss_array.append(training_loss)

            random_index = np.random.randint(self.data.shape[0])
            random_data_point = self.data[random_index]
            random_label = self.labels[random_index]

            forward_pass_list = self.perform_forward_pass(random_data_point, weights_v, weights_w)

            gradient_v, gradient_w = self.perform_backward_pass(
                random_data_point,
                random_label,
                weights_v,
                weights_w,
                forward_pass_list
            )

            weights_v = np.subtract(weights_v, self.learning_rate_this_iteration(num_iter) * gradient_v)
            weights_w = np.subtract(weights_w, self.learning_rate_this_iteration(num_iter) * gradient_w)

            converged, training_error, training_loss = self.converged_gradient(
                num_iter,
                self.data,
                weights_v,
                weights_w,
                gradient_v=gradient_v,
                gradient_w=gradient_w
            )

            num_iter += 1

            if epoch_sample_num < epoch_size - 1:
                epoch_sample_num += 1
            else:
                print("Finished an epoch.")
                epoch_sample_num = 0

        return weights_v, weights_w, training_error_array, training_loss_array

    def predict(self, X, V, W, return_Z=False):
        '''
        The logistics of prediction follow similar logic to that presented in the write up.

        Once we've trained weight matrices V and W, we compute the hidden layer output for
        each sample in X (the data matrix) by computing H = tanh(np.dot(V, X.T)) using np.vectorize.
        Note that H.shape = (n_hid, num_samples). This is a bit of an issue since we would
        like to add a bias term to the model, so we append a row of 1s to the matrix in order
        to make H.shape = (n_hid + 1, num_samples).

        We then compute the matrix Z = s(np.dot(W, H)).
        This results in a matrix of size Z.shape = (n_out, num_samples). By taking the argmax over
        the columns of the matrix, we compute num_samples predictions, and complete the
        classification algorithm.
        '''
        print("Starting the prediction algorithm.")

        sigmoid_vectorized = np.vectorize(compute_sigmoid)
        tanh_vectorized = np.vectorize(math.tanh)

        X = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0], 1), 1)
        H = tanh_vectorized(np.dot(V, X.T))
        H = np.vstack((H, np.ones(H.shape[1])))
        Z = sigmoid_vectorized(np.dot(W, H))

        print("Completed the prediction algorithm.")

        classifications = np.argmax(Z, 0)
        classifications_as_vector = classifications.reshape(len(classifications), 1)
        if not return_Z:
            return classifications_as_vector
        else:
            return classifications_as_vector, Z


    def perform_forward_pass(self, x_j, V, W):
        '''
        In the forward pass stage, we compute z, h and return it as input for
        the backward pass (see below). This resembles the predict function to
        some degree, but for a single data point.
        '''
        sigmoid_vectorized = np.vectorize(compute_sigmoid)
        tanh_vectorized = np.vectorize(math.tanh)

        x_j = np.append(x_j, 1)

        h = tanh_vectorized(np.dot(V, x_j))
        h = np.append(h, 1)

        z_linear = np.dot(W, h)
        z = sigmoid_vectorized(z_linear).reshape(z_linear.shape[0], 1)

        return [h, z]

    def perform_backward_pass(self, x_j, y_j, V, W, forward_pass_list):
        '''
        The first stage of backpropagation. From the write up, the gradient update for V
        mean squared error as the loss function (J) is:

        V <- V - epsilon * [   (1 - h_1^2)\sum_{i = 1}^10 (w_{i1}z_i(1 - z_i)(z_i - y_i)) x_j.T   ]
                           [   (1 - h_2^2)\sum_{i = 1}^10 (w_{i2}z_i(1 - z_i)(z_i - y_i)) x_j.T   ]
                           [                                  ...                                 ]
                           [                                  ...                                 ]
                           [                                  ...                                 ]
                           [(1 - h_{200}^2)\sum_{i = 1}^10 (w_{i200}z_i(1 - z_i)(z_i - y_i)) x_j.T]

        Note that this update can be writen in the form of an outer product (for the purposes of
        performance enhancement):

        V <- V - epsilon * (
            np.outer(
                np.subtract(np.ones(self.hidden_layer_size), np.square(h)),
                np.dot(
                    W.T,
                    np.multiply(
                        np.multiply(z, np.subtract(np.ones(self.output_layer_size), z)),
                        np.subtract(z, y)
                    )
                ),
                x
            )
        )

        Using cross-entropy error as the the loss function (J), the gradient update for V is:

        V <- V - epsilon * [   (1 - h_1^2)\sum_{i = 1}^10 (w_{i1}z_i(1 - z_i)((1 - y_i)/(1 - z_i) - y_i/z_i)) x_j.T   ]
                           [   (1 - h_2^2)\sum_{i = 1}^10 (w_{i2}z_i(1 - z_i)((1 - y_i)/(1 - z_i) - y_i/z_i)) x_j.T   ]
                           [                                            ...                                           ]
                           [                                            ...                                           ]
                           [                                            ...                                           ]
                           [(1 - h_{200}^2)\sum_{i = 1}^10 (w_{i200}z_i(1 - z_i)((1 - y_i)/(1 - z_i) - y_i/z_i)) x_j.T]

        This is written as an outer product in the form:

        V <- V - epsilon * (
            np.outer(
                np.subtract(np.ones(self.hidden_layer_size), np.square(h)),
                np.dot(
                    W.T, np.multiply(
                        np.multiply(z, np.subtract(np.ones(self.output_layer_size), z)),
                        np.subtract(
                            np.divide(
                                np.subtract(np.ones(self.output_layer_size), self.labels),
                                np.subtract(np.ones(self.output_layer_size), z)
                            ),
                            np.divide(self.labels, z)
                        )
                    )
                ),
                x
            )
        )

        The gradient update for W using mean squared error as the loss function is:

        W <- W - epsilon * [  (z_1 - y_1)z_1(1 - z_1) h.T  ]
                           [  (z_2 - y_2)z_2(1 - z_2) h.T  ]
                           [              ...              ]
                           [              ...              ]
                           [              ...              ]
                           [(z_10 - y_10)z_10(1 - z_10) h.T]

        Note that this update can be writen in the form of an outer product:

        W <- W - epsilon * (
            np.outer(
                np.multiply(
                    np.multiply(
                        np.subtract(z - y), z
                    ),
                    np.subtract(
                        np.ones(self.output_layer_size), z)
                    )
                ),
                h
            )
        )

        The gradient update for W using cross-entropy error as the loss function is:

        W <- W - epsilon * [   ((1 - y_1)/(1 - z_1) - y_1/z_1)z_1(1 - z_1) h.T   ]
                           [   ((1 - y_2)/(1 - z_2) - y_1/z_1)z_2(1 - z_2) h.T   ]
                           [                          ...                        ]
                           [                          ...                        ]
                           [                          ...                        ]
                           [((1 - y_10)/(1 - z_10) - y_10/z_10)z_10(1 - z_10) h.T]

        Which can be written in the form of an outer product as:

        W <- W - epsilon * (
            np.outer(
                np.multiply(
                    np.subtract(
                        np.divide(
                            np.subtract(np.ones(self.output_layer_size), self.labels),
                            np.subtract(np.ones(self.output_layer_size), z)
                        ),
                        np.divide(self.labels, z)
                    )
                    np.multiply(
                        np.multiply(
                            z,
                            np.subtract(np.ones(self.output_layer_size), z)
                        )
                    )
                ),
                h
            )
        )

        (np.multiply is an element-wise multiplication algorithm for vectors.)
        (np.divide is an element-wise division algorithm for vectors.)
        '''
        h, z = forward_pass_list

        y = np.zeros(z.shape[0])
        y[y_j] = 1

        y = y.reshape(z.shape[0], 1).astype("float64")
        x_j = np.append(x_j, 1).astype("float64")
        if self.loss_function == "mean-squared-error":
            gradient_v = np.outer(
                np.multiply(
                    np.subtract(
                        np.ones(self.hidden_layer_size + 1), np.square(h)
                    ).reshape(self.hidden_layer_size + 1, 1),
                    np.dot(
                        W.T,
                        np.multiply(
                            np.multiply(
                                z,
                                np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z)
                            ),
                            np.subtract(z, y)
                        )
                    )
                ),
                x_j
            )
            gradient_v = np.delete(gradient_v, -1, 0)
            gradient_w = np.outer(
                np.multiply(
                    np.multiply(
                        np.subtract(z, y), z
                    ),
                    np.subtract(
                        np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z
                    )
                ),
                h
            )
        else:
            gradient_v = np.outer(
                np.multiply(
                    np.subtract(
                        np.ones(self.hidden_layer_size + 1), np.square(h)
                    ).reshape(self.hidden_layer_size + 1, 1),
                    np.dot(
                        W.T,
                        np.multiply(
                            np.multiply(z, np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z)),
                            np.subtract(
                                np.divide(
                                    np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), y),
                                    np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z)
                                ),
                                np.divide(y, z)
                            )
                        )
                    )
                ),
                x_j
            )
            gradient_v = np.delete(gradient_v, -1, 0)
            gradient_w = np.outer(
                np.multiply(
                    np.subtract(
                        np.divide(
                            np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), y),
                            np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z)
                        ),
                        np.divide(y, z)
                    ),
                    np.multiply(
                        z,
                        np.subtract(np.ones(self.output_layer_size).reshape(self.output_layer_size, 1), z)
                    )
                ),
                h
            )

        return gradient_v, gradient_w

    # TODO: Modularize this function. Especially the section on gradient checking.
    def converged_gradient(self, num_iter, X, V, W, iter_check=50000, threshold=0.005,
                           gradient_v=None, gradient_w=None, error=True, gradient_check=False,
                           epsilon=10.**-5, x_j=None, y_j=None):
        training_error = None
        training_loss = None

        if num_iter > 1000000:
            return (True, training_error, training_loss)
        # There are two ways to determine if the gradient has converged.
        # (1) Use the training error (error=True)
        # (2) Use the magnitude of the gradient (error=False)
        # In both cases, training_error and training_loss are attached to the response
        # for the purposes of plotting.
        if error:
            if num_iter % iter_check != 0:
                return (False, training_error, training_loss)
            else:
                if gradient_check:
                    # Randomly check five weights.
                    for _ in range(5):
                        # import pdb; pdb.set_trace()
                        random_wi = np.random.randint(W.shape[0])
                        random_wj = np.random.randint(W.shape[1])
                        random_vi = np.random.randint(V.shape[0])
                        random_vj = np.random.randint(V.shape[1])

                        W_plus_epsilon = W.copy()
                        W_plus_epsilon[random_wi][random_wj] = W_plus_epsilon[random_wi][random_wj] + epsilon
                        Z_W_plus = self.perform_forward_pass(x_j, V, W_plus_epsilon)[1]

                        W_minus_epsilon = W.copy()
                        W_minus_epsilon[random_wi][random_wj] = W_minus_epsilon[random_wi][random_wj] - epsilon
                        Z_W_minus = self.perform_forward_pass(x_j, V, W_minus_epsilon)[1]

                        V_plus_epsilon = V.copy()
                        V_plus_epsilon[random_vi][random_vj] = V_plus_epsilon[random_vi][random_vj] + epsilon
                        Z_V_plus = self.perform_forward_pass(x_j, V_plus_epsilon, W)[1]

                        V_minus_epsilon = V.copy()
                        V_minus_epsilon[random_vi][random_vj] = V_minus_epsilon[random_vi][random_vj] - epsilon
                        Z_V_minus = self.perform_forward_pass(x_j, V_minus_epsilon, W)[1]

                        y = np.zeros(10)
                        y[y_j] = 1

                        if self.loss_function == "mean-squared-error":
                            W_plus_cost = mean_squared_error(Z_W_plus, y)
                            W_minus_cost = mean_squared_error(Z_W_minus, y)
                            V_plus_cost = mean_squared_error(Z_V_plus, y)
                            V_minus_cost = mean_squared_error(Z_V_minus, y)
                        else:
                            W_plus_cost = cross_entropy_loss(Z_W_plus.T, y)
                            W_minus_cost = cross_entropy_loss(Z_W_minus.T, y)
                            V_plus_cost = cross_entropy_loss(Z_V_plus.T, y)
                            V_minus_cost = cross_entropy_loss(Z_V_minus.T, y)

                        gradient_approx_wij = (W_plus_cost - W_minus_cost) / (2. * epsilon)
                        gradient_approx_vij = (V_plus_cost - V_minus_cost) / (2. * epsilon)

                        if gradient_approx_wij > gradient_w[random_wi][random_wj] + threshold or \
                           gradient_approx_wij < gradient_w[random_wi][random_wj] - threshold or \
                           gradient_approx_vij > gradient_v[random_vi][random_vj] + threshold or \
                           gradient_approx_vij < gradient_v[random_vi][random_vj] - threshold:
                            raise AssertionError("The gradient was incorrectly computed.")

                classifications_training, training_Z = self.predict(X, V, W, return_Z=True)
                training_error, training_indices_error = benchmark(classifications_training, self.labels)

                if self.validation_data is not None and self.validation_labels is not None:
                    classifications_validation = self.predict(self.validation_data, V, W)
                    validation_error, validation_indices_error = benchmark(classifications_validation, self.validation_labels)

                if self.loss_function == "mean-squared-error":
                    training_loss = mean_squared_error(training_Z.T, self.Y)
                else:
                    training_loss = cross_entropy_loss(training_Z.T, self.Y)

                print("Completed %d iterations.\nThe training error is %.2f.\n The training loss is %.2f."
                      % (num_iter, training_error, training_loss))

                if self.validation_data is not None and self.validation_labels is not None:
                    print("The error on the validation set is %.2f." % validation_error)

                if training_error < threshold:
                    return (True, training_error, training_loss)

                return (False, training_error, training_loss)
        else:
            if num_iter % iter_check == 0:
                classifications_training, training_Z = self.predict(X, V, W, return_Z=True)
                training_error, indices_error = benchmark(classifications_training, self.labels)

                if self.validation_data is not None and self.validation_labels is not None:
                    classifications_validation = self.predict(self.validation_data, V, W)
                    validation_error, validation_indices_error = benchmark(classifications_validation, self.validation_labels)

                if self.loss_function == "mean-squared-error":
                    training_loss = mean_squared_error(training_Z.T, self.Y)
                else:
                    training_loss = cross_entropy_loss(training_Z.T, self.Y)

                print("Completed %d iterations. The training error is %.2f. Training loss is %.2f" % (num_iter, training_error))

                if self.validation_data is not None and self.validation_labels is not None:
                    print("The error on the validation set is %.2f." % validation_error)

            if np.linalg.norm(gradient_v) < threshold and np.linalg.norm(gradient_w) < threshold:
                return (True, training_error, training_loss)
            else:
                return (False, training_error, training_loss)

    def learning_rate_this_iteration(self, num_iter):
        '''
        Adjust this function as necessary to decrement the learning rate over time.
        This only changes self.learning_rate if self.decreasing_rate == True.
        '''
        if self.decreasing_rate:
            return (0.5 ** (num_iter / 100000)) * self.learning_rate
        else:
            return self.learning_rate
