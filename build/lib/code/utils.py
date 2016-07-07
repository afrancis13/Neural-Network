import numpy as np
import matplotlib.pyplot as plt


def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def cross_entropy_loss(pred_labels, true_labels):
    return -1.0 * np.sum(
        np.add(
            np.multiply(true_labels, safe_log(pred_labels)),
            np.multiply(
                np.subtract(np.ones(len(true_labels)).reshape(len(true_labels), 1), true_labels),
                safe_log(np.subtract(np.ones(len(pred_labels)).reshape(len(pred_labels), 1), pred_labels))
            )
        )
    )


def mean_squared_error(pred_labels, true_labels):
    return 1.0 / 2.0 * np.sum(np.square(np.subtract(true_labels, pred_labels)))


def compute_sigmoid(gamma):
    return 1. / (1. + np.e ** (-1. * np.clip(gamma, -709, 709)))


def plot_image(image, label="Test"):
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %s' % str(label))

    plt.show()


def benchmark(pred_labels, true_labels):
    """benchmark.m, converted. From Piazza, February 2016."""
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices


def shuffle_in_unison_inplace(a, b):
    """
    Included in HW4 Submission in March 2016.

    Shuffles any two sets in unison. Assumes that the length of the sets
    are equal, and asserts this (if this is not true, this method has no
    meaning).
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def cross_validate(k, X, y):
    """
    Adapted from previous homework submissions.

    Takes in a value k (k-fold cross validation), the parameters to cross
    validate on (in the case of ridge regression, lambda), and some
    predefined black box, which does all of the work for the particular problem,
    and takes in a parameter.

    Parameter decreasing is an added kwarg for this function. If set to true,
    the use tells the parameter to decrease with the number of iterations during the
    fit_procedure call.
    """
    partition_length = 1.0 * y.shape[0] / k
    X_shuffled, y_shuffled = shuffle_in_unison_inplace(X, y)

    cross_validation_sets = []
    for i in range(k):
        validation_X = X_shuffled[partition_length * i: partition_length * (i + 1)]
        validation_y = y_shuffled[partition_length * i: partition_length * (i + 1)]
        training_X = np.vstack((X_shuffled[:partition_length * i], X_shuffled[partition_length * (i + 1):]))
        training_y = np.vstack((y_shuffled[:partition_length * i], y_shuffled[partition_length * (i + 1):]))

        cross_validation_sets.append([training_X, training_y, validation_X, validation_y])

    return cross_validation_sets
