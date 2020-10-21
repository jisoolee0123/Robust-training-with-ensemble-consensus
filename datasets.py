import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils

NUM_CLASSES = {'mnist': 10}

def symmetric(n_class, current_class):
    # from https://github.com/xingjunm/dimensionality-driven-learning
    if current_class < 0 or current_class >= n_class:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_class))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)

    return other_class


def get_data(dataset='mnist', noise_ratio=0, noise_type='sym', random_shuffle=False, seed=None):

    np.random.seed(seed)

    noise_fraction = noise_ratio / 100.

    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_clean = np.copy(y_train)

    # generate random noisy labels
    if noise_ratio > 0:
        if noise_type == 'sym':
            # from https://github.com/xingjunm/dimensionality-driven-learning
            n_samples = y_train.shape[0]
            n_noisy = int(n_samples * noise_fraction)
            noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)

            for i in noisy_idx:
                y_train[i] = symmetric(n_class=NUM_CLASSES[dataset], current_class=y_train[i])

        elif noise_type == 'asym':
            # from https://github.com/udibr/noisy_labels
            perm = np.arange(NUM_CLASSES[dataset])
            perm = np.roll(perm, -1)
            noise = perm[y_train]
            _, noisy_idx = next(iter(StratifiedShuffleSplit(n_splits=1,
                                                            test_size=noise_ratio/100.,
                                                            random_state=seed).split(X_train, y_train)))
            y_train[noisy_idx] = noise[noisy_idx]

    if random_shuffle:
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train, y_clean = X_train[idx_perm], y_train[idx_perm], y_clean[idx_perm]

    # one-hot-encode the labels
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_clean = np_utils.to_categorical(y_clean, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("Actual noise: %.2f" % np.mean(np.argmax(y_clean, axis=1) != np.argmax(y_train, axis=1)))

    return X_train, y_train, X_test, y_test, y_clean







