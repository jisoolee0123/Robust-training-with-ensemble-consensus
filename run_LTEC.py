import os
import argparse
import numpy as np
from model import convnet
from datasets import get_data
from functools import reduce

def ce_loss(y_pred, y_true, eps=1e-12):
    predictions = np.clip(y_pred, eps, 1. - eps)
    ce = -np.sum(y_true * np.log(predictions + eps), axis=1)
    return ce


batch_size = 128

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)

# data information
parser.add_argument('--dataset', default='mnist', type=str, help='dataset to use')
parser.add_argument('--noise_ratio', default=20, type=int, help='ratio should be in [0-100]')
parser.add_argument('--noise_type', default='sym', type=str, help="sym or asym")

# optimizer
parser.add_argument('--epoch_decay_start', default=80, type=int, help='the number of epoch')
parser.add_argument('--n_epoch', default=200, type=int, help='the number of epoch')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate of optimizer')
parser.add_argument('--seed', default=45, type=int)

# LTEC hyperparameters
parser.add_argument('-m', '--num_of_prediction', default=5, type=int)
parser.add_argument('-w', '--warming_up_period', default=10, type=int)


config = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % config.gpu

# load dataset
train_x, train_y, test_x, test_y, clean_y = get_data(dataset=config.dataset,
                                                        noise_ratio=config.noise_ratio,
                                                        noise_type=config.noise_type,
                                                        random_shuffle=True,
                                                        seed=config.seed)

np.random.seed(config.seed)

n_samples = train_x.shape[0]
x_shape = train_x.shape[1:]
y_shape = train_y.shape[1]
n_batches = n_samples // batch_size + 1

# define model
model = convnet(config, x_shape=x_shape, y_shape=y_shape, mom2=0.1)

forget_rate = config.noise_ratio / 100.
remember_rate = 1. - forget_rate

small_loss_earlier_epochs = []

for epoch in range(config.n_epoch):
    small_loss_current_epoch = []

    # random permutation
    perm = np.random.permutation(n_samples)

    # optimizer update
    model.adjust_learning_rate(model.f.optimizer, epoch)

    # FIFO
    if len(small_loss_earlier_epochs) == config.num_of_prediction:
        small_loss_earlier_epochs.pop(0)

    # compute the intersection of small-loss examples at the last M-1 epochs
    if len(small_loss_earlier_epochs) > 0:
        intersection_of_earlier = reduce(np.intersect1d, small_loss_earlier_epochs)

    # batch update
    for _ in range(n_batches):
        start, end = _ * batch_size, min((_+1) * batch_size, n_samples)
        target_idx = perm[start:end]
        num_remember = int(len(target_idx) * remember_rate)

        x_batch, y_batch = train_x[target_idx], train_y[target_idx]

        # compute small-loss examples within a single batch
        y_pred = model.pred_f(inputs=[x_batch, 1])[0]
        loss = ce_loss(y_pred=y_pred, y_true=y_batch)
        ind_sorted = np.argsort(loss)
        ind_sorted = ind_sorted[:num_remember]

        # save small-loss examples
        small_loss_current_epoch.extend(target_idx[ind_sorted])

        # warming-up
        if epoch < config.warming_up_period:
            xi_batch, yi_batch = x_batch, y_batch

        # ensemble consensus
        else:
            inter_i = np.intersect1d(target_idx[ind_sorted], intersection_of_earlier)
            xi_batch, yi_batch = train_x[inter_i], train_y[inter_i]

        # update
        model.f.train_on_batch(xi_batch, yi_batch)

    # at the end of epoch
    small_loss_earlier_epochs.append(small_loss_current_epoch)

    # evaluation
    pred = model.f.predict(test_x, batch_size=100)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(test_y, axis=1))

    print('[EPOCH %03d] test accuracy: %.3f' % (epoch, acc))

