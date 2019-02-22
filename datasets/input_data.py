from __future__ import division

import numpy as np
import time


def generate_batch(x, y, batch_size=100, shuffle=True):
    total_iteration = x.shape[0] // batch_size
    while True:
        if shuffle:
            idx_perm = np.random.permutation(x.shape[0])
        else:
            idx_perm = range(x.shape[0])

        for i, batch in enumerate(range(0, x.shape[0], batch_size)):
            if i == total_iteration:
                continue

            local_sorted_perm = list(map(int, idx_perm[batch:batch + batch_size]))
            print(len(local_sorted_perm), x.shape)
            x_out = x[local_sorted_perm]
            y_out = y[local_sorted_perm]
            yield x_out, y_out


def read_dataset(train, train_labels, val, val_labels, to_onehot=False, num_classes=6):
    print("Dataset loading...")

    stt = time.time()
    train_x = np.load(train)
    train_y = np.load(train_labels).astype(np.int)
    ett = time.time()

    stv = time.time()
    val_x = np.load(val)
    val_y = np.load(val_labels).astype(np.int)
    etv = time.time()

    if to_onehot:
        y = np.zeros([train_y.shape[0], num_classes])
        y[np.arange(train_y.shape[0]), train_y.astype(np.int64)] = 1
        train_y = y

        y = np.zeros([val_y.shape[0], num_classes])
        y[np.arange(val_y.shape[0]), val_y.astype(np.int64)] = 1
        val_y = y

    print("Dataset info:")
    print("\tTrain: {} - time: {:.3f}s".format(train_x.shape, ett - stt))
    print("\tVal: {} - time {:.3f}s".format(val_x.shape, etv - stv))
    return train_x, train_y, val_x, val_y


def normalize(x, axis=(2, 3), keepdims=True, name="dataset"):
    print("Normalizing {}".format(name))
    s = time.time()
    x_mean = np.mean(x, axis=axis, keepdims=keepdims)
    x_std = np.std(x, axis=axis, keepdims=keepdims)
    x_ms = (x - x_mean) / x_std
    x_ms = np.where(np.isfinite(x_ms), x_ms, np.zeros_like(x_std))
    print("\ttime: {:.3f}".format(time.time() - s))
    return x_ms
