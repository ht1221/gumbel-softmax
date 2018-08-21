# coding: utf-8
import os
import struct
import numpy as np
import logging

log = logging.getLogger("root")

def load_mnist(images_path, labels_path):
    """Load MNIST data from `path`"""
    # labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    # images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def next_batch(data, batch_size):
    images, labels = data
    nb_samples = len(images)
    if batch_size > nb_samples:
        log.fatal("batch_size is larger than data scale!")

    idx = list(range(nb_samples))
    np.random.shuffle(idx)

    batch_images = images[idx[:batch_size], :]
    batch_labels = labels[idx[:batch_size]]

    return batch_images, batch_labels



if __name__ == "__main__":
    train_images, train_labels = load_mnist("")
    print train_images.shape, train_labels.shape