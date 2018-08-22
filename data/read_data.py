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

    return normalized(images), labels


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

def normalized(images):
    normalized_images = np.array(images, dtype=np.float32)
    max_value = normalized_images.max()
    # print max_value
    normalized_images = normalized_images / max_value
    return normalized_images


def print_image(image):
    # shape = image.shape
    for i in range(28):
        for j in range(28):
            print image[i*28+j],
        print ""
    return

if __name__ == "__main__":
    train_images, train_labels = load_mnist("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
    print train_images.shape, train_labels.shape
    print train_images[789]
    # bw_images = (1-np.equal(train_images, 0)).astype(np.int32)
    # print_image(bw_images[0])

    a = np.array([0, 1, 2, 3])
    aa = a.astype(np.float32)
    m = aa.max()
    print m
    print aa / m