# ================================================
# Imageset of Mobile Traffic Data 2017 (IMTD17)
# with a MNIST-like interface using TensorFlow.
#
# Copyright 2017 by Ding Li. All Rights Reserved.
# ================================================

import os
import gzip
import numpy
import struct
import collections
from tensorflow.python.framework import dtypes

class Utility(object):
    
    def __init__(self, train_dir):
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        self.train_dir = train_dir
    
    def _read32(self, stream):
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(stream.read(4), dtype=dt)[0]
    
    def _read_check_file(self, file_name):
        file_path = self.train_dir + '/' + file_name
        if not os.path.exists(file_path):
            raise FileExistsError('File not exist')
        return file_path
    
    def extract_images(self, file_name):
        file_path = self._read_check_file(file_name)
        
        with open(file_path, 'rb') as gzip_file:
            print('Extracting', gzip_file.name)
            with gzip.GzipFile(fileobj=gzip_file) as stream:
                magic = self._read32(stream)
                if magic != 2051:
                    raise ValueError('Invalid magic number')
                num_images = self._read32(stream)
                rows = self._read32(stream)
                cols = self._read32(stream)
                buf = stream.read(rows * cols * num_images)
                data = numpy.frombuffer(buf, dtype=numpy.uint8)
                data = data.reshape(num_images, rows, cols, 1)
                return data
    
    def _dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    
    def extract_labels(self, file_name, one_hot=True, num_classes=12):
        file_path = self._read_check_file(file_name)
        
        with open(file_path, 'rb') as gzip_file:
            print('Extracting', gzip_file.name)
            with gzip.GzipFile(fileobj=gzip_file) as stream:
                magic = self._read32(stream)
                if magic != 2049:
                    raise ValueError('Invalid magic number')
                num_items = self._read32(stream)
                buf = stream.read(num_items)
                labels = numpy.frombuffer(buf, dtype=numpy.uint8)
                if one_hot:
                    return self._dense_to_one_hot(labels, num_classes)
                return labels

class DataSet(object):

    def __init__(self, images, labels, dtype=dtypes.float32, reshape=True):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype, expected uint8 or float32')
        if numpy.any(labels):
            assert images.shape[0] == labels.shape[0], ('Number of images and labels not match')
        self._num_examples = images.shape[0]
        self._shapes = images.shape[1:3]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def shapes(self):
        return self._shapes    

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True, rewind=False):
        """Return the next `batch_size` examples from this data set."""
        if rewind:
            self._index_in_epoch = 0
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            if numpy.any(self._labels):
                self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            if numpy.any(self._labels):
                labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                if numpy.any(self._labels):
                    self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            if numpy.any(self._labels):
                labels_new_part = self._labels[start:end]
            if numpy.any(self._labels):
                return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), None
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            if numpy.any(self._labels):
                return self._images[start:end], self._labels[start:end]          
            return self._images[start:end], None

def read_data_sets(train_dir,
                   one_hot=True,
                   num_classes=12,
                   dtype=dtypes.float32,
                   reshape=True):
    S1TRAIN_IMAGES = 's1train-images-idx3-ubyte.gz'
    S2TRAIN_IMAGES = 's2train-images-idx3-ubyte.gz'
    S2TRAIN_LABELS = 's2train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 'test-images-idx3-ubyte.gz'
    TEST_LABELS = 'test-labels-idx1-ubyte.gz'
    
    util = Utility(train_dir)
    
    s1train_images = util.extract_images(S1TRAIN_IMAGES)
    s2train_images = util.extract_images(S2TRAIN_IMAGES)
    assert s1train_images.shape[1:3] == s2train_images.shape[1:3]
    s2train_labels = util.extract_labels(S2TRAIN_LABELS, one_hot=one_hot, num_classes=num_classes)
    test_images = util.extract_images(TEST_IMAGES)
    assert s1train_images.shape[1:3] == test_images.shape[1:3]
    test_labels = util.extract_labels(TEST_LABELS, one_hot=one_hot, num_classes=num_classes+1)
    if one_hot:
        test_labels = test_labels[:,:num_classes]
    
    if dtype == None:
        dtype = dtypes.uint8
    s1train = DataSet(s1train_images, None, dtype=dtype, reshape=reshape)
    s2train = DataSet(s2train_images, s2train_labels, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    
    Datasets = collections.namedtuple('Datasets', ['s1train', 's2train', 'test'])
    
    return Datasets(s1train=s1train, s2train=s2train, test=test)
