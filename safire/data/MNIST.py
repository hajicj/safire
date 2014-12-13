#!/usr/bin/env python

import os
import gzip
import cPickle

from .loader import DataLoader
from .safire.datasets.supervised_dataset import SupervisedDataset


class MNIST(object):
    """A static class for keeping information about the MNIST dataset
    downloaded from deeplearning.net tutorials.
    
    It has a ``path`` and ``filename`` attribute that together
    describe the location of the ``mnist.pkl.gz`` file, which
    contains the MNIST dataset cPickled from the deeplearning.net
    tutorials.
    
    Assumes that the ``SAFIRE`` environmental variable is set.
    
    """
    
    path = os.environ.get('SAFIRE') + '/source/safire/data'
    filename = 'mnist.pkl.gz'

    @staticmethod
    def load():
        """Loads the MNIST dataset as a triplet of train, devel, test
        theano shared variables and wraps them into 
        a :class:`SupervisedDataset`.
        
        Expects the MNIST dataset to be the cPickled file from
        the deeplearning.net tutorials.
        
        :returns: A :class:`SupervisedDataset` initialized with the
                  MNIST dataset.
                  
        :raises: IOError
        """
        mnist_full_filename = os.path.join(MNIST.path, MNIST.filename)
        with gzip.open(mnist_full_filename, 'rb') as mnist_file:
            train_set, devel_set, test_set = cPickle.load(mnist_file)
            return SupervisedDataset(
                    (DataLoader.as_shared(train_set), 
                    DataLoader.as_shared(devel_set), 
                    DataLoader.as_shared(test_set)),
                    n_in = 28*28,
                    n_out = 10)
