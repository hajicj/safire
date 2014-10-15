#!/usr/bin/env python
"""
Testing a small training procedure to check that the loading and training
architecture is OK.
"""

import logging
import os
import unittest

from safire.data.loaders import MultimodalDatasetLoader
from safire.learning.interfaces.model_handle import ModelHandle
from safire.learning.models import RestrictedBoltzmannMachine
from safire.learning.models.denoising_autoencoder import DenoisingAutoencoder
from safire.learning.learners.base_sgd_learner import BaseSGDLearner


class TestUnsupervisedTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.testdir = os.path.dirname(__file__)
        cls.data_root = os.path.join(cls.testdir, 'test-data')

        cls.loader = MultimodalDatasetLoader(cls.data_root, 'test-data')

        dataset = cls.loader.load_text()

        # Let's compress the dataset to 10 dimensions, for testing speed.
        # (Testing will take long anyway, due to compilation.)
        cls.model_handle = RestrictedBoltzmannMachine.setup(dataset, n_out=10)

        cls.learner = BaseSGDLearner(3, 2, validation_frequency=4)

    @classmethod
    def tearDownClass(cls):

        del cls.learner
        del cls.model_handle
        del cls.loader

    def test_training(self):


        self.assertIsInstance(self.model_handle, ModelHandle)
        self.assertTrue(hasattr(self.model_handle, 'train'))

        param_before = self.model_handle.model_instance.W.get_value()[0,0]
        print "Weight [0,0] before training: %f" % param_before

        dataset = self.loader.load_text()
        self.learner.run(self.model_handle, dataset)

        param_after = self.model_handle.model_instance.W.get_value()[0,0]
        print "Weight [0,0] after training: %f" % param_after

        self.assertNotEqual(param_before, param_after)

        test_batch = dataset.test_X_batch(0, 1)
        output = self.model_handle.run(test_batch)

        print type(output)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()