"""
Testing the test procedure itself.
"""

import logging
import os
import unittest

import theano

from safire.data.loaders import MultimodalDatasetLoader, ModelLoader
from safire.learning.interfaces.model_handle import ModelHandle
from safire.learning.models.denoising_autoencoder import DenoisingAutoencoder
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from test import SafireTestCase

class TestModelLoader(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestModelLoader, cls).setUpClass()

        cls.mloader = ModelLoader(cls.data_root, 'test-data')

        dataset = cls.loader.load_text()

        # Let's compress the dataset to 10 dimensions, for testing speed.
        # (Testing will take long anyway, due to compilation.)
        cls.model_handle = DenoisingAutoencoder.setup(dataset, n_out=10)
        cls.model = cls.model_handle.model_instance

        cls.learner = BaseSGDLearner(3, 2, validation_frequency=4)

        cls.infix = '.da-10'

    @classmethod
    def tearDownClass(cls):

        os.remove(os.path.join(cls.mloader.root,
                               cls.mloader.layout.get_handle_file(cls.infix)))

        del cls.learner
        del cls.model_handle
        del cls.loader
        del cls.mloader

        super(TestModelLoader, cls).tearDownClass()

    def test_save_load(self):

        self.mloader.save_handle(self.model_handle, self.infix)

        loaded_handle = self.mloader.load_handle(self.infix)
        loaded_model = loaded_handle.model_instance

        self.assertIsInstance(loaded_handle, ModelHandle)
        self.assertIsInstance(loaded_model, DenoisingAutoencoder)
        self.assertEqual(self.model.n_in, loaded_model.n_in)
        self.assertEqual(loaded_model.activation, theano.tensor.nnet.sigmoid)

        before_loading = self.model.W.get_value()[0,0]
        after_loading = loaded_model.W.get_value()[0,0]

        self.assertEqual(before_loading, after_loading)

    def test_load_and_train(self):

        self.mloader.save_handle(self.model_handle, self.infix)

        loaded_handle = self.mloader.load_handle(self.infix)

        before_training = loaded_handle.model_instance.W.get_value()[0, 0]

        dataset = self.loader.load_text()
        self.learner.run(loaded_handle, dataset)

        after_training = loaded_handle.model_instance.W.get_value()[0, 0]

        self.assertNotEqual(before_training, after_training)

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestModelLoader)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
