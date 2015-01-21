import logging
import os
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.learning.interfaces import ModelHandle
from safire.learning.interfaces.clamped_sampler import MultimodalClampedSampler
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import RestrictedBoltzmannMachine

from test.safire_test_case import SafireTestCase

__author__ = 'Lenovo'

import unittest


class TestClampedSampler(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestClampedSampler, cls).setUpClass()

        cls.loader = MultimodalShardedDatasetLoader(cls.data_root, 'test-data')

        dataset = cls.loader.load()
        dataset.set_mode(0)

        cls.model_handle = RestrictedBoltzmannMachine.setup(dataset, n_out=10)

        cls.learner = BaseSGDLearner(3, 2, validation_frequency=4)

        logging.info('Initializing sampler...')

        cls.sampler = MultimodalClampedSampler(cls.model_handle.model_instance,
                                               dataset.dim_text, dataset.dim_img)

    @classmethod
    def tearDownClass(cls):

        del cls.learner
        del cls.model_handle
        del cls.loader

        super(TestClampedSampler, cls).tearDownClass()

    def test_training(self):

        self.assertIsInstance(self.model_handle, ModelHandle)
        self.assertTrue(hasattr(self.model_handle, 'train'))

        param_before = self.model_handle.model_instance.W.get_value()[0,0]
        print "Weight [0,0] before training: %f" % param_before

        dataset = self.loader.load()
        self.learner.run(self.model_handle, dataset)

        param_after = self.model_handle.model_instance.W.get_value()[0,0]
        print "Weight [0,0] after training: %f" % param_after

        self.assertNotEqual(param_before, param_after)

        test_batch = dataset.test_X_batch(0, 1)
        output = self.model_handle.run(test_batch)

        print type(output)

    def test_clamped_sampling(self):

        dataset = self.loader.load()
        dataset.set_mode(1) # Text docs

        text = dataset.train_X_batch(0,1)

        img = self.sampler.t2i_step(text)
        print img

        self.assertEqual(len(img), len(text))

        img2 = self.sampler.t2i_run_chain(text, k=10)
        print img

        img_s = self.sampler.t2i_step(text,
                                      sample_hidden=True,
                                      sample_visible=True)
        print img_s

        img_lm = self.sampler.t2i_run_chain_mean_last(text, k=10)
        print img_lm

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestClampedSampler)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
