"""
Testing the test procedure itself.
"""

import logging
import os
import unittest

import numpy
from scipy import sparse

from gensim.utils import is_corpus
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.sharded_dataset import ShardedDataset

from safire.data.loaders import MultimodalShardedDatasetLoader, ShardedDatasetLoader
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from safire.utils import mock_data
from safire_test_case import SafireTestCase


# This class should test stuff like train_X_batch, etc.
class TestShardedDataset(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        # Create mock data instead of using real data
        # (easier to check, better isolation)
        cls.data_dim = 100
        cls.data = mock_data(n_items=1000, dim=cls.data_dim)

        cls.devel_p = 0.1
        cls.test_p = 0.1

        super(TestShardedDataset, cls).setUpClass(no_datasets=True)

        # Files will get removed through SafireTestCase.tearDownClass()
        cls.shcorp_name = os.path.join(cls.data_root, 'corpora', 'tshcorp')
        ShardedCorpus.serialize(cls.shcorp_name,
                                cls.data, dim=cls.data_dim)

        cls.shdat_name = os.path.join(cls.data_root, 'datasets', 'tshdat')

    def setUp(self):
        # Can do this without Loaders, as we are
        # using "live" mock data that was later serialized.
        self.corpus = ShardedCorpus.load(self.shcorp_name)
        self.dataset = ShardedDataset(corpus=self.corpus,
                                      devel_p=self.devel_p,
                                      test_p=self.test_p)

    # def test_init(self):
    # No need to test initialization, as setUp would fail right away.

    def test_saveload(self):
        self.dataset.save(fname=self.shdat_name)
        loaded_dataset = ShardedDataset.load(self.shdat_name)

        self.assertEqual(self.dataset.train_X_batch(0, 100),
                         loaded_dataset.train_X_batch(0, 100))

    def test_n_train_batches(self):
        self.assertEqual(self.dataset.n_train_batches(100), 8)
        self.assertEqual(self.dataset.n_train_batches(101), 7)
        self.assertEqual(self.dataset.n_train_batches(99), 8)
        self.assertEqual(self.dataset.n_train_batches(999), 0)

    def test_n_devel_batches(self):
        self.assertEqual(self.dataset.n_devel_batches(100), 1)
        self.assertEqual(self.dataset.n_devel_batches(101), 0)
        self.assertEqual(self.dataset.n_devel_batches(99), 1)

    def test_n_test_batches(self):
        self.assertEqual(self.dataset.n_test_batches(100), 1)
        self.assertEqual(self.dataset.n_test_batches(101), 0)
        self.assertEqual(self.dataset.n_test_batches(99), 1)

    def test_train_X_batch(self):
        pass

    def test_getitem(self):
        pass


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestShardedDataset)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
