"""
Testing the test procedure itself.
"""

import logging
import os
import unittest

import numpy

from safire.datasets.sharded_dataset import ShardedDataset
from safire.data.loaders import MultimodalDatasetLoader, ShardedDatasetLoader
from safire.learning.learners.base_sgd_learner import BaseSGDLearner


class TestShardedDataset(unittest.TestCase):

    def setUp(self):

        self.testdir = os.path.dirname(__file__)
        self.data_root = os.path.join(self.testdir, 'test-data')

        self.loader = MultimodalDatasetLoader(self.data_root, 'test-data')
        self.dloader = ShardedDatasetLoader(self.data_root, 'test-data')

        self.learner = BaseSGDLearner(3, 2, validation_frequency=4)

        self.savename = 'test-data.mhandle'

    def tearDown(self):

        # Cleanup...
        test_dir = os.path.join(self.data_root, 'datasets')
        files = os.listdir(test_dir)
        for f in files:
            os.remove(os.path.join(test_dir, f))

    def test_init(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2)

        # Test that the shards were actually created
        self.assertTrue(os.path.isfile(output_prefix + '.1'))

    def test_load(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2)

        # Test that the shards were actually created
        self.assertTrue(os.path.isfile(output_prefix + '.1'))

        dataset.save()
        loaded_dataset = ShardedDataset.load(output_prefix)

        self.assertEqual(loaded_dataset.dim, dataset.dim)
        self.assertEqual(loaded_dataset.n_shards, dataset.n_shards)

    def test_getitem(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2)

        item = dataset[3]

        print item

        self.assertEqual(dataset.current_shard_n, 1)

        item = dataset[1:7]

        print item

        self.assertEqual((6, dataset.n_in), item.shape)

        for i in xrange(1,7):
            self.assertTrue(numpy.array_equal(dataset[i], item[i-1]))

    def test_resize(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2)

        self.assertEqual(10, dataset.n_shards)

        dataset.resize_shards(4)

        self.assertEqual(5, dataset.n_shards)
        for n in xrange(dataset.n_shards):
            fname = dataset._shard_name(n)
            self.assertTrue(os.path.isfile(fname))

        dataset.resize_shards(3)

        #print dataset.n_shards






if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()