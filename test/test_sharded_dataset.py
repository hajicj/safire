"""
Testing the test procedure itself.
"""

import logging
import os
import unittest

import numpy
from scipy import sparse

from gensim.utils import is_corpus

from safire.datasets.sharded_dataset import ShardedDataset
from safire.data.loaders import MultimodalShardedDatasetLoader, ShardedDatasetLoader
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from safire_test_case import SafireTestCase


class TestShardedDataset(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestShardedDataset, cls).setUpClass(no_datasets=True)

    def setUp(self):

        self.loader = MultimodalShardedDatasetLoader(self.data_root,
                                                     'test-data')
        self.dloader = ShardedDatasetLoader(self.data_root,
                                            'test-data')

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

        #print item

        self.assertEqual(dataset.current_shard_n, 1)

        item = dataset[1:7]

        #print item

        self.assertEqual((6, dataset.n_in), item.shape)

        for i in xrange(1, 7):
            self.assertTrue(numpy.array_equal(dataset[i], item[i-1]))

    def test_sparse_serialization(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        no_exception = True
        try:
            dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                     sparse_serialization=True)
        except Exception:
            no_exception = False
            raise
        self.assertTrue(no_exception)

    def test_getitem_dense2dense(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                 sparse_serialization=False,
                                 sparse_retrieval=False)

        item = dataset[3]
        self.assertIsInstance(item, numpy.ndarray)
        self.assertEqual(item.shape, (dataset.n_out,))

        dslice = dataset[2:6]
        self.assertIsInstance(dslice, numpy.ndarray)
        self.assertEqual(dslice.shape, (4, dataset.n_out))

        ilist = dataset[[2, 3, 4, 5]]
        self.assertIsInstance(ilist, numpy.ndarray)
        self.assertEqual(ilist.shape, (4, dataset.n_out))

        self.assertEqual(ilist.all(), dslice.all())

    def test_getitem_dense2sparse(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                 sparse_serialization=False,
                                 sparse_retrieval=True)

        item = dataset[3]
        self.assertIsInstance(item, sparse.csr_matrix)
        self.assertEqual(item.shape, (1, dataset.n_out))

        dslice = dataset[2:6]
        self.assertIsInstance(dslice, sparse.csr_matrix)
        self.assertEqual(dslice.shape, (4, dataset.n_out))

        ilist = dataset[[2, 3, 4, 5]]
        self.assertIsInstance(ilist, sparse.csr_matrix)
        self.assertEqual(ilist.shape, (4, dataset.n_out))

        self.assertEqual((ilist != dslice).getnnz(), 0)

    def test_getitem_sparse2sparse(self):

        icorp = self.loader.load_image_corpus()
        sp_output_prefix = self.dloader.output_prefix(sparse_serialization=True)
        dataset = ShardedDataset(sp_output_prefix, icorp, shardsize=2,
                                 sparse_serialization=True,
                                 sparse_retrieval=True)

        output_prefix = self.dloader.output_prefix()
        dense_dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                       sparse_serialization=False,
                                       sparse_retrieval=True)

        item = dataset[3]
        self.assertIsInstance(item, sparse.csr_matrix)
        self.assertEqual(item.shape, (1, dataset.n_out))

        dslice = dataset[2:6]
        self.assertIsInstance(dslice, sparse.csr_matrix)
        self.assertEqual(dslice.shape, (4, dataset.n_out))

        expected_nnz = 5810
        self.assertEqual(dslice.getnnz(), expected_nnz)

        ilist = dataset[[2, 3, 4, 5]]
        self.assertIsInstance(ilist, sparse.csr_matrix)
        self.assertEqual(ilist.shape, (4, dataset.n_out))

        # Also compare with what the dense dataset is giving us
        d_dslice = dense_dataset[2:6]
        self.assertEqual((d_dslice != dslice).getnnz(), 0)
        self.assertEqual((ilist != dslice).getnnz(), 0)

    def test_getitem_sparse2dense(self):

        icorp = self.loader.load_image_corpus()
        sp_output_prefix = self.dloader.output_prefix(sparse_serialization=True)
        dataset = ShardedDataset(sp_output_prefix, icorp, shardsize=2,
                                 sparse_serialization=True,
                                 sparse_retrieval=False)

        output_prefix = self.dloader.output_prefix()
        dense_dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                       sparse_serialization=False,
                                       sparse_retrieval=False)

        item = dataset[3]
        self.assertIsInstance(item, numpy.ndarray)
        self.assertEqual(item.shape, (1, dataset.n_out))

        dslice = dataset[2:6]
        self.assertIsInstance(dslice, numpy.ndarray)
        self.assertEqual(dslice.shape, (4, dataset.n_out))

        ilist = dataset[[2, 3, 4, 5]]
        self.assertIsInstance(ilist, numpy.ndarray)
        self.assertEqual(ilist.shape, (4, dataset.n_out))

        # Also compare with what the dense dataset is giving us
        d_dslice = dense_dataset[2:6]
        self.assertEqual(dslice.all(), d_dslice.all())
        self.assertEqual(ilist.all(), dslice.all())

    def test_getitem_dense2gensim(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()
        dataset = ShardedDataset(output_prefix, icorp, shardsize=2,
                                 sparse_serialization=False,
                                 gensim=True)

        item = dataset[3]
        self.assertIsInstance(item, list)
        self.assertIsInstance(item[0], tuple)

        dslice = dataset[2:6]
        self.assertIsInstance(dslice, list)
        self.assertIsInstance(dslice[0], list)
        self.assertIsInstance(dslice[0][0], tuple)

        iscorp, _ = is_corpus(dslice)
        self.assertTrue(iscorp, "Is the object returned by slice notation "
                                "a gensim corpus?")

        ilist = dataset[[2, 3, 4, 5]]
        self.assertIsInstance(ilist, list)
        self.assertIsInstance(ilist[0], list)
        self.assertIsInstance(ilist[0][0], tuple)

        self.assertEqual(len(ilist), len(dslice))
        for i in xrange(len(ilist)):
            self.assertEqual(len(ilist[i]), len(dslice[i]),
                             "Row %d: dims %d/%d" % (i, len(ilist[i]),
                                                     len(dslice[i])))
            for j in xrange(len(ilist[i])):
                self.assertEqual(ilist[i][j], dslice[i][j],
                                 "ilist[%d][%d] = %s ,dslice[%d][%d] = %s" % (
                                     i, j, str(ilist[i][j]), i, j,
                                     str(dslice[i][j])))


        iscorp, _ = is_corpus(ilist)
        self.assertTrue(iscorp, "Is the object returned by list notation "
                                "a gensim corpus?")

    def test_resize(self):

        icorp = self.loader.load_image_corpus()
        output_prefix = self.dloader.output_prefix()

        #print icorp.input
        #print self.dloader.output_prefix()

        dataset = ShardedDataset(output_prefix, icorp, shardsize=2)

        self.assertEqual(10, dataset.n_shards)

        dataset.resize_shards(4)

        self.assertEqual(5, dataset.n_shards)
        for n in xrange(dataset.n_shards):
            fname = dataset._shard_name(n)
            self.assertTrue(os.path.isfile(fname))

        dataset.resize_shards(3)

        #print dataset.n_shards

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestShardedDataset)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
