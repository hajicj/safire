import os
import sys
import unittest
import logging
import numpy
from safire.utils import profile_run

import theano.tensor as TT

from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.utils.transcorp import get_id2word_obj
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.word2vec_transformer import Word2VecTransformer
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr.'


def _init_word2vec_dtransformer(*args, **kwargs):
    """Initializes a Word2VecDatasetTransformer with the given args.
    Used for profiling."""
    return Word2VecSamplingDatasetTransformer(*args, **kwargs)


class TestWord2VecDatasetTransformer(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestWord2VecDatasetTransformer, cls).setUpClass()

        cls.infix = ''

        if not os.getenv('HOME'):
            homepath = os.getenv('USERPROFILE')
        else:
            homepath = os.getenv('HOME')

        cls.w2v_data_root = cls.data_root
        cls.w2v_data_root = os.path.join(homepath, 'word2vec')
        cls.edict_pkl_file = os.path.join(cls.data_root,
                                          'test-data.edict.pkl')
        cls.e_matrix_file = os.path.join(cls.data_root,
                                         'test-data.emtr.pkl')

    def _setup_profileable(self):
        self.corpus = self.loader.load_text_corpus(self.infix)
        self.id2word = get_id2word_obj(self.corpus)
        self.w2v_transformer = Word2VecTransformer(self.edict_pkl_file,
                                                   self.id2word)
        if os.path.isfile(self.e_matrix_file):
            report, self.w2v = profile_run(
                _init_word2vec_dtransformer,
                self.w2v_transformer,
                embeddings_matrix=self.e_matrix_file)
            #print 'Profiling report - W2VDT init from matrix:'
            #print report.getvalue()
        else:
            report, self.w2v = profile_run(
                _init_word2vec_dtransformer,
                self.w2v_transformer,
                pickle_embeddings_matrix=self.e_matrix_file)
            #print 'Profiling report - W2VDT init from transformer:'
            #print report.getvalue()
            #
            # self.w2v = Word2VecSamplingDatasetTransformer(
            #     self.w2v_transformer,
            #     pickle_embeddings_matrix=self.e_matrix_file)
        self.dataset = self.loader.load_text(self.infix)

    def setUp(self):

        report, _ = profile_run(self._setup_profileable)
        #print 'Setup profiling report:'
        print self.dataset
        print self.dataset.data
        print self.dataset.data.offsets
        print len(self.dataset.vtcorp)
        print len(self.dataset.data)
        print self.dataset.indexed_corpus

    def test_get_batch_sample(self):

        batch = self.dataset.train_X_batch(0, 5)
        batch_sample = self.w2v.get_batch_sample(batch)

        # Which words were sampled?
        # for i, row in enumerate(batch_sample):
        #     wid = numpy.argmax(row)
        #     word = self.id2word[wid]
        #     print 'Doc %d : chose word %d --> %s' % (i, wid, word)

        batch_sample_2 = self.w2v.get_batch_sample(batch)
        # for i, row in enumerate(batch_sample_2):
        #     wid = numpy.argmax(row)
        #     word = self.id2word[wid]
        #     print 'Doc %d : chose word %d --> %s' % (i, wid, word)

        self.assertEqual(numpy.sum(batch_sample), batch.shape[0])

    def test_getitem(self):

        batch = self.dataset.train_X_batch(0, 2)
        transformed_batch = self.w2v[batch]

        self.assertEqual(transformed_batch.shape, (2, self.w2v.n_out))

    def test_apply(self):

        batch_size = 5
        transformed_dataset = self.w2v[self.dataset]
        transformed_batch = transformed_dataset.train_X_batch(0, batch_size)

        self.assertEqual(transformed_batch.shape, (batch_size, self.w2v.n_out))

###############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestWord2VecDatasetTransformer)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)