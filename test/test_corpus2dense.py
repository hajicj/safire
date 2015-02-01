import numpy
from safire.utils import mock_data, IndexedTransformedCorpus
from safire.utils.transformers import Corpus2Dense
from test.safire_test_case import SafireTestCase

__author__ = 'Lenovo'

import unittest


class TestCorpus2Dense(SafireTestCase):

    def setUp(self):
        self.dim = 100
        self.data = mock_data(dim=self.dim, prob_nnz=0.1)

    def test_corpus2dense(self):

        corpus2dense = Corpus2Dense(dim=self.dim)
        dense_corpus = corpus2dense._apply(self.data)
        self.assertIsInstance(dense_corpus, IndexedTransformedCorpus)

        doc = dense_corpus[0]

        self.assertIsInstance(doc, numpy.ndarray)

        batch = dense_corpus[0:3]

        self.assertIsInstance(batch, numpy.ndarray)

        print doc.shape

##############################################################################


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestCorpus2Dense)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

