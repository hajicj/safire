from gensim.interfaces import TransformedCorpus
from safire.data.loaders import IndexLoader
from safire.utils.transformers import SimilarityTransformer

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase


class TestSimilarityTransformer(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):

        super(TestSimilarityTransformer, cls).setUpClass(clean_only,
                                                         no_datasets)

        cls.iloader = IndexLoader(cls.data_root, 'test-data')

    def setUp(self):

        self.corpus = self.loader.get_default_image_corpus()

    def test_apply(self):

        self.transformer = SimilarityTransformer(self.corpus,
                                                 self.iloader.output_prefix())
        self.index_corpus = self.transformer[self.corpus]

        self.assertIsInstance(self.index_corpus, TransformedCorpus)
        print iter(self.index_corpus).next()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSimilarityTransformer)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
