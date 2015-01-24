"""
Tests that pipeline components fit together.

This is more of an integration testing scenario: we want to be sure that
individual changes, although well-tested by unit tests, do not somehow break
compatibility with the pipeline-building framework. Note that a lot of
functions from safire.utils.transcorp should be tested here.
"""
import os
from gensim.models import TfidfModel
import logging
import numpy
from safire.utils.transcorp import dimension
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset
from safire.utils.transformers import LeCunnVarianceScalingTransform, \
    GeneralFunctionTransform
from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from test.safire_test_case import SafireTestCase

__author__ = 'hajicj@ufal.mff.cuni.cz'

import unittest

vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A', 'V'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True}
vtlist_fname = 'test-data.vtlist'

freqfilter_settings = {'k': 110,
                       'discard_top': 10}

tanh = 0.5

serialization_fname = 'serialized.shcorp'

##############################################################################

class TestPipeline(SafireTestCase):

    def setUp(self):
        self.vtlist = os.path.join(self.data_root, vtlist_fname)

        self.vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                  **vtcorp_settings)
        pipeline = self.vtcorp

        #self.tfidf = TfidfModel(self.vtcorp)
        #pipeline = self.tfidf[pipeline]

        #self.freqfilter = FrequencyBasedTransformer(pipeline,
        #                                            **freqfilter_settings)
        #pipeline = self.freqfilter[pipeline]

        #self.ucov = LeCunnVarianceScalingTransform(pipeline)
        #pipeline = self.ucov[pipeline]

        #self.tanh = GeneralFunctionTransform(numpy.tanh,
        #                                     multiplicative_coef=tanh)
        #pipeline = self.tanh[pipeline]

        serization_file = os.path.join(self.data_root, 'corpora',
                                       serialization_fname)
        self.serializer = Serializer(pipeline, ShardedCorpus,
                                     fname=serization_file)
        self.pipeline = self.serializer[pipeline]  # Swapout corpus

    def test_cast_to_dataset(self):

        dataset = Dataset(self.pipeline, dim=dimension(self.pipeline))
        self.assertEqual(dataset.dim, 100)
        batch = dataset[1:4]
        print batch

###############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestPipeline)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
