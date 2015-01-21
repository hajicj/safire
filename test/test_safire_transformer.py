"""
Testing the test procedure itself.
"""

import logging
import os
import unittest

from gensim import similarities
from gensim.interfaces import TransformedCorpus
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.vtextcorpus import VTextCorpus
from safire.data.loaders import MultimodalDatasetLoader, IndexLoader, \
    MultimodalShardedDatasetLoader
from safire.learning.models.denoising_autoencoder import DenoisingAutoencoder
from safire.learning.models.logistic_regression import LogisticRegression
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from safire.learning.interfaces.safire_transformer import SafireTransformer
from safire.utils.transcorp import bottom_corpus, reset_vtcorp_input
from test.safire_test_case import SafireTestCase


class TestSafireTransformer(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestSafireTransformer, cls).setUpClass()

        cls.iloader = IndexLoader(cls.data_root, 'test-data')
        cls.output_prefix = cls.iloader.output_prefix()

        dataset = cls.loader.load_text()

        # Let's compress the dataset to 10 dimensions, for testing speed.
        # (Testing will take long anyway, due to compilation.)
        cls.model_handle = DenoisingAutoencoder.setup(dataset, n_out=10)

        cls.learner = BaseSGDLearner(3, 2, validation_frequency=4)
        cls.learner.run(cls.model_handle, dataset)

        cls.model = cls.model_handle.model_instance

        cls.savename = 'test-data.mhandle'

    @classmethod
    def tearDownClass(cls):

        del cls.learner
        del cls.model_handle
        del cls.loader

        super(TestSafireTransformer, cls).tearDownClass()

    def test_init(self):

        # Non-training setup
        transformer = SafireTransformer(self.model_handle)

        self.assertIsInstance(transformer, SafireTransformer)
        self.assertIsInstance(transformer.model_handle.model_instance,
                              DenoisingAutoencoder)
        self.assertIs(transformer.model_handle, self.model_handle)

        before_training = self.model.W.get_value()[0, 0]

        print "Before training: %f" % before_training

        dataset = self.loader.load_text()
        transformer = SafireTransformer(self.model_handle,
                                        dataset, self.learner)

        self.assertIs(self.model, transformer.model_handle.model_instance)

        # Training should change the model.
        after_training = self.model.W.get_value()[0, 0]

        print "After training: %f" % after_training

        self.assertNotEqual(before_training, after_training)

    def test_getitem(self):

        transformer = SafireTransformer(self.model_handle)

        corpus = self.loader.get_text_corpus({})
        self.assertIsInstance(corpus, VTextCorpus)

        applied_corpus = transformer[corpus]

        transformed_item = applied_corpus.__iter__().next()

        print transformed_item ### DEBUG

        self.assertEqual(transformer.n_out, len(transformed_item))

    def test_applied_saveload(self):

        transformer = SafireTransformer(self.model_handle)

        corpus = self.loader.get_text_corpus({})
        self.assertIsInstance(corpus, VTextCorpus)

        saveload_infix = '.applied'
        applied_corpus = transformer[corpus]

        self.loader.save_text_corpus(applied_corpus, saveload_infix)

        loaded_corpus = self.loader.load_text_corpus(saveload_infix)

        self.assertIsInstance(loaded_corpus, TransformedCorpus)

        # How to feed texts?
        # - there has to be a .vtlist file which is given to the VTextCorpus
        #   on the bottom of the TransformedCorpus stack.
        vtlist_file = os.path.join(self.data_root, self.loader.layout.vtlist)
        reset_vtcorp_input(loaded_corpus, vtlist_file)

        outputs = [ item for item in loaded_corpus ]

        loaded_transformer = loaded_corpus.obj
        loaded_model = loaded_transformer.model_handle.model_instance
        loaded_model_W = loaded_model.W.get_value()
        self.assertEqual(self.model.W.get_value()[0,0], loaded_model_W[0,0])

        #print type(loaded_corpus.corpus)

        self.assertEqual(10, len(outputs))


    def test_query(self):

        dataset = self.loader.load()
        dataset.set_mode(1)
        query_model_handle = LogisticRegression.setup(dataset)

        img_corpus = self.loader.get_image_corpus({ 'delimiter' : ';'})
        text_corpus = self.loader.get_text_corpus()

        self.assertIsInstance(img_corpus, ImagenetCorpus)
        self.assertIsInstance(text_corpus, VTextCorpus)

        transformer = SafireTransformer(query_model_handle)

        applied_corpus = transformer[text_corpus]

        print applied_corpus

        query = applied_corpus.__iter__().next()

        print query
        print len(query)
        print transformer.n_out

        image = img_corpus.__iter__().next()

        print image
        print len(image)

        similarity_index = similarities.Similarity(self.output_prefix,
                                                   img_corpus,
                                                   transformer.n_out,
                                                   num_best=3)

        query_results = similarity_index[query]

        self.assertEqual(transformer.n_out, len(query))

        print query_results


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSafireTransformer)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
