#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing Loader classes:

* safire.data.loaders.MultimodalDatasetLoader

"""

import logging
import os
import unittest
from gensim.corpora import MmCorpus
from gensim.interfaces import TransformedCorpus

from safire.data.loaders import MultimodalDatasetLoader
from safire.data.vtextcorpus import VTextCorpus
from safire.data.frequency_based_transform import FrequencyBasedTransformer
from test import SafireTestCase


class TestFrequencyBasedTransform(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFrequencyBasedTransform, cls).setUpClass()

        cls.full_vtlist_path = os.path.join(cls.data_root,
                                            cls.loader.layout.vtlist)

        cls.test_infix = '.test_label'
        cls.temporary_files = []

    @classmethod
    def tearDownClass(cls):

        if hasattr(cls, 'temporary_files'):
            for temp_file in cls.temporary_files:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                else:
                    logging.warn('Temporary file %s not found for removal.' % temp_file)

        if hasattr(cls, 'loader'):
            del cls.loader
        super(TestFrequencyBasedTransform, cls).tearDownClass()

    def setUp(self):

        self.k = 100
        self.vtcorp = VTextCorpus(self.full_vtlist_path,
                                  input_root=self.data_root)

        self.transformer = FrequencyBasedTransformer(self.vtcorp, self.k)
        self.transformation_label = '.top' + str(self.k)

    def tearDown(self):

        del self.vtcorp
        del self.transformer

    def test_default_label(self):

        self.assertTrue(hasattr(self.transformer, 'label'))
        self.assertEqual('.top100', self.transformer.label)

    def test_init(self):

        self.assertTrue(hasattr(self.transformer, 'freqdict'))
        self.assertEqual(dict, type(self.transformer.freqdict))
        self.assertEqual(frozenset, type(self.transformer.allowed_features))
        self.assertEqual(self.transformer.k,
                         len(self.transformer.allowed_features))

    def test_apply(self):

        transformed_vtcorp = self.transformer._apply(self.vtcorp)

        self.assertTrue(hasattr(transformed_vtcorp.corpus, 'dictionary'))

        transformed_names = self.loader.layout.required_text_corpus_names(self.transformation_label)
        text_data_name = os.path.join(self.data_root,
                                      self.loader.layout.corpus_dir,
                                      transformed_names[0])
        text_obj_name = os.path.join(self.data_root,
                                      self.loader.layout.corpus_dir,
                                      transformed_names[2])

        MmCorpus.serialize(text_data_name, transformed_vtcorp)
        transformed_vtcorp.save(text_obj_name)

        self.assertTrue(self.loader.has_text_corpora(self.transformation_label))

        self.temporary_files.extend([ os.path.join(self.data_root,
                                                   self.loader.layout.corpus_dir,
                                                   transformed_name)
                                      for transformed_name in transformed_names])

        transformed_vtcorp = TransformedCorpus.load(text_obj_name)

        self.assertIsInstance(transformed_vtcorp, TransformedCorpus)
        self.assertIsInstance(transformed_vtcorp.corpus, VTextCorpus)
        self.assertTrue(hasattr(transformed_vtcorp.corpus, 'dictionary'))

        print 'Transformed corpus dictionary size: %i' % len(transformed_vtcorp.corpus.dictionary)
        self.assertEqual(self.k, len(transformed_vtcorp.obj.orig2transformed))


        # print 'Retained features:'
        # feature_report = self.transformer.report_features()
        # for f in feature_report:
        #     print '\t'.join([unicode(ff) for ff in f])

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestFrequencyBasedTransform)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)


