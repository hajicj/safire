#!/usr/bin/env python
"""
Testing Loader classes:

* safire.data.loaders.MultimodalDatasetLoader

"""

import logging
import os
import unittest

from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.datasets.multimodal_dataset import MultimodalDataset
from safire.datasets.sharded_multimodal_dataset import \
    UnsupervisedShardedVTextCorpusDataset, \
    UnsupervisedShardedImagenetCorpusDataset
from safire.data.vtextcorpus import VTextCorpus
from test import SafireTestCase


class TestMultimodalShardedDatasetLoader(SafireTestCase):

    def setUp(self):

        self.testdir = os.path.dirname(__file__)
        self.data_root = os.path.join(self.testdir, 'test-data')
        self.loader = MultimodalShardedDatasetLoader(self.data_root,
                                                     'test-data')

        self.test_infix = '.test_label'
        self.temporary_files = []

        vtext_args = { 'label' : self.test_infix }
        img_args = { 'label' : self.test_infix,
                     'delimiter' : ';' }

        self.loader.build_corpora(vtext_args, img_args)

        required_corpora = self.loader.layout.required_corpus_names(self.test_infix)
        required_corpus_files = [ os.path.join(self.loader.layout.corpus_dir,
                                               req_corp)
                                  for req_corp in required_corpora]
        self.temporary_files.extend([os.path.join(self.loader.root, corpus_file)
                                     for corpus_file in required_corpus_files])


    def tearDown(self):

        for temp_file in self.temporary_files:
            os.remove(temp_file)

        del self.loader

    def test_has_corpora(self):

        self.assertTrue(self.loader.has_corpora())
        self.assertTrue(self.loader.has_text_corpora())
        self.assertTrue(self.loader.has_image_corpora())
        self.assertFalse(self.loader.has_corpora('some_infix'))


    def test_build_corpora(self):

        required_corpora = self.loader.layout.required_corpus_names(self.test_infix)
        required_corpus_files = [ os.path.join(self.loader.layout.corpus_dir,
                                               req_corp)
                                  for req_corp in required_corpora]

        for corpus_file in required_corpus_files:
            corpus_file_path = os.path.join(self.loader.root, corpus_file)
            cexists = os.path.isfile(corpus_file_path)
            self.assertTrue(cexists)

    def test_get_corpora(self):

        text_corpus = self.loader.get_text_corpus()

        self.assertIsInstance(text_corpus, VTextCorpus)

        image_corpus = self.loader.get_image_corpus()

        self.assertIsInstance(image_corpus, ImagenetCorpus)

    def test_load(self):

        self.assertTrue(self.loader.has_corpora(self.test_infix))
        self.assertTrue(self.loader.has_text_corpora(self.test_infix))
        self.assertTrue(self.loader.has_image_corpora(self.test_infix))

        dataset = self.loader.load(self.test_infix)

        self.assertIsInstance(dataset, MultimodalDataset)
        self.assertEqual(len(dataset), 20)

        text_dataset = self.loader.load_text(self.test_infix)

        self.assertIsInstance(text_dataset, UnsupervisedShardedVTextCorpusDataset)
        self.assertEqual(10, len(text_dataset))

        img_dataset = self.loader.load_img(self.test_infix)

        self.assertIsInstance(img_dataset, UnsupervisedShardedImagenetCorpusDataset)

        self.assertEqual(20, len(img_dataset))

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestMultimodalShardedDatasetLoader)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
