#!/usr/bin/env python
"""
Testing Loader classes:

* safire.data.loaders.MultimodalDatasetLoader

"""

import logging
import os
import unittest
from safire.data.corpus_dataset import UnsupervisedVTextCorpusDataset, \
    UnsupervisedImagenetCorpusDataset
from safire.data.imagenetcorpus import ImagenetCorpus

from safire.data.loaders import MultimodalDatasetLoader, ModelLoader
from safire.data.multimodal_dataset import MultimodalDataset
from safire.data.vtextcorpus import VTextCorpus


class TestMultimodalDataset(unittest.TestCase):

    def setUp(self):

        self.testdir = os.path.dirname(__file__)
        self.data_root = os.path.join(self.testdir, 'test-data')

        self.full_corpus_path = os.path.join(self.data_root, 'test-data')
        self.loader = MultimodalDatasetLoader(self.data_root,
                                              'test-data')

        self.test_infix = '.test_label'
        self.temporary_files = []

        vtext_args = { 'label' : self.test_infix }
        img_args = { 'label' : self.test_infix,
                     'delimiter' : ';' }

        if not self.loader.has_corpora(self.test_infix):
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

    def test_init(self):

        dataset = self.loader.load(self.test_infix)

        self.assertIsInstance(dataset, MultimodalDataset)

    def test_get_batch(self):

        dataset = self.loader.load(self.test_infix)

        dim_text = dataset.text.dim
        dim_img = dataset.img.dim

        dataset.set_mode(0)
        multimodal_batch = dataset.train_X_batch(0, 2)
        self.assertEqual((2, dim_text + dim_img), multimodal_batch.shape)

        dataset.set_mode(1)
        text_batch = dataset.train_X_batch(0, 2)
        self.assertEqual((2, dim_text), text_batch.shape)
        img_batch = dataset.train_y_batch(0, 2)
        self.assertEqual((2, dim_img), img_batch.shape)

        dataset.set_mode(2)
        text_batch = dataset.train_y_batch(0, 2)
        self.assertEqual((2, dim_text), text_batch.shape)
        img_batch = dataset.train_X_batch(0, 2)
        self.assertEqual((2, dim_img), img_batch.shape)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    logging.info('Running Loader tests...')
    unittest.main()