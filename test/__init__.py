"""
This package contains unit test for all other safire packages.
"""

import logging
import os
import unittest
import gensim
from safire.data.imagenetcorpus import ImagenetCorpus

from safire.data.loaders import MultimodalShardedDatasetLoader
from scripts import clean

##############################################################################


def clean_data_root(root, name='test-data'):
    """Cleans the given root."""
    parser = clean.build_argument_parser()
    clean_args = parser.parse_args(['-r', root, '-n', name, '-f'])
    clean.main(clean_args)


class SafireTestCase(unittest.TestCase):
    """This class implements the setUpClass and tearDownClass fixtures
    to keep the test data for safire consistent for all further TestCases.

    Upon setUpClass, it will clean the test data root and build the default
    corpora and datasets needed for test operation. Upon tearDownClass, it
    will clean the test data root: corpora, datasets, models, learners, etc.

    All Safire test cases should inherit from this class; if they implement
    their own further setUpClass and tearDownClass methods, these should call
    `their `super()``. (The setUpClass method should call it as the first
    thing it does, the tearDownClass method should call it as the *last*
    thing it does.)
    """
    @classmethod
    def setUpClass(cls):

        cls.testdir = os.path.dirname(__file__)
        cls.data_root = os.path.join(cls.testdir, 'test-data')

        # Clean the test data - in case the previous TestCase was NOT
        # a SafireTestCase
        clean_data_root(cls.data_root)

        # Re-generate the default corpora/datasets.
        cls.loader = MultimodalShardedDatasetLoader(cls.data_root,
                                                    'test-data')
        cls.loader.build_default_text_corpora(serializer=gensim.corpora.MmCorpus)
        cls.loader.build_default_image_corpora(
            serializer=gensim.corpora.MmCorpus)

        default_vtcorp = cls.loader.load_text_corpus()
        cls.loader.build_text(default_vtcorp,
                              dataset_init_args={'overwrite': True})

        default_icorp = cls.loader.load_image_corpus()
        cls.loader.build_img(default_icorp,
                             dataset_init_args={'overwrite': True})

    @classmethod
    def tearDownClass(cls):
        """Cleans up the test data.

        Just in case the next TestCase is NOT a SafireTestCase:"""
        clean_data_root(cls.data_root)

    def test_has_default_corpora(self):
        """Checks that the default corpora and datasets were created
        successfully."""
        # None of these should raise an exception.
        no_exceptions = False
        try:
            self.loader.load_text_corpus()
            self.loader.load_image_corpus()
            self.loader.load_text()
            self.loader.load_img()
            no_exceptions = True
        except ValueError:
            pass

        self.assertTrue(no_exceptions)