"""
This package contains unit test for all other safire packages.
"""

import logging
import os
import unittest
from safire.data.imagenetcorpus import ImagenetCorpus

from safire.data.loaders import MultimodalShardedDatasetLoader
from scripts import clean


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

        # Clean the test data
        clean_data_root(cls.data_root)

        # Re-generate the default corpora/datasets.
        cls.loader = MultimodalShardedDatasetLoader(cls.data_root,
                                                    'test-data')
        image_file = os.path.join(cls.data_root,
                                  cls.loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';',
                               dim=4096, label='')
        icorp.save()


