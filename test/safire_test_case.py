import os
import gensim
import logging
from safire.data import VTextCorpus
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.layouts import clean_data_root, init_data_root
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus

__author__ = 'hajicj'

import unittest


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
    #@profile
    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        """Sets up the default data root structure (clean root + default
        corpora and datasets).

        :param clean_only: If set, will not create default corpora/datasets.
        :type clean_only: bool

        :param no_datasets: If set, will only create corpora, not
            ShardedDatasets.
        :type no_datasets: bool
        """

        cls.testdir = os.path.dirname(__file__)
        cls.data_root = os.path.join(cls.testdir, 'test-data')

        # Clean the test data - in case the previous TestCase was NOT
        # a SafireTestCase
        init_data_root(cls.data_root, overwrite=True)

        cls._clean_only = clean_only
        if cls._clean_only:
            return

        # Re-generate the default corpora/datasets.
        cls.loader = MultimodalShardedDatasetLoader(cls.data_root,
                                                    'test-data')
        cls.loader.build_default_text_corpora(serializer=gensim.corpora.MmCorpus)
        cls.loader.build_default_image_corpora(
            serializer=gensim.corpora.MmCorpus)

        vtlist = os.path.join(cls.loader.root, cls.loader.layout.vtlist)
        cls.vtcorp = VTextCorpus(vtlist, input_root=cls.loader.root)
        cls.vtcorp_name = cls.loader.pipeline_name('')
        cls.vtcorp.save(cls.vtcorp_name)

        ivectors = os.path.join(cls.loader.root,
                                cls.loader.layout.image_vectors)
        cls.icorp = ImagenetCorpus(ivectors, delimiter=';', dim=4096)
        cls.icorp_name = cls.loader.pipeline_name('.img')
        cls.icorp.save(cls.icorp_name)

        cls._no_datasets = no_datasets
        if not no_datasets:
            default_vtcorp = cls.loader.load_text_corpus()
            cls.loader.build_text(default_vtcorp,
                                  dataset_init_args={'overwrite': True})

            default_icorp = cls.loader.load_image_corpus()
            cls.loader.build_img(default_icorp,
                                 dataset_init_args={'overwrite': True})

            cls.vtcorp_s_name = cls.loader.pipeline_serialization_target()
            cls.vtcorp.dry_run()
            t_serializer = Serializer(cls.vtcorp, ShardedCorpus,
                                      cls.vtcorp_s_name)

            cls.icorp_s_name = cls.loader.pipeline_serialization_target('.img')
            i_serializer = Serializer(cls.icorp, ShardedCorpus,
                                      cls.icorp_s_name)



    @classmethod
    def tearDownClass(cls):
        """Cleans up the test data.

        Just in case the next TestCase is NOT a SafireTestCase:"""
        clean_data_root(cls.data_root)

    def test_has_default_corpora(self):
        """Checks that the default corpora and datasets were created
        successfully."""
        # None of these should raise an exception.
        if self._clean_only:
            logging.info('Clean-only SafireTestCase, not performing check'
                         'for default corpus/dataset existence.')
            return
        no_exceptions = False
        try:
            self.loader.load_text_corpus()
            self.loader.load_image_corpus()
            if not self._no_datasets:
                self.loader.load_text()
                self.loader.load_img()
            no_exceptions = True
        except ValueError:
            pass

        self.assertTrue(no_exceptions)

########################################################################


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(SafireTestCase)
    suite.addTests(tests)

    runner = unittest.TextTestRunner()
    runner.run(suite)
