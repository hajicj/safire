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
from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.learning.interfaces import SafireTransformer
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import DenoisingAutoencoder
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.transformations import FlattenComposite
from safire.utils.transcorp import dimension
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
from safire.utils.transformers import LeCunnVarianceScalingTransform, \
    GeneralFunctionTransform
from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from test.safire_test_case import SafireTestCase

__author__ = 'hajicj@ufal.mff.cuni.cz'

import unittest

# Pipeline settings. These will be migrated to some Config class (thin wrapper
# around YAML?)
vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A', 'V'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}
vtlist_fname = 'test-data.vtlist'

freqfilter_settings = {'k': 110,
                       'discard_top': 10}

tanh = 0.5

serialization_vtname = 'serialized.vt.shcorp'
serialization_iname = 'serialized.i.shcorp'

if not os.getenv('HOME'):
    homepath = os.getenv('USERPROFILE')
else:
    homepath = os.getenv('HOME')

w2v_data_root = os.path.join(homepath, 'word2vec')
edict_pkl_fname = os.path.join(w2v_data_root, 'ces_wiki.edict.pkl')
e_matrix_fname = os.path.join(w2v_data_root, 'ces_wiki.emtr.pkl')

##############################################################################


class TestPipeline(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestPipeline, cls).setUpClass(clean_only=clean_only,
                                            no_datasets=no_datasets)

        cls.w2v = Word2VecSamplingDatasetTransformer(
            embeddings_matrix=e_matrix_fname)

    def setUp(self):
        self.vtlist = os.path.join(self.data_root, vtlist_fname)

        self.vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                  **vtcorp_settings)
        self.vtcorp.dry_run()
        pipeline = self.vtcorp

        self.tfidf = TfidfModel(self.vtcorp)
        pipeline = self.tfidf[pipeline]

        self.freqfilter = FrequencyBasedTransformer(pipeline,
                                                    **freqfilter_settings)
        pipeline = self.freqfilter[pipeline]

        self.ucov = LeCunnVarianceScalingTransform(pipeline)
        pipeline = self.ucov[pipeline]

        self.tanh = GeneralFunctionTransform(numpy.tanh,
                                             multiplicative_coef=tanh)
        pipeline = self.tanh[pipeline]

        serization_vtfile = os.path.join(self.data_root, 'corpora',
                                       serialization_vtname)
        self.serializer = Serializer(pipeline, ShardedCorpus,
                                     fname=serization_vtfile)
        self.pipeline = self.serializer[pipeline]  # Swapout corpus

    def test_cast_to_dataset(self):

        dataset = Dataset(self.pipeline, dim=dimension(self.pipeline))
        self.assertEqual(dataset.dim, 100)
        batch = dataset[1:4]
        print batch

    def test_setup_multimodal(self):

        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')

        print 'Initializing image serialization...'
        serialization_ifile = os.path.join(self.data_root, serialization_iname)
        iserializer = Serializer(icorp, ShardedCorpus,
                                 fname=serialization_ifile,
                                 overwrite=True)
        ipipeline = iserializer[icorp]

        print 'Image pipeline type: {0}'.format(type(ipipeline))


        # Building datasets
        text_dataset = Dataset(self.pipeline, dim=dimension(self.pipeline))

        text_dataset = self.w2v[text_dataset]

        img_dataset = Dataset(ipipeline, dim=dimension(ipipeline))

        print 'Image dataset length: {0}'.format(len(img_dataset))

        multimodal_dataset = CompositeDataset((text_dataset, img_dataset),
                                              names=('txt', 'img'),
                                              test_p=0.1, devel_p=0.1,
                                              aligned=False)
        flatten = FlattenComposite(multimodal_dataset) # Missing: indexes
        flat_multimodal_corpus = flatten[multimodal_dataset]
        flat_multimodal_dataset = Dataset(flat_multimodal_corpus)

        self.model_handle = DenoisingAutoencoder.setup(flat_multimodal_dataset,
            n_out=10,
            reconstruction='cross-entropy',
            heavy_debug=False)
        self.learner = BaseSGDLearner(3, 1, validation_frequency=4)

        sftrans = SafireTransformer(self.model_handle,
                                    flat_multimodal_dataset,
                                    self.learner)
        output = sftrans[flat_multimodal_dataset]

###############################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestPipeline)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
