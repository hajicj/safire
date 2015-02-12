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
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.learning.interfaces import SafireTransformer, \
    MultimodalClampedSamplerModelHandle
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import DenoisingAutoencoder
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.transformations import FlattenComposite, docnames2indexes
from safire.utils import parse_textdoc2imdoc_map
from safire.utils.transcorp import dimension, get_id2word_obj, \
    get_composite_source, reset_vtcorp_input, get_transformers, \
    run_transformations, log_corpus_stack, bottom_corpus, keymap2dict
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


##############################################################################


class TestPipeline(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestPipeline, cls).setUpClass(clean_only=clean_only,
                                            no_datasets=no_datasets)

        if not os.getenv('HOME'):
            homepath = os.getenv('USERPROFILE')
        else:
            homepath = os.getenv('HOME')

        #cls.w2v_data_root = os.path.join(homepath, 'word2vec')
        cls.edict_pkl_fname = os.path.join(cls.data_root, 'test-data.edict.pkl')
        cls.e_matrix_fname = os.path.join(cls.data_root, 'test-data.emtr.pkl')

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

        serization_vtfile = os.path.join(self.data_root,
                                         self.loader.layout.corpus_dir,
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
        serialization_ifile = os.path.join(self.data_root,
                                           self.loader.layout.corpus_dir,
                                           serialization_iname)
        iserializer = Serializer(icorp, ShardedCorpus,
                                 fname=serialization_ifile,
                                 overwrite=True)
        ipipeline = iserializer[icorp]

        print 'Image pipeline type: {0}'.format(type(ipipeline))

        self.w2v_t = Word2VecTransformer(self.edict_pkl_fname,
                                         id2word=get_id2word_obj(self.pipeline))
        print self.w2v_t
        self.w2v = Word2VecSamplingDatasetTransformer(w2v_transformer=self.w2v_t)

        # Building datasets
        text_dataset = Dataset(self.pipeline, dim=dimension(self.pipeline))
        text_dataset = self.w2v[text_dataset]

        print 'Text dataset: {0}'.format(text_dataset)
        print '    text dim: {0}'.format(text_dataset.dim)

        img_dataset = Dataset(ipipeline, dim=dimension(ipipeline))

        print 'Image dataset: {0}'.format(img_dataset)
        print '    image dim: {0}'.format(img_dataset.dim)

        print '--Constructing multimodal dataset--'
        multimodal_dataset = CompositeDataset((text_dataset, img_dataset),
                                              names=('txt', 'img'),
                                              test_p=0.1, devel_p=0.1,
                                              aligned=False)

        print '--Obtaining text-image mapping--'
        # Get text-image mapping
        t2i_file = os.path.join(self.data_root,
                                self.loader.layout.textdoc2imdoc)
        #with open(t2i_file) as t2i_handle:
        #    t2i_linecount = sum([1 for _ in t2i_handle])

        t2i_map = parse_textdoc2imdoc_map(t2i_file)
        t2i_list = [[text, image]
                    for text in t2i_map
                    for image in t2i_map[text]]
        t2i_indexes = docnames2indexes(multimodal_dataset, t2i_list)

        print '--Creating flattened dataset--'
        flatten = FlattenComposite(multimodal_dataset,
                                   indexes=t2i_indexes)
        print '--applying flattened dataset--'
        flat_multimodal_corpus = flatten[multimodal_dataset]

        print '--serializing flattened data--'
        serialization_name = self.loader.pipeline_serialization_target('__test_mmdata__')
        serializer = Serializer(flat_multimodal_corpus, ShardedCorpus,
                                serialization_name, dim=dimension(flat_multimodal_corpus))
        flat_multimodal_corpus = serializer[flat_multimodal_corpus]

        print '--casting flattened corpus to dataset--'
        flat_multimodal_dataset = Dataset(flat_multimodal_corpus,
                                          ensure_dense=True)
        # Have to ensure dense output, otherwise model setup will be getting
        # gensim vectors!

        print 'Corpus stack:'
        print log_corpus_stack(flat_multimodal_dataset)

        print '--Creating model handle--'
        self.model_handle = DenoisingAutoencoder.setup(flat_multimodal_dataset,
            n_out=10,
            reconstruction='cross-entropy',
            heavy_debug=False)

        print 'Model weights shape: {0}'.format(
            self.model_handle.model_instance.W.get_value(borrow=True).shape)

        batch = flat_multimodal_dataset.train_X_batch(0, 1)
        print 'Batch shape: {0}'.format(batch.shape)

        self.learner = BaseSGDLearner(3, 1, validation_frequency=4)

        print '--running learning--'
        sftrans = SafireTransformer(self.model_handle,
                                    flat_multimodal_dataset,
                                    self.learner,
                                    dense_throughput=False)
        output = sftrans[flat_multimodal_dataset]

        print '--output--'
        print 'Type: {0}'.format(type(output))
        print 'Output: {0}'.format(output)

        print '--extracting text runtime pipeline--'
        text_runtime_pipeline = get_composite_source(flat_multimodal_dataset,
                                                     'txt')
        reset_vtcorp_input(text_runtime_pipeline, self.vtlist)

        print 'Coming out of text_runtime_pipeline: {0}' \
              ''.format(iter(text_runtime_pipeline).next())
        print 'Type: {0}'.format(type(iter(text_runtime_pipeline).next()))
        print 'Shape: {0}'.format(iter(text_runtime_pipeline).next().shape)

        print '--creating clamped sampling handle--'
        text2img_handle = MultimodalClampedSamplerModelHandle.clone(
            sftrans.model_handle,
            dim_text=get_composite_source(flat_multimodal_dataset, 'txt').dim,
            dim_img=get_composite_source(flat_multimodal_dataset, 'img').dim,
            k=10
        )
        print '--applying handle to text runtime pipeline--'
        t2i_transformer = SafireTransformer(text2img_handle)
        text2img_pipeline = t2i_transformer[text_runtime_pipeline]

        print '--applying dense throughput--'


        print '--debugging pipeline run--'
        print log_corpus_stack(text2img_pipeline)
        transformers = get_transformers(text2img_pipeline)
        bottom_corp = bottom_corpus(text2img_pipeline)
        print 'Transformers: {0}'.format('\n'.join([str(tr)
                                                    for tr in transformers]))
        # This runs the transformation bottom-up, *without* first building
        # the recursive call stack.
        _ = run_transformations(iter(bottom_corp).next(), *transformers)
        print '\n...the definitive pipeline:'
        print log_corpus_stack(text2img_pipeline)

        print '--running text to image transformation--'
        ideal_images = [ideal_image for ideal_image in text2img_pipeline]

        print 'Ideal images: {0}'.format(type(ideal_images))

    def test_multimodal_dense_throughput(self):
        """Tests the same setup, but this time forcing dense throughput
        through the Dataset transformers."""
        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')

        print 'Initializing image serialization...'
        serialization_ifile = os.path.join(self.data_root,
                                           self.loader.layout.corpus_dir,
                                           serialization_iname)
        iserializer = Serializer(icorp, ShardedCorpus,
                                 fname=serialization_ifile,
                                 overwrite=True)
        ipipeline = iserializer[icorp]

        print 'Image pipeline type: {0}'.format(type(ipipeline))

        self.w2v_t = Word2VecTransformer(self.edict_pkl_fname,
                                         id2word=get_id2word_obj(self.pipeline))
        print self.w2v_t
        self.w2v = Word2VecSamplingDatasetTransformer(w2v_transformer=self.w2v_t)

        # Building datasets
        text_dataset = Dataset(self.pipeline, dim=dimension(self.pipeline))
        text_dataset = self.w2v[text_dataset]

        print 'Text dataset: {0}'.format(text_dataset)
        print '    text dim: {0}'.format(text_dataset.dim)

        img_dataset = Dataset(ipipeline, dim=dimension(ipipeline))

        print 'Image dataset: {0}'.format(img_dataset)
        print '    image dim: {0}'.format(img_dataset.dim)

        print '--Constructing multimodal dataset--'
        multimodal_dataset = CompositeDataset((text_dataset, img_dataset),
                                              names=('txt', 'img'),
                                              test_p=0.1, devel_p=0.1,
                                              aligned=False)

        print '--Obtaining text-image mapping--'
        # Get text-image mapping
        t2i_file = os.path.join(self.data_root,
                                self.loader.layout.textdoc2imdoc)
        #with open(t2i_file) as t2i_handle:
        #    t2i_linecount = sum([1 for _ in t2i_handle])

        t2i_map = parse_textdoc2imdoc_map(t2i_file)
        t2i_list = [[text, image]
                    for text in t2i_map
                    for image in t2i_map[text]]
        t2i_indexes = docnames2indexes(multimodal_dataset, t2i_list)

        print '--Creating flattened dataset--'
        flatten = FlattenComposite(multimodal_dataset,
                                   indexes=t2i_indexes)
        print '--applying flattened dataset--'
        flat_multimodal_corpus = flatten[multimodal_dataset]

        print '--casting flattened corpus to dataset--'
        flat_multimodal_dataset = Dataset(flat_multimodal_corpus,
                                          ensure_dense=True)

        print '\nCorpus stack before model setup:\n'
        print log_corpus_stack(flat_multimodal_dataset)

        print '--Creating model handle--'
        self.model_handle = DenoisingAutoencoder.setup(flat_multimodal_dataset,
            n_out=10,
            reconstruction='cross-entropy',
            heavy_debug=False)

        print 'Model weights shape: {0}'.format(
            self.model_handle.model_instance.W.get_value(borrow=True).shape)

        batch = flat_multimodal_dataset.train_X_batch(0, 1)
        print 'Batch shape: {0}'.format(batch.shape)

        self.learner = BaseSGDLearner(3, 1, validation_frequency=4)

        print '--running learning--'
        sftrans = SafireTransformer(self.model_handle,
                                    flat_multimodal_dataset,
                                    self.learner,
                                    dense_throughput=True)
        output = sftrans[flat_multimodal_dataset]

        print '--output--'
        print 'Type: {0}'.format(type(output))
        print 'Output: {0}'.format(output)

        print '--extracting text runtime pipeline--'
        text_runtime_pipeline = get_composite_source(flat_multimodal_dataset,
                                                     'txt')
        reset_vtcorp_input(text_runtime_pipeline, self.vtlist)

        print 'Coming out of text_runtime_pipeline: {0}'.format(iter(text_runtime_pipeline).next())
        print 'Type: {0}'.format(type(iter(text_runtime_pipeline).next()))
        print 'Shape: {0}'.format(iter(text_runtime_pipeline).next().shape)

        print '--creating clamped sampling handle--'
        text2img_handle = MultimodalClampedSamplerModelHandle.clone(
            sftrans.model_handle,
            dim_text=get_composite_source(flat_multimodal_dataset, 'txt').dim,
            dim_img=get_composite_source(flat_multimodal_dataset, 'img').dim,
            k=10
        )
        print '--applying handle to text runtime pipeline--'
        t2i_transformer = SafireTransformer(text2img_handle)
        text2img_pipeline = t2i_transformer[text_runtime_pipeline]

        print '--debugging pipeline run--'
        print log_corpus_stack(text2img_pipeline)
        transformers = get_transformers(text2img_pipeline)
        bottom_corp = bottom_corpus(text2img_pipeline)
        print 'Transformers: {0}'.format('\n'.join([str(tr)
                                                    for tr in transformers]))
        # This runs the transformation bottom-up, *without* first building
        # the recursive call stack.
        _ = run_transformations(iter(bottom_corp).next(), *transformers)
        print '\n...the definitive pipeline:'
        print log_corpus_stack(text2img_pipeline)

        print '--running text to image transformation--'
        ideal_images = [ideal_image for ideal_image in text2img_pipeline]

        print 'Ideal images: {0}'.format(type(ideal_images))

        print '--running text to image transformation as slice--'
        ideal_slice = text2img_pipeline[0:len(text2img_pipeline)]

        self.assertEqual(len(ideal_images), len(ideal_slice))

    def test_tokenbased_word2vec(self):

        # Set up the standard pipeline with tf-idf and frequency filtering
        # Set up a new VTextCorpus that iterates over tokens.
        # We could just lock the vocabulary and use the same tf-idf and
        # frequency filters, but this would give us empty documents for tokens
        # that get filtered out at the frequency filtering level. Instead, we
        # need to limit the vocabulary of the new VTextCorpus directly and pass
        # the corpus to the tf-idf transformer (which only cares about columns).
        # On top of the token pipeline, we put a simple word2vec transformer.
        doc_vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                 **vtcorp_settings)
        doc_vtcorp.dry_run()

        tfidf = TfidfModel(doc_vtcorp)
        doc_pipeline = tfidf[doc_vtcorp]

        freqfilter = FrequencyBasedTransformer(doc_pipeline,
                                               110, discard_top=10)
        doc_pipeline = freqfilter[doc_pipeline]

        # Pass the dictionary
        freqfiltered_dict = keymap2dict(get_id2word_obj(doc_pipeline))
        token_vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                   tokens=True,
                                   dictionary=freqfiltered_dict,
                                   **vtcorp_settings)
        token_vtcorp.lock()

        # Initialize word2vec
        word2vec = Word2VecTransformer(self.edict_pkl_fname,
                                       get_id2word_obj(token_vtcorp))
        token_word2vec_pipeline = word2vec[token_vtcorp]

        doc = iter(token_word2vec_pipeline).next()
        print doc
        self.assertEqual(len(doc), 200)
        docs = [d for d in token_word2vec_pipeline]
        print 'Total tokens: {0}'.format(len(docs))


###############################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestPipeline)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
