"""
Tests that pipeline components fit together.

This is more of an integration testing scenario: we want to be sure that
individual changes, although well-tested by unit tests, do not somehow break
compatibility with the pipeline-building framework. Note that a lot of
functions from safire.utils.transcorp should be tested here.
"""
import os
from PIL import Image
from gensim.interfaces import TransformedCorpus
from gensim.models import TfidfModel
import logging
from gensim.similarities import Similarity
import numpy
import time
import operator
import theano
import webbrowser
from safire.data.composite_corpus import CompositeCorpus
from safire.data.document_filter import DocumentFilterTransform
from safire.data.filters.frequency_filters import zero_length_filter
from safire.data.loaders import IndexLoader
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.introspection.interfaces import IntrospectionTransformer
from safire.introspection.writers import HtmlVocabularyWriter, HtmlSimpleWriter, \
    HtmlStructuredFlattenedWriter, HtmlImageWriter, HtmlSimilarImagesWriter
from safire.learning.interfaces import SafireTransformer, \
    MultimodalClampedSamplerModelHandle
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import DenoisingAutoencoder
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.transformations import FlattenComposite
from safire.utils import parse_textdoc2imdoc_map
from safire.utils.transcorp import dimension, get_id2word_obj, \
    get_composite_source, reset_vtcorp_input, get_transformers, \
    run_transformations, log_corpus_stack, bottom_corpus, keymap2dict, \
    compute_docname_flatten_mapping, docnames2indexes, get_id2doc_obj
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset, \
    CastPipelineAsDataset
from safire.utils.transformers import LeCunnVarianceScalingTransform, \
    GeneralFunctionTransform, SimilarityTransformer, ItemAggregationTransform
from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from test.safire_test_case import SafireTestCase

__author__ = 'hajicj@ufal.mff.cuni.cz'

import unittest

# Pipeline settings. These will be migrated to some Config class (thin wrapper
# around YAML?)
vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
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

n_items_requested = 2

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

        dataset_caster = CastPipelineAsDataset()
        dataset = dataset_caster[self.pipeline]

        self.assertEqual(dataset.dim, 100)

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

        # t2i_map = parse_textdoc2imdoc_map(t2i_file)
        # t2i_list = [[text, image]
        #             for text in t2i_map
        #             for image in t2i_map[text]]
        # t2i_indexes = docnames2indexes(multimodal_dataset, t2i_list)
        t2i_indexes = compute_docname_flatten_mapping(multimodal_dataset,
                                                      t2i_file)

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
        #
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

        # Generate the dictionary
        freqfiltered_dict = keymap2dict(get_id2word_obj(doc_pipeline))
        # Initialize token vtcorp and lock the dictionary
        token_vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                   tokens=True,
                                   dictionary=freqfiltered_dict,
                                   **vtcorp_settings)
        token_vtcorp.lock()
        token_vtcorp.dry_run()

        # Initialize word2vec
        word2vec = Word2VecTransformer(self.edict_pkl_fname,
                                       get_id2word_obj(token_vtcorp))
        token_word2vec_pipeline = word2vec[token_vtcorp]

        # Add empty document filtering.
        word2vec_miss_filter = DocumentFilterTransform(zero_length_filter)
        token_word2vec_pipeline = word2vec_miss_filter[token_word2vec_pipeline]

        ttanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        token_word2vec_pipeline = ttanh[token_word2vec_pipeline]

        # At this point, token_word2vec_pipeline does *NOT* support __getitem__.
        # This would start being a problem when flattening the dataset based on
        # indices. So, we serialize.
        serializer = Serializer(token_word2vec_pipeline, ShardedCorpus,
                                self.loader.pipeline_serialization_target(
                                    '.tokenw2v'))
        token_word2vec_pipeline = serializer[token_word2vec_pipeline]

        doc = iter(token_word2vec_pipeline).next()
        self.assertEqual(len(doc), 200)

        # Combine the pipeline with the image data.
        tdata = Dataset(token_word2vec_pipeline)

        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')
        #icorp.dry_run()

        itanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        icorp = itanh[icorp]

        print '--serializing icorp--'
        serializer = Serializer(icorp, ShardedCorpus,
                                self.loader.pipeline_serialization_target(
                                    '.icorp'))
        icorp = serializer[icorp]
        idata = Dataset(icorp)

        print '--building mmdata--'
        mmdata = CompositeDataset((tdata, idata), names=('text', 'img'),
                                  aligned=False)

        print '--flattening mmdata--'
        t2i_file = os.path.join(self.loader.root,
                                self.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)
        flatten = FlattenComposite(mmdata, indexes=t2i_indexes)

        flat_mmdata = flatten[mmdata]

        # Serialize.
        print '--serializing flattened mmdata--'
        serializer = Serializer(flat_mmdata, ShardedCorpus,
                                self.loader.pipeline_serialization_target(
                                    '.tokenw2v_mmdata'))
        serialized_mmdata = serializer[flat_mmdata]

        print '--casting to dataset--'
        dataset = Dataset(serialized_mmdata)
        dataset.set_test_p(0.1)
        dataset.set_devel_p(0.1)

        print 'Total token dataset length: {0}'.format(len(dataset))

        # Train a model over the combination.
        self.model_handle = DenoisingAutoencoder.setup(dataset,
            n_out=200,
            activation=theano.tensor.tanh,
            backward_activation=theano.tensor.tanh,
            reconstruction='mse',
            heavy_debug=False)

        self.learner = BaseSGDLearner(20, 400, validation_frequency=10,
                                      plot_transformation=False)

        print '--running training--'
        start = time.clock()
        sftrans = SafireTransformer(self.model_handle,
                                    dataset,
                                    self.learner,
                                    dense_throughput=False)
        end = time.clock()
        print '  total trainng time: {0} s'.format(end - start)
        output = sftrans[dataset]

        self.assertIsInstance(output, TransformedCorpus)

    def test_multimodal_retrieval(self):
        """Tests a scenario where a multimodal pipeline is created,
        a model is trained, a ClampedMultimodalSamplingHandle is created and
        retrieval is attempted from an index by means of SimilarityTransformer.
        The index is built from the highest available level of the image data
        source.

        Uses over a dozen safire features:

        * POS filtering,
        * token-based retrieval,
        * word2vec conversion,
        * filtering word2vec vocabulary misses,
        * tanh transform on both text and image pipeline,
        * serialization of text and image pipeline pre-flattening,
        * combining text and image pipelines by indexes derived from t2i file
          for token-based corpus,
        * serialization post-flattening; clearing pre-flattening temporary data,
        * DenoisingAutoencoder training,
        * clamped sampling from joint model,
        * conversion of image pipeline to similarity index,
        * querying image index with text queries converted by trained pipeline,
        * combining the token queries,
        * building an introspection pipeline that shows extracted tokens and
          retrieved images and their similarities

        This scenario does NOT test further processing of similarity query
        results, it only collects those queries.
        """
        # Building the text pipeline.
        pre_w2v_text_pipeline = VTextCorpus(self.vtlist,
                                            input_root=self.data_root,
                                            tokens=True,
                                            **vtcorp_settings)
        pre_w2v_text_pipeline.dry_run()

        # TODO: Derive tfidf weights.
        doc_text_pipeline = VTextCorpus(self.vtlist,
                                        input_root=self.data_root,
                                        tokens=False,
                                        **vtcorp_settings)
        tfidf = TfidfModel[doc_text_pipeline]
        tfidf_pipeline = tfidf[doc_text_pipeline]

        # TODO: Downsample document based on tfidf weights.
        # TfIdf weights are global for each vocabulary member.
        # Serialize text.
        # prew2v_t_serializer = Serializer(pre_w2v_text_pipeline, ShardedCorpus,
        #                           self.loader.pipeline_serialization_target(
        #                               '.pre_w2v.tcorp'))
        # s_prew2v_t_pipeline = prew2v_t_serializer[pre_w2v_text_pipeline]

        word2vec = Word2VecTransformer(self.edict_pkl_fname,
                                       get_id2word_obj(pre_w2v_text_pipeline))
        text_pipeline = word2vec[pre_w2v_text_pipeline]

        # Add empty document filtering.
        word2vec_miss_filter = DocumentFilterTransform(zero_length_filter)
        text_pipeline = word2vec_miss_filter[text_pipeline]

        # Squish to (-1, 1) activation range.
        ttanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        text_pipeline = ttanh[text_pipeline]


        # Serialize text.
        t_serializer = Serializer(text_pipeline, ShardedCorpus,
                                  self.loader.pipeline_serialization_target(
                                      '.tcorp'))
        text_pipeline = t_serializer[text_pipeline]

        # Building the image pipeline.
        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        image_pipeline = ImagenetCorpus(image_file, delimiter=';',
                                        dim=4096, label='')
        image_pipeline.dry_run()

        itanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        image_pipeline = itanh[image_pipeline]

        # Serialize images.
        i_serializer = Serializer(image_pipeline, ShardedCorpus,
                                  self.loader.pipeline_serialization_target(
                                      'icorp'))
        image_pipeline = i_serializer[image_pipeline]

        # Flatten dataset
        print '-- flattening dataset --'
        idata = Dataset(image_pipeline)
        tdata = Dataset(text_pipeline)
        mmdata = CompositeDataset((tdata, idata), names=('txt', 'img'),
                                  aligned=False)
        t2i_file = os.path.join(self.loader.root,
                                self.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)
        flatten = FlattenComposite(mmdata, indexes=t2i_indexes)
        mm_pipeline = flatten[mmdata]

        # Serialize
        print '-- serializing multimodal data --'
        mm_serializer = Serializer(mm_pipeline, ShardedCorpus,
                                   self.loader.pipeline_serialization_target(
                                       'mmcorp'))
        mm_pipeline = mm_serializer[mm_pipeline]

        # Train model
        print '-- training model --'
        dataset = Dataset(mm_pipeline)
        self.model_handle = DenoisingAutoencoder.setup(
            dataset,
            n_out=200,
            activation=theano.tensor.nnet.sigmoid,  # Sampleable activation
            backward_activation=theano.tensor.tanh,
            reconstruction='mse',
            heavy_debug=False)

        self.learner = BaseSGDLearner(5, 400, validation_frequency=4,
                                      plot_transformation=False)

        sftrans = SafireTransformer(self.model_handle,
                                    dataset,
                                    self.learner,
                                    dense_throughput=False)
        mm_pipeline = sftrans[mm_pipeline]

        # Build clamped sampler
        print '-- building clamped sampler --'
        t2i_handle = MultimodalClampedSamplerModelHandle.clone(
            self.model_handle,
            dim_text=get_composite_source(mm_pipeline, 'txt').dim,
            dim_img=get_composite_source(mm_pipeline, 'img').dim,
            k=10,
            sample_visible=False,
        )
        print '-- applying clamped sampler --'
        t2i_transformer = SafireTransformer(t2i_handle)
        t2i_pipeline = t2i_transformer[text_pipeline]

        # Build index
        print '-- building index --'
        iloader = IndexLoader(self.data_root, 'test-data')
        index = Similarity(iloader.output_prefix('.img'), image_pipeline,
                           num_features=dimension(image_pipeline),
                           num_best=10)

        # Add similarity search
        print '-- building similarity transformer --'
        similarity_transformer = SimilarityTransformer(index=index)
        retrieval_pipeline = similarity_transformer[t2i_pipeline]

        print '-- building aggregation transformer by source doc --'
        aggregator = ItemAggregationTransform()
        retrieval_pipeline = aggregator[retrieval_pipeline]

        print '-- resetting vtcorp input --'
        reset_vtcorp_input(retrieval_pipeline, self.vtlist)

        print '-- retrieving sampled images --'
        start = time.clock()
        sampled_images = [img for img in t2i_pipeline[:n_items_requested]]
        end = time.clock()
        print '       Done: {0} images in {1} s'.format(len(sampled_images),
                                                        end - start)
        print '       First image: {0}'.format(sampled_images[0])

        print '-- querying similarity index --'
        start = time.clock()
        query_results = [qres
                         for qres in retrieval_pipeline[:n_items_requested]]
        end = time.clock()
        print '       Done: {0} images in {1} s'.format(len(sampled_images),
                                                        end - start)
        # print '       First image: {0}'.format(sampled_images[0])

        print '-- building introspection pipeline --'
        # Introspection of results: combine retrieval_pipeline (multi-image
        # writer?) and the document pipeline

        # The introspection composite corpus is aligned - there's a source
        # document for each text.
        # print 'Doc text pipeline, first three:\n{0}' \
        #       ''.format(doc_text_pipeline[:n_items_requested])
        # print 'Retrieval pipeline, first three:\n{0}' \
        #       ''.format(retrieval_pipeline[:n_items_requested])

        intro_combined_corpus = CompositeCorpus((doc_text_pipeline,
                                                 retrieval_pipeline),
                                                aligned=True)
        # print 'Composite corpus for introspection:\n{0}' \
        #       ''.format(log_corpus_stack(intro_combined_corpus))
        intro_flatten = FlattenComposite(intro_combined_corpus,
                                         structured=True)
        intro_flattened_pipeline = intro_flatten[intro_combined_corpus]

        # intro_pipeline_items = intro_flattened_pipeline[:n_items_requested]
        # print 'Intro pipeline items:\n{0}'.format(intro_pipeline_items)
        # print 'Intro pipeline item 0:\n{0}'.format(intro_pipeline_items[0])

        twriter = HtmlVocabularyWriter(root=self.loader.root,
                                       top_k=30,
                                       min_freq=0.001)
        iwriter = HtmlSimilarImagesWriter(
            root=os.path.join(self.loader.root, self.loader.layout.img_dir),
            image_id2doc=get_id2doc_obj(image_pipeline))
        composite_writer = HtmlStructuredFlattenedWriter(root=self.loader.root,
                                                         writers=(twriter,
                                                                  iwriter))
        introspection = IntrospectionTransformer(intro_flattened_pipeline,
                                                 writer=composite_writer)
        introspected_pipeline = introspection[intro_flattened_pipeline]
        # idocs = [idoc for idoc in introspected_pipeline[:n_items_requested]]
        iid2intro = introspected_pipeline.obj.iid2introspection_filename
        filenames = [iid2intro[iid] for iid in sorted(iid2intro.keys())]
        for f in filenames:
            self.assertTrue(os.path.isfile(f))
        # webbrowser.open(filenames[0])

        print '-- mapping query results to images --'
        img_id2doc = get_id2doc_obj(image_pipeline)
        query_results_docs = []
        for qres in query_results:
            query_results_docs.append([(img_id2doc[iid], sim)
                                       for iid, sim in qres])
        sorted_query_results_docs = [sorted(qres_d, key=operator.itemgetter(1),
                                            reverse=True)
                                     for qres_d in query_results_docs]

        print '\n'.join(['{0}'.format(d) for d in sorted_query_results_docs])

        print '-- testing visualization --'
        img_root = os.path.join(self.loader.root,
                                self.loader.layout.img_dir)
        image = Image.open(os.path.join(img_root,
                                        sorted_query_results_docs[0][0][0]))
        # image.show()
        print 'Image: {0}'.format(image)

        self.assertTrue(len(sampled_images) == len(query_results))


###############################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestPipeline)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
