#!/usr/bin/env python
"""
``retrieval_w2v_tfidf.py`` is a scenario that runs the word2vec token-based
retrieval.

Safire introspection to html is used, with two parallel HtmlVocabularyWriters.
At the end, the script will open the first document in a web browser.
"""
import argparse
import copy
import logging
import time
import os
import webbrowser
from gensim.similarities import Similarity
import numpy
from safire.learning.interfaces import SafireTransformer, \
    MultimodalClampedSamplerModelHandle
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import DenoisingAutoencoder
import theano
from safire.data.document_filter import DocumentFilterTransform
from safire.data.filters.frequency_filters import zero_length_filter
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.datasets.dataset import Dataset
from safire.utils.transformers import GeneralFunctionTransform, \
    SimilarityTransformer, ItemAggregationTransform
from safire.data import VTextCorpus
from safire.data.composite_corpus import CompositeCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader, IndexLoader
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.datasets.transformations import FlattenComposite
from safire.introspection.interfaces import IntrospectionTransformer
from safire.introspection.writers import HtmlVocabularyWriter, \
    HtmlStructuredFlattenedWriter, HtmlSimilarImagesWriter
from safire.utils.transcorp import dry_run, compute_docname_flatten_mapping, \
    get_id2word_obj, get_composite_source, dimension, get_id2doc_obj

__author__ = 'Jan Hajic jr.'

# This is really the variable part: block settings
doc_vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                       'pfilter': 0.2,
                       'pfilter_full_freqs': True,
                       'filter_capital': True,
                       'precompute_vtlist': True}

token_vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                         'pfilter': 0.2,
                         'pfilter_full_freqs': True,
                         'filter_capital': True,
                         'precompute_vtlist': False,
                         'tokens': True}

edict_pkl_name = os.path.join('C:/', 'Users', 'Lenovo',
                              'word2vec', 'ces_wiki.edict.pkl')

tfidf_settings = {'normalize': False}

icorp_settings = {'dim': 4096,
                  'delimiter': ';'}

image_serialization_settings = {'overwrite': True}


def main(args):
    logging.info(b'Executing tfidf_vs_plain.py...')
    _start_time = time.clock()

    loader = MultimodalShardedDatasetLoader(root=args.root,
                                            name=args.name)

    logging.info('Build vtcorp.')
    vtlist = os.path.join(loader.root,
                          loader.layout.vtlist)

    logging.info('Document vtlist')
    doc_vtcorp = VTextCorpus(input=vtlist, input_root=loader.root,
                             **doc_vtcorp_settings)

    logging.info('Token vtlist')
    token_vtcorp = VTextCorpus(input=vtlist, input_root=loader.root,
                               **token_vtcorp_settings)
    token_vtcorp_serializer = Serializer(token_vtcorp,
                                         ShardedCorpus,
                                         loader.pipeline_serialization_target('.tok.pre-w2v'),
                                         gensim_serialization=True,
                                         gensim_retrieval=True,
                                         overwrite=args.overwrite)
    token_vtcorp = token_vtcorp_serializer[token_vtcorp]

    logging.info('Word2Vec')
    word2vec = Word2VecTransformer(edict_pkl_name,
                              get_id2word_obj(token_vtcorp))
    w2v_token_pipeline = word2vec[token_vtcorp]

    logging.info('Empty document filtering')
    word2vec_miss_filer = DocumentFilterTransform(zero_length_filter)
    w2v_token_pipeline = word2vec_miss_filer[w2v_token_pipeline]

    logging.info('Compressing into desired range')
    token_tanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
    tanh_w2v_token_pipeline = token_tanh[w2v_token_pipeline]

    logging.info('Serialize.')
    tanh_w2v_serializer = Serializer(tanh_w2v_token_pipeline,
                                     ShardedCorpus,
                                     loader.pipeline_serialization_target('.tanh.w2v.tok'),
                                     overwrite=args.overwrite)
    tanh_w2v_token_pipeline = tanh_w2v_serializer[tanh_w2v_token_pipeline]


    logging.info('Images')
    image_file = os.path.join(loader.root, loader.layout.image_vectors)
    icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')

    itanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
    icorp = itanh[icorp]

    logging.info('Serialize')
    i_serializer = Serializer(icorp, ShardedCorpus,
                              loader.pipeline_serialization_target(
                                  '.icorp'),
                              overwrite=args.overwrite)
    icorp = i_serializer[icorp]

    logging.info('Flatten')
    logging.info('  - build composite')
    mmcorp = CompositeCorpus((tanh_w2v_token_pipeline, icorp),
                             names=('txt', 'img'),
                             aligned=False)

    logging.info('  - get indices')
    t2i_file = os.path.join(loader.root, loader.layout.textdoc2imdoc)
    t2i_indexes = compute_docname_flatten_mapping(mmcorp, t2i_file)

    logging.info('  - create flattening corpus')
    flatten = FlattenComposite(mmcorp, indexes=t2i_indexes)
    flat_mmcorp = flatten[mmcorp]

    logging.info('  - serialize')
    mm_serializer = Serializer(flat_mmcorp, ShardedCorpus,
                               loader.pipeline_serialization_target('.mmdata'),
                               overwrite=args.overwrite)
    serialized_mmcorp = mm_serializer[flat_mmcorp]

    logging.info('Train')
    dataset = Dataset(serialized_mmcorp)

    logging.info('  - setup handle.')
    model_handle = DenoisingAutoencoder.setup(
        dataset,
        n_out=200,
        activation=theano.tensor.nnet.sigmoid,
        backward_activation=theano.tensor.tanh,
        reconstruction='mse',
        heavy_debug=False
    )

    logging.info('  - setup learner')
    learner = BaseSGDLearner(5, 400, validation_frequency=4,
                             plot_transformation=False)

    logging.info('  - apply transformer')
    sftrans = SafireTransformer(model_handle,
                                dataset,
                                learner,
                                dense_throughput=False)
    joint_rep_pipeline = sftrans[serialized_mmcorp]

    logging.info('Create clamped sampler')
    txt_source = get_composite_source(serialized_mmcorp, 'txt')
    img_source = get_composite_source(serialized_mmcorp, 'img')
    logging.info('Composite sources:\ntxt -- {0}: {1}\nimg -- {2}: {3}'
                 ''.format(txt_source, dimension(txt_source),
                           img_source, dimension(img_source)))
    t2i_handle = MultimodalClampedSamplerModelHandle.clone(
        model_handle,
        dim_text=dimension(get_composite_source(serialized_mmcorp, 'txt')),
        dim_img=dimension(get_composite_source(serialized_mmcorp, 'img')),
        k=10,
        sample_visible=False
    )
    t2i_transformer = SafireTransformer(t2i_handle)
    text_pipeline = get_composite_source(serialized_mmcorp, 'txt')
    t2i_pipeline = t2i_transformer[text_pipeline]

    logging.info('Similarity')
    logging.info('  - build index')
    iloader = IndexLoader(loader.root, loader.layout.name)
    index = Similarity(iloader.output_prefix('.img'), icorp,
                       num_features=dimension(icorp),
                       num_best=10)
    logging.info('  - apply similarity transformer')
    similarity_transformer = SimilarityTransformer(index=index)
    retrieval_pipeline = similarity_transformer[t2i_pipeline]

    logging.info('Aggregate by doc')
    aggregator = ItemAggregationTransform()
    doc_retrieval_pipeline = aggregator[retrieval_pipeline]

    logging.info('Run retrieval (on training data)')
    n_items_requested = 10
    query_results = [qres for qres in retrieval_pipeline[:n_items_requested]]

    logging.info('Introspection')
    logging.info('  - flatten retrieval and document')
    intro_composite_corp = CompositeCorpus((doc_vtcorp, doc_retrieval_pipeline),
                                           aligned=True)
    intro_flatten = FlattenComposite(intro_composite_corp,
                                     structured=True)
    intro_flattened_corp = intro_flatten[intro_composite_corp]

    logging.info('  - build writers')
    twriter = HtmlVocabularyWriter(root=loader.root,
                                   top_k=30,
                                   min_freq=0.001)
    iwriter = HtmlSimilarImagesWriter(
        root=os.path.join(loader.root, loader.layout.img_dir),
        image_id2doc=get_id2doc_obj(icorp))
    composite_writer = HtmlStructuredFlattenedWriter(root=loader.root,
                                                     writers=(twriter, iwriter))

    logging.info('  - run introspection pipeline')
    introspection = IntrospectionTransformer(intro_flattened_corp,
                                             writer=composite_writer)
    intro_pipeline = introspection[intro_flattened_corp]
    dry_run(intro_pipeline, max=n_items_requested)

    logging.info('Retrieve introspection filenames and open first intro doc.')
    iid2intro = intro_pipeline.obj.iid2introspection_filename
    filenames = [iid2intro[iid] for iid in sorted(iid2intro.keys())]
    webbrowser.open(filenames[0])


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    # Loader setup args
    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory.')
    parser.add_argument('-n', '--name', required=False,
                        help='The name of the source dataset.')

    parser.add_argument('--overwrite', action='store_true',
                        help='If set, will force overwrite of each serialized '
                             'pipeline stage.')
    parser.add_argument('--pos', action='store',
                        help='The string of individual POS tags that '
                             'should be retained inthe corpus. Example: '
                             '\'NADV\' for nouns, adjectives, adverbs and '
                             'verbs.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO logging messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG logging messages. (May get very '
                             'verbose.')

    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
