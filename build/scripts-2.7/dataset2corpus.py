#!c:\users\lenovo\canopy\user\scripts\python.exe
# -*- coding: utf-8 -*-
import cPickle
import os
import argparse
import logging
import time

from gensim import corpora
from gensim.models import TfidfModel
from gensim.utils import SaveLoad
import numpy
from safire.data import VTextCorpus
from safire.data.document_filter import DocumentFilterTransform
from safire.data.filters.frequency_filters import zero_length_filter

from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.serializer import Serializer, SwapoutCorpus
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
from safire.datasets.sharded_dataset import ShardedDataset
from safire.datasets.sharded_multimodal_dataset import \
    UnsupervisedShardedVTextCorpusDataset
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.datasets.transformations import FlattenComposite
import safire.utils
from safire.utils import profile_run
from safire.data.loaders import MultimodalShardedDatasetLoader, IndexLoader
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.frequency_based_transform import FrequencyBasedTransformer
from safire.utils.transcorp import get_id2word_obj, \
    log_corpus_stack, dimension, compute_docname_flatten_mapping, \
    convert_to_gensim
from safire.utils.transformers import GlobalUnitScalingTransform, \
    LeCunnVarianceScalingTransform, GeneralFunctionTransform, \
    NormalizationTransform, CappedNormalizationTransform, SimilarityTransformer


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################


def _create_transformer(*args, **kwargs):
    """Wrapper for transformer creation, used for profiling."""
    transformer = FrequencyBasedTransformer(*args, **kwargs)
    return transformer


def build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to' +
                         ' the directory which is the root of a dataset.'+
                         ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the'+
                        ' Loader. Has to correspond to the *.vtlist file name.')

    parser.add_argument('-s', '--sentences', action='store_true',
                        help='Consider' +
                         'each sentence of a vtext file as a separate ' +
                         'document (default behavior is document per file)')
    parser.add_argument('-t', '--tokens', action='store_true',
                        help='Consider each token a separate document. All '
                             'token filters apply.')

    parser.add_argument('-i', '--input_label', action='store', default=None,
                        help='For image processing, which is time-consuming: '
                             'use this label to process the given *input*'
                             ' corpus. (Useful for layering tanh, UCov., etc.)')
    parser.add_argument('-l', '--label', action='store',
                        help='The corpus label. This is to help distinguish '
                             'corpora made with different filtering & transform'
                             'ation options. Also controls saving names.')

    parser.add_argument('--images', action='store_true',
                        help='If set, will create an image corpus instead of'
                             ' a text corpus. Note that the script currently '
                             'only supports creating the default image corpus.')

    parser.add_argument('--flatten', action='store_true',
                        help='Will use this t2i file to flatten the provided '
                             'text and image datasets. '
                             'If set, will expect the --f_text and -f_images '
                             'labels to be provided. The provided pipelines'
                             'will then be flattened and serialized with the'
                             ' -l infix. (No futher transformations will be'
                             ' applied.)')
    parser.add_argument('--f_text', action='store', default=None,
                        help='The label of the text pipeline to flatten.')
    parser.add_argument('--f_images', action='store', default=None,
                        help='The label of the image pipeline to flatten.')

    parser.add_argument('--scale', action='store', type=float, default=None,
                        help='Will scale the image dataset to the unit interval'
                             'with the given number as the cutoff for 1.0 after'
                             ' scaling.')
    parser.add_argument('--uniform_covariance', action='store_true',
                        help='If set, will scale the data so that all items'
                             ' have the same covariance according to LeCunn\'s'
                             ' 1998 paper, eq. 13.')
    parser.add_argument('--tanh', action='store', type=float, default=None,
                        help='If set, will run the data through a tanh sigmoid'
                             'function. The higher the number, the steeper'
                             ' the function.')
    parser.add_argument('--normalize', action='store', type=float, default=None,
                        help='If set, will normalize each data item to sum to'
                             ' the given number.')
    parser.add_argument('--capped_normalize', action='store', type=float,
                        default=None,
                        help='If set, will normalize each data item to sum to '
                             'the maximum possible number so that no value in '
                             'the corpus will be higher than the given number.')
    parser.add_argument('--shardsize', type=int, default=4096,
                        help='The output sharded dataset will have this shard '
                             'size.')

    parser.add_argument('--pos', action='store',
                        help='The string of individual POS tags that '+
                        'should be retained inthe corpus. Example: \'NADV\' '+
                        'for nouns, adjectives, adverbs and verbs.')
    parser.add_argument('-k', '--top_k', type=int, default=None,
                        help='Keep only this amount of most frequent '+
                        'words as features.')
    parser.add_argument('--discard_top', type=int, default=0,
                        help='Discard this many '+
                        'most frequent features. (Useful for pruning stoplist'+
                        ' words.)')
    parser.add_argument('--tfidf', action='store_true',
                        help='Apply a TfIdf transformation BEFORE frequency'
                             'filtering. (Can be combined with --post_tfidf.)')
    parser.add_argument('--post_tfidf', action='store_true',
                        help='Apply a TfIdf transformation AFTER frequency '
                             'filtering. (Can be combined with --tfidf.)')
    parser.add_argument('--pfilter', type=float, default=None,
                        help='If used, only lets a certain proportion or number'
                             ' of sentences from the beginning of a document '
                             'contribute to the document. Use float for a '
                             'proportion, int for a fixed number.')
    parser.add_argument('--pfilter_fullfreq', action='store_true',
                        help='If used, counts frequencies for words that pass'
                             ' through --pfilter from the whole document, not'
                             ' just from the sentences that pass pfilter.')
    parser.add_argument('--filter_capital', action='store_true',
                        help='If set, will filter out all words starting with '
                             'capital letters.')

    parser.add_argument('--word2vec', action='store',
                        help='If set, will apply word2vec embeddings from the'
                             'given file. (Path given relative to current'
                             'directory.) The file is expected to be a pickled'
                             'embeddings dict.')
    parser.add_argument('--word2vec_op', action='store', default='max',
                        help='The operation word2vec should do to combine word'
                             'embeddings into a document embedding. Supported:'
                             '\'max\', \'sum\' and \'avg\'.')
    parser.add_argument('--word2vec_export', action='store',
                        help='Saves the embeddings trimmed for the '
                             'processed corpus. This will help speed up '
                             'subsequent processing by only loading embeddings'
                             'for the vocabulary present in the processed data.')
    parser.add_argument('--word2vec_dataset', action='store',
                        help='If set, will initialize the *dataset* to be '
                             'a word2vec sampled dataset, sampling one word '
                             'vector per item. This happens *after* data is'
                             'serialized and works only when word2vec is NOT'
                             'applied to the corpus - it applies a'
                             'DatasetTransformer on the dataset before the'
                             'dataset object is saved. [NOT IMPLEMENTED]')

    parser.add_argument('--w2v_filter_empty', action='store_true',
                        help='If set, will filter out all tokens that are not '
                             'in the vocabulary of the word2vec transformer. '
                             'Only applicable if --tokens is also set.')

    parser.add_argument('--no_overwrite', action='store_true',
                        help='If set, will not overwrite an existing serialized'
                             ' dataset, printing a warning and quitting before '
                             '.')
    parser.add_argument('--no_save_corpus', action='store_true',
                        help='If set, will not save the corpus object. ONLY'
                             'for very limited use cases -- normally, in order'
                             'to proceed with training models, you *will* need'
                             'this corpus. (I put this feature in to allow for'
                             'filtering the word2vec embeddings by corpus '
                             'without saving the processed corpora in the '
                             'process.)')

    parser.add_argument('--index', action='store_true',
                        help='If set, will build a similarity index on top of '
                             'the pipeline and save it. The similarity index '
                             'can then be retrieved when running the text to '
                             'image transformation.')

    parser.add_argument('-c', '--clear', action='store_true', help='If given,'+
                        'instead of creating a corpus, will attempt to clear '+
                        'all corpora in the dataset with the infix given by '+
                        'the --label argument. [NOT IMPLEMENTED]')

    parser.add_argument('--profile_main', action='store_true',
                        help='If set, will profile the main() function.')
    parser.add_argument('--profile_serialization', action='store_true',
                        help='If given, will profile the serialization.')
    parser.add_argument('--profile_transformation', action='store_true',
                        help='If given, will profile frequency-based '
                             'transformation initialization time.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    parser.add_argument('--serialization_format', action='store',
                        default='dense',
                        help='One of the following: \'dense\', \'sparse\' '
                             'and \'gensim\'. Defines the format in which the '
                             'data will be serialized: numpy ndarray, scipy '
                             'CSR matrix or gensim sparse vectors. Default '
                             'value is \'dense\'.')

    return parser


def main(args):

    _starttime = time.clock()
    if args.root == 'test':
        args.root = safire.get_test_data_root()
        args.name = 'test-data'
    logging.info('Initializing dataset loader with root %s, name %s' % (args.root, args.name))
    loader = MultimodalShardedDatasetLoader(args.root, args.name)

    if args.clear:
        raise NotImplementedError('Cleaning not implemented properly through'
                                  ' a loader/layout object.')

    if args.flatten:
        if args.f_text is None or args.f_images is None:
            raise argparse.ArgumentError('Must provide --f_text and --f_images'
                                         ' when attempting to flatten.')

        logging.info('Loading text pipeline and casting to dataset...')
        t_pipeline_name = loader.pipeline_name(args.f_text)
        t_pipeline = SaveLoad.load(t_pipeline_name)
        t_data = Dataset(t_pipeline)

        logging.info('Loading image pipeline and casting to dataset...')
        i_pipeline_name = loader.pipeline_name(args.f_images)
        i_pipeline = SaveLoad.load(i_pipeline_name)
        i_data = Dataset(i_pipeline)

        logging.info('Creating composite dataset...')
        mm_data = CompositeDataset((t_data, i_data), names=('text', 'img'),
                                   aligned=False)

        logging.info('Flattening dataset...')
        t2i_file = os.path.join(loader.root,
                                loader.layout.textdoc2imdoc)
        flatten_indexes = compute_docname_flatten_mapping(mm_data, t2i_file)
        flatten = FlattenComposite(mm_data, indexes=flatten_indexes)
        flat_mm_data = flatten[mm_data]

        if not args.label:
            logging.info('Generating flattened label automatically...')
            args.label = '__'.join([args.f_text, args.f_images])
            logging.info('    Generated label: {0}'.format(args.label))

        logging.info('Serializing flattened data...')
        serialization_name = loader.pipeline_serialization_target(args.label)
        serializer = Serializer(flat_mm_data, ShardedCorpus, serialization_name)
        pipeline = serializer[flat_mm_data]

        logging.info('Saving pipeline...')
        pipeline_name = loader.pipeline_name(args.label)
        pipeline.save(fname=pipeline_name)

        return

    if args.input_label is not None:
        logging.info('Loading corpus with label %s' % args.input_label)
        pipeline_fname = loader.pipeline_name(args.input_label)
        pipeline = SaveLoad.load(pipeline_fname)

        logging.info('Loaded corpus report:\n')
        logging.info(log_corpus_stack(pipeline))

    elif args.images:
        logging.info('Reading raw image data.')
        image_file = os.path.join(args.root, loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';',
                               dim=4096, label='')
        pipeline = icorp

    else:
        logging.info('Reading raw text data.')
        vtargs = {}
        if args.label:
            logging.info('VTextCorpus will have label %s' % args.label)
            vtargs['label'] = args.label
        if args.pos:
            logging.info('Constructing POS filter with values {0}'
                         ''.format(list(args.pos)))
            vtargs['token_filter'] = PositionalTagTokenFilter(list(args.pos), 0)
        if args.pfilter:
            logging.info('Constructing positional filter: %d.' % args.pfilter)
            # If a fixed number of sentences is requested, use this.
            if args.pfilter % 1 == 0:
                args.pfilter = int(args.pfilter)
            vtargs['pfilter'] = args.pfilter
            if args.pfilter_fullfreq:
                vtargs['pfilter_full_freqs'] = args.pfilter_fullfreq
        if args.filter_capital:
            vtargs['filter_capital'] = True
        vtargs['tokens'] = args.tokens
        vtargs['sentences'] = args.sentences

        if args.tokens or args.sentences:
            # This already happens automatically inside VTextCorpus, but it
            # raises a warning we can avoid if we know about this in advance.
            vtargs['precompute_vtlist'] = False

        logging.info(u'Deriving corpus from loader with vtargs:\n{0}'.format(
            u'\n'.join(u'  {0}: {1}'.format(k, v)
                       for k, v in sorted(vtargs.items())))
        )

        vtcorp = loader.get_text_corpus(vtargs)
        # VTextCorpus initialization is still the same, refactor or not.
        logging.info('Corpus: %s' % str(vtcorp))
        logging.info('  vtlist: %s' % str(vtcorp.input))

        pipeline = vtcorp  # Holds the data

    if args.tfidf:

        tfidf = TfidfModel(pipeline)
        pipeline = tfidf[pipeline]

    if args.top_k is not None:
        if args.images:
            logging.warn('Running a frequency-based transformer on image data'
                         ' not a lot of sense makes, hmm?')

        logging.info('Running transformer with k=%i, discard_top=%i' % (
            args.top_k, args.discard_top))

        if args.profile_transformation:
            report, transformer = safire.utils.profile_run(_create_transformer,
                                                           pipeline,
                                                           args.top_k,
                                                           args.discard_top)
            # Profiling output
            print report.getvalue()
        else:
            transformer = FrequencyBasedTransformer(pipeline,
                                                    args.top_k,
                                                    args.discard_top)

        pipeline = transformer[pipeline]

    if args.post_tfidf:
        post_tfidf = TfidfModel(pipeline)
        pipeline = post_tfidf[pipeline]

    if args.word2vec is not None:
        logging.info('Applying word2vec transformation with embeddings '
                     '{0}'.format(args.word2vec))
        w2v_dictionary = get_id2word_obj(pipeline)
        # Extracting dictionary from FrequencyBasedTransform supported
        # through utils.transcorp.KeymapDict
        pipeline = convert_to_gensim(pipeline)
        word2vec = Word2VecTransformer(args.word2vec,
                                       w2v_dictionary,
                                       op=args.word2vec_op)
        pipeline = word2vec[pipeline]

    if args.w2v_filter_empty:
        print 'Applying word2vec empty doc filtering.'
        document_filter = DocumentFilterTransform(zero_length_filter)
        pipeline = document_filter[pipeline]

    if args.uniform_covariance:
        ucov = LeCunnVarianceScalingTransform(pipeline)
        pipeline = ucov[pipeline]

    if args.tanh:
        pipeline = convert_to_gensim(pipeline)
        tanh_transform = GeneralFunctionTransform(numpy.tanh,
                                                  multiplicative_coef=args.tanh)
        pipeline = tanh_transform[pipeline]

    if args.capped_normalize is not None:
        logging.info('Normalizing each data point to '
                     'max. value %f' % args.capped_normalize)
        cnorm_transform = CappedNormalizationTransform(pipeline,
                                                        args.capped_normalize)
        pipeline = cnorm_transform[pipeline]

    if args.normalize is not None:
        logging.info('Normalizing each data point to %f' % args.normalize)
        norm_transform = NormalizationTransform(args.normalize)
        pipeline = norm_transform[pipeline]

    logging.info('Serializing...')
    # Rewrite as applying a Serializer block.

    if isinstance(pipeline, VTextCorpus):
        logging.info('Checking that VTextCorpus dimension is available.')
        #if not pipeline.precompute_vtlist:
        #    logging.info('    ...to get dimension: precomputing vtlist.')
        #    pipeline._precompute_vtlist(pipeline.input)
        if pipeline.n_processed < len(pipeline.vtlist):
            logging.info('Have to dry_run() the pipeline\'s VTextCorpus,'
                         'because we cannot derive its dimension.')
            if args.serialization_format == 'gensim':
                logging.info('...deferring dimension check to serialization,'
                             ' as the requested serialization format does not'
                             ' need dimension defined beforehand.')
            else:
                pipeline.dry_run()

    data_name = loader.pipeline_serialization_target(args.label)
    logging.info('  Data name: {0}'.format(data_name))

    serializer_class = ShardedCorpus

    # Here, the 'serializer_class' will not be called directly. Instead,
    # a Serializer block will be built & applied. (Profiling serialization
    # currently not supported.)
    serialization_start_time = time.clock()
    logging.info('Starting serialization: {0}'.format(serialization_start_time))
    sparse_serialization = False
    gensim_serialization = False
    if args.serialization_format == 'sparse':
        sparse_serialization = True
    elif args.serialization_format == 'gensim':
        gensim_serialization = True
    elif args.serialization_format != 'dense':
        logging.warn('Invalid serialization format specified ({0}), serializing'
                     ' as dense.'.format(args.serialization_format))
    serializer_block = Serializer(pipeline, serializer_class,
                                  data_name,
                                  dim=dimension(pipeline),
                                  gensim_serialization=gensim_serialization,
                                  sparse_serialization=sparse_serialization,
                                  overwrite=(not args.no_overwrite),
                                  shardsize=args.shardsize)
    serialization_end_time = time.clock()
    logging.info('Serialization finished: {0}'.format(serialization_end_time))

    logging.debug('After serialization: n_processed = {0}'
                  ''.format(safire.utils.transcorp.bottom_corpus(pipeline).n_processed))

    pipeline = serializer_block[pipeline]

    assert isinstance(pipeline, SwapoutCorpus), 'Serialization not applied' \
                                                ' correctly.'

    if args.index:
        iloader = IndexLoader(args.root, args.name)
        index_name = iloader.output_prefix(args.label)
        logging.info('Building index with name {0}'.format(index_name))
        similarity_transformer = SimilarityTransformer(pipeline, index_name)
        # Should the pipeline get transformed? Or do we only want
        # the transformer?
        # What is the use case here? We need the *transformer*, not the
        # transformed data (that would be just the self-similarity of our
        # dataset), so we need to get some new input. We can retrieve
        # the pipeline.obj and lock the transformer onto another pipeline.
        pipeline = similarity_transformer[pipeline]

    logging.info('Corpus stats: {0} documents, {1} features.'.format(
        len(pipeline),
        safire.utils.transcorp.dimension(pipeline)))

    if not args.no_save_corpus:
        obj_name = loader.pipeline_name(args.label)
        logging.info('Saving pipeline to {0}'.format(obj_name))
        pipeline.save(obj_name)

    # HACK: logging word2vec OOV
    if args.word2vec:
        # Report out-of-vocabulary statistics
        #oov_report = word2vec.report_oov()
        #logging.info(u'OOV report:\n%s' % oov_report)
        word2vec.log_oov()

    if args.word2vec_export:
        word2vec_to_export = word2vec.export_used()
        embeddings_dict = word2vec_to_export.embeddings
        with open(args.word2vec_export, 'wb') as w2v_export_handle:
            cPickle.dump(embeddings_dict, w2v_export_handle, protocol=-1)

    _endtime = time.clock()
    _totaltime = _endtime - _starttime
    logging.info('Total main() runtime: %d s' % int(_totaltime))
    return

     
##############

if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    if args.profile_main:
        report, _ = profile_run(main, args)
        print report.getvalue()
    else:
        main(args)
