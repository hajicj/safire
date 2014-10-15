#!c:\users\lenovo\canopy\user\scripts\python.exe
# -*- coding: utf-8 -*-
import cProfile
import os
import pstats
import sys
import argparse
import logging
import StringIO

from gensim import corpora
from gensim.models import TfidfModel
import numpy
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.sharded_dataset import ShardedDataset
from safire.data.sharded_multimodal_dataset import \
    UnsupervisedShardedVTextCorpusDataset

import safire.utils
from safire.data.vtextcorpus import VTextCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.frequency_based_transform import FrequencyBasedTransformer
from safire.utils.transformers import GlobalUnitScalingTransform, \
    LeCunnVarianceScalingTransform, GeneralFunctionTransform, \
    NormalizationTransform, CappedNormalizationTransform


description="""
Given a vtlist, serializes the dataset using gensim.corpora.MmCorpus.serialize
and saves the VTextCorpus used for reading the vtlist.

Usage:

   dataset2corpus -l dataset.vtlist -r path/to/dataset/root [-s] [-g]
                  -o dataset.mmcorp -c dataset.vtcorp

The difference between the -o and -c option is that -o serializes the
data themselves, while -c exports the corpus object around the data.
"""
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################

def _create_transformer(*args, **kwargs):
    """Wrapper for transformer creation, used for profiling."""
    transformer = FrequencyBasedTransformer(*args, **kwargs)
    return transformer


def build_argument_parser():

    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                         ' the directory which is the root of a dataset.'+
                         ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the'+
                        ' Loader. Has to correspond to the *.vtlist file name.')

    parser.add_argument('-s', '--sentences', action='store_true', help='Consider ' +
                         'each sentence of a vtext file as a separate ' +
                         'document (default behavior is document per file)')
    parser.add_argument('-i', '--input_label', action='store', default=None,
                        help='For image processing, which is time-consuming: '
                             'use this label to process the given *input*'
                             ' corpus. (Useful for layering tanh, UCov., etc.)')
    parser.add_argument('-l', '--label', action='store',
                         help='The corpus label. This is to help distinguish '+
                         'corpora made with different filtering & transformat'+
                         'ion options. Also controls saving names.')

    parser.add_argument('--images', action='store_true',
                        help='If set, will create an image corpus instead of'
                             ' a text corpus. Note that the script currently '
                             'only supports creating the default image corpus.')
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
    parser.add_argument('--dataset_only', action='store_true',
                        help='If set, will assume the tcorp/icorp and mmcorp for the '
                             'given label have already been initialized and will'
                             ' only create the dataset. Will overwrite.')
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

    parser.add_argument('--no_shdat', action='store_true',
                        help='If set, will NOT automatically create the '
                             'sharded dataset with the given label.')
    parser.add_argument('--no_overwrite_shdat', action='store_true',
                        help='If set, will overwrite an existing dataset.')

    parser.add_argument('--serializer', action='store', help='Which '+
                        'gensim serializer to use: Mm, SvmLight, Blei, Low')

    parser.add_argument('-c', '--clear', action='store_true', help='If given,'+
                        'instead of creating a corpus, will attempt to clear '+
                        'all corpora in the dataset with the infix given by '+
                        'the --label argument.')
    parser.add_argument('--profile_serialization', action='store_true',
                        help='If given, will profile the serialization.')
    parser.add_argument('--profile_transformation', action='store_true',
                        help='If given, will profile frequency-based '
                             'transformation initialization time.')


    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    return parser

def main(args):

    logging.info('Initializing dataset loader with root %s, name %s' % (args.root, args.name))
    loader = MultimodalShardedDatasetLoader(args.root, args.name)

    if args.clear:

        fnames = loader.layout.required_text_corpus_names(args.label)
        logging.info('Clearing corpora with label %s' % args.label)

        for name in fnames:
            full_name = os.path.join(loader.root, loader.layout.corpus_dir,
                                     name)

            logging.debug('Attempting to clear file %s' % full_name)
            if os.path.isfile(full_name):
                os.remove(full_name)

        # TODO: Clear dataset
        output_prefix = loader.output_prefix(args.label)

        return

    if args.images:

        if args.dataset_only:
            logging.info('Building dataset only (assumes icorp and mmcorp)...')
            idata = loader.load_img(args.label, { 'overwrite' : True,
                                                  'shardsize' : args.shardsize})
            idata.data.save()
            return

        if loader.has_image_corpora(args.label):
            logging.warn('Overwriting image corpora with label %s' % args.label)

        if args.input_label:
            if args.input_label == '.':
                args.input_label = None
            icorp = loader.load_image_corpus(args.input_label)
        else:
            # Build the default corpus.
            image_file = os.path.join(args.root, loader.layout.image_vectors)
            icorp = ImagenetCorpus(image_file, delimiter=';',
                                   dim=4096, label='')
        corpus_to_serialize = icorp

        if args.uniform_covariance:
            logging.info('Transforming to uniform covariance.')
            covariance_transform = LeCunnVarianceScalingTransform(
                                                            corpus_to_serialize)
            corpus_to_serialize = covariance_transform[corpus_to_serialize]

        if args.scale is not None:
            logging.info('Scaling with cutoff at %f' % args.scale)
            scaling_transform = GlobalUnitScalingTransform(icorp,
                                                            cutoff=args.scale)
            corpus_to_serialize = scaling_transform[corpus_to_serialize]

        if args.tanh:
            logging.info('Squishing through the tanh function with coef. %f' % args.tanh)
            tanh_transform = GeneralFunctionTransform(numpy.tanh,
                                                      multiplicative_coef=args.tanh)
            corpus_to_serialize = tanh_transform[corpus_to_serialize]

        if args.normalize is not None:
            logging.info('Normalizing each data point to %f' % args.normalize)
            norm_transform = NormalizationTransform(args.normalize)
            corpus_to_serialize = norm_transform[corpus_to_serialize]

        # i1 = icorp.__iter__().next()
        # s1 = corpus_to_serialize.__iter__().next()
        # print i1[:20]
        # print s1[:20]

        loader.serialize_image_corpus(corpus_to_serialize, args.label)
        loader.save_image_corpus(corpus_to_serialize, args.label)

        output_prefix = loader.img_output_prefix(args.label)
        dataset = ShardedDataset(output_prefix, corpus_to_serialize,
                                 shardsize=args.shardsize,
                                 overwrite=(not args.no_overwrite_shdat))
        dataset.save()

        return

    ###########################################################################

    # Text processing #

    if args.dataset_only:

        logging.info('Only creating dataset, assuming serialized corpus available for %s' % args.label)
        output_prefix = loader.text_output_prefix(args.label)

        vt_corpus_filename = loader.layout.required_text_corpus_names(args.label)[1]
        vt_full_filename = os.path.join(args.root, loader.layout.corpus_dir,
                                        vt_corpus_filename)


        mm_corpus_filename = loader.layout.required_text_corpus_names(args.label)[0]
        mm_full_filename = os.path.join(args.root, loader.layout.corpus_dir,
                                        mm_corpus_filename)

        dataset = UnsupervisedShardedVTextCorpusDataset(output_prefix,
                                            vt_corpus_filename=vt_full_filename,
                                            mm_corpus_filename=mm_full_filename,
                                            shardsize=args.shardsize,
                                            overwrite=args.overwrite_shdat)
        dataset.data.save() # Save the ShardedDataset, not the wrapper


    if args.input_label is not None:
        logging.info('Loading corpus with label %s' % args.input_label)
        corpus_to_serialize = loader.load_text_corpus(args.input_label)

    else:
        vtargs = {}
        if args.label:
            logging.info('VTextCorpus will have label %s' % args.label)
            vtargs['label'] = args.label
        if args.pos:
            logging.info('Constructing POS filter with values %s' % str(list(args.pos)))
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

        logging.info('Deriving corpus from loader with vtargs %s' % str(vtargs))
        vtcorp = loader.get_text_corpus(vtargs)
        logging.info('Corpus: %s' % str(vtcorp))
        logging.info('  vtlist: %s' % str(vtcorp.input))

        corpus_to_serialize = vtcorp  # Holds the data

    if args.tfidf:

        tfidf = TfidfModel(corpus_to_serialize)
        corpus_to_serialize = tfidf[corpus_to_serialize]

    if args.top_k is not None:

        logging.info('Running transformer with k=%i, discard_top=%i' % (
            args.top_k, args.discard_top))

        if args.profile_transformation:
            report, transformer = safire.utils.profile_run(_create_transformer,
                                     corpus_to_serialize,
                                     args.top_k, args.discard_top)
            # Profiling output
            print report
        else:
            transformer = FrequencyBasedTransformer(corpus_to_serialize,
                                                args.top_k, args.discard_top)

        corpus_to_serialize = transformer._apply(corpus_to_serialize)

        # Cannot save the TransformedCorpus that comes out of TfidfModel
        # - transform original VTextCorpus instead.
        #corpus_to_save = transformer._apply(corpus_to_save)

    if args.post_tfidf:

        post_tfidf = TfidfModel(corpus_to_serialize)
        corpus_to_serialize = post_tfidf[corpus_to_serialize]

    if args.uniform_covariance:

        ucov = LeCunnVarianceScalingTransform(corpus_to_serialize)
        corpus_to_serialize = ucov[corpus_to_serialize]

    if args.tanh:

        tanh_transform = GeneralFunctionTransform(numpy.tanh,
                                                  multiplicative_coef=args.tanh)
        corpus_to_serialize = tanh_transform[corpus_to_serialize]

    if args.capped_normalize is not None:
            logging.info('Normalizing each data point to max. value %f' % args.capped_normalize)
            cnorm_transform = CappedNormalizationTransform(corpus_to_serialize,
                                                          args.capped_normalize)
            corpus_to_serialize = cnorm_transform[corpus_to_serialize]

    if args.normalize is not None:
            logging.info('Normalizing each data point to %f' % args.normalize)
            norm_transform = NormalizationTransform(args.normalize)
            corpus_to_serialize = norm_transform[corpus_to_serialize]

    logging.info('Serializing...')
    cnames = loader.layout.required_text_corpus_names(args.label)

    data_name = os.path.join(loader.root, loader.layout.corpus_dir, cnames[0])
    obj_name = os.path.join(loader.root, loader.layout.corpus_dir, cnames[2])

    logging.info('  Data name: %s' % cnames[0])
    logging.info('  Obj name:  %s' % cnames[2])

    serializer = corpora.MmCorpus

    if args.serializer:
        if args.serializer == 'SvmLight':
            serializer = corpora.SvmLightCorpus
        elif args.serializer == 'Blei':
            serializer = corpora.BleiCorpus
        elif args.serializer == 'Low':
            serializer = corpora.LowCorpus
        elif serializer == 'Mm':
            serializer = corpora.MmCorpus

    if args.profile_serialization:
        logging.info('Profiling serialization.')
        profiler_results, _ = safire.utils.profile_run(serializer.serialize,
                                                       data_name,
                                                       corpus_to_serialize)

        logging.info('Profiling results:')
        print profiler_results.getvalue()
    else:
        serializer.serialize(data_name, corpus_to_serialize)

    # We are saving the VTextCorpus rather than the transformed corpus,
    # in order to be able to load it.

    logging.info('Corpus stats: %d documents, %d features.' % (
        len(corpus_to_serialize), safire.utils.transcorp.dimension(corpus_to_serialize)))

    corpus_to_serialize.save(obj_name)

    if not args.no_shdat:

        output_prefix = loader.text_output_prefix(args.label)

        dataset = ShardedDataset(output_prefix, corpus_to_serialize,
                                 shardsize=args.shardsize,
                                 overwrite=(not args.no_overwrite_shdat))
        dataset.save()


    # Or: should we save the transformed corpus? That would save both the
    # original vtcorp and the transformation object...
    #### corpus_to_serialize.save(obj_name)

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

    main(args)
