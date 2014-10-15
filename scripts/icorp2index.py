#!/usr/bin/env python
"""Converts an image corpus with the given label to an image index."""
import logging
import safire

__author__ = 'Jan Hajic jr.'


import argparse

from gensim import similarities
from safire.data.loaders import MultimodalDatasetLoader, \
    MultimodalShardedDatasetLoader
from safire.data.loaders import IndexLoader


def _build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                        ' the directory which is the root of a dataset.' +
                        ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the' +
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-l', '--label', action='store', default='',
                        help='The index label. This is to help distinguish ' +
                        'indexes made with different filtering & transformat' +
                        'ion options. Controls saving names. This label will '
                        'be the same for the loaded image corpus and output '
                        'index.')
    parser.add_argument('--text', action='store_true',
                        help='If given, will build a text index instead of an '
                             'image index.')

    parser.add_argument('--use_dataset', action='store_true',
                        help='If set, will load the data from a dataset rather'
                             ' than from a corpus. (The dataset can act like a'
                             ' corpus.) This is for cases when an image corpus'
                             ' has been transformed and serialized.')
    parser.add_argument('--try_loading', action='store_true',
                        help='If --use_dataset is set, this flag will have the'
                             ' dataset object attempt to load the whole dataset'
                             ' into memory instead of streaming it from an'
                             ' IndexedCorpus. Speedup for datasets that fit'
                             ' into memory, slowdown for those that don\'t.')

    parser.add_argument('-c', '--clear', action='store_true', help='If given,' +
                        'instead of creating an index, will attempt to clear ' +
                        'all indexes in the dataset with the infix given by ' +
                        'the --label argument.')
    # parser.add_argument('--profile_index_creation', action='store_true',
    #                     help='If given, will profile index creation time.')


    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    return parser

def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    iloader = IndexLoader(args.root, args.name)

    logging.info('Loading corpus with label %s' % args.label)

    if args.use_dataset:
        # Luckily, the Similarity object can work with numpy arrays as corpus
        # documents.
        #icorp_init_args = { 'try_loading' : args.try_loading }
        if args.text:
            corpus = dloader.load_text(args.label)
        else:
            corpus = dloader.load_img(args.label)
    else:
        if args.text:
            corpus = dloader.load_text_corpus(args.label)
        else:
            corpus = dloader.load_image_corpus(args.label)

    logging.info('Icorp type: %s' % type(corpus))

    if args.text:
        # Text index has a different naming scheme..? That would break the
        # index - data symmetry... but how else to do it?
        logging.warn('Text index selected, %s suffix added to index label.' % iloader.layout.text_corpname)
        index_prefix = iloader.text_output_prefix(args.label)
    else:
        index_prefix = iloader.output_prefix(args.label)

    logging.info('Creating index with prefix %s' % index_prefix)

    dimension = safire.utils.transcorp.dimension(corpus)
    index = similarities.Similarity(index_prefix, corpus,
                                    num_features=dimension)

    if args.text:
        iloader.save_text_index(index, args.label)
    else:
        iloader.save_index(index, args.label)

if __name__ == '__main__':

    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
