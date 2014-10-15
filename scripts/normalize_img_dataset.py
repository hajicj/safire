#!/usr/bin/env python
"""Normalizes an image dataset so that all its items sum to 1."""
import os
from safire.utils.transformers import NormalizationTransform

__author__ = 'Jan Hajic jr.'

import argparse
import logging

from gensim.corpora import MmCorpus

from safire.data.loaders import MultimodalDatasetLoader


def _build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                        ' the directory which is the root of a dataset.' +
                        ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the' +
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-i', '--img_label', help='The image corpus label from'
                                                  'which to load data.')
    parser.add_argument('-l', '--transformation_label', default='.norm',
                        action='store',
                        help='The output label. This is to help distinguish ' +
                        'normalized corpora made with different options. '
                        'Controls saving names. Default is \'.norm\'')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    return parser

def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    dloader = MultimodalDatasetLoader(args.root, args.name)

    icorp = dloader.load_image_corpus(args.img_label)

    transformer = NormalizationTransform()

    normalized_icorp = transformer._apply(icorp)

    corpus_names = dloader.layout.required_img_corpus_names(args.transformation_label)
    corpus_full_path = os.path.join(args.root, corpus_names[0])

    logging.info('Serializing to file %s' % corpus_full_path)

    MmCorpus.serialize(corpus_full_path, normalized_icorp)

    logging.info('Re-saving original corpus object with infix %s' % args.transformation_label)

    dloader.save_image_corpus(normalized_icorp.corpus, args.transformation_label)


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
