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
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.utils.transformers import TfidfModel
from safire.data import VTextCorpus
from safire.data.composite_corpus import CompositeCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.datasets.transformations import FlattenComposite
from safire.introspection.interfaces import IntrospectionTransformer
from safire.introspection.writers import HtmlVocabularyWriter, \
    HtmlStructuredFlattenedWriter, HtmlImageWriter
from safire.utils.transcorp import dry_run, compute_docname_flatten_mapping

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

tfidf_settings = {'normalize': False}

icorp_settings = {'dim': 4096,
                  'delimiter': ';'}

image_serialization_settings = {'overwrite': True}

def main(args):
    logging.info('Executing tfidf_vs_plain.py...')
    _start_time = time.clock()

    loader = MultimodalShardedDatasetLoader(root=args.root,
                                            name=args.name)

    logging.info('Build vtcorp.')
    vtlist = os.path.join(loader.root,
                          loader.layout.vtlist)

    # Document vtlist
    doc_vtcorp = VTextCorpus(input=vtlist, input_root=loader.root,
                             **doc_vtcorp_settings)

    # Token vtlist
    token_vtcorp = VTextCorpus(input=vtlist, input_root=loader.root,
                               **token_vtcorp_settings)

    # Word2Vec



    # Images
    # Flatten
    #  - get indices
    #  - create flattening corpus
    # Train
    # Create clamped sampler
    # Similarity
    # Run retrieval
    # Aggregate by doc
    # Introspection
    #  - flatten retrieval and document
    #  - build writers
    #  - run introspection pipeline


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    # Loader setup args
    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory.')
    parser.add_argument('-n', '--name', required=False,
                        help='The name of the source dataset.')

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
