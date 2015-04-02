#!/usr/bin/env python
"""
``tfidf_vs_plain.py`` is a scenario that shows the difference between text
documents with tf-idf and without and draws the associated image as well.

Two ways of invoking the scenario are available: with pre-built corpora, or from
source data. Using pre-built corpora is good when you have already created
(read: serialized) them and only want to see the difference.

Safire introspection to html is used, with two parallel HtmlVocabularyWriters.
At the end, the script will open the first document in a web browser. Note that
nothing is serialized; the artifact produced by this scenario is just the
introspection files.
"""
import argparse
import copy
import logging
import time
import os
import webbrowser
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.utils.transformers import TfidfModel
from safire.data import VTextCorpus
from safire.data.composite_corpus import CompositeCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.datasets.transformations import FlattenComposite
from safire.introspection.interfaces import IntrospectionTransformer
from safire.introspection.writers import HtmlVocabularyWriter, \
    HtmlStructuredFlattenedWriter
from safire.utils.transcorp import dry_run

__author__ = 'Jan Hajic jr.'

# This is really the variable part: block settings
vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}

tfidf_settings = {'normalize': False}

icorp_settings = {'dim': 4096,
                  'delimiter': ';'}


def main(args):
    logging.info('Executing tfidf_vs_plain.py...')
    _start_time = time.clock()

    loader = MultimodalShardedDatasetLoader(root=args.root,
                                            name=args.name)

    logging.info('Build vtcorp.')
    vtlist = os.path.join(loader.root,
                          loader.layout.vtlist)
    vtcorp = VTextCorpus(input=vtlist, input_root=args.root, **vtcorp_settings)

    logging.info('Build tfidf.')
    tfidf = TfidfModel(vtcorp, **tfidf_settings)
    tfidf_pipeline = tfidf[vtcorp]

    logging.info('Build normalized tfidf.')
    tfidf_n = copy.deepcopy(tfidf)
    tfidf_n.normalize = True
    tfidf_n_pipeline = tfidf_n[vtcorp]

    logging.info('Build icorp.')
    image_file = os.path.join(loader.root,
                              loader.layout.image_vectors)
    icorp = ImagenetCorpus(image_file, **icorp_settings)

    logging.info('Create composite (aligned).')
    composite = CompositeCorpus((vtcorp, tfidf_pipeline, tfidf_n_pipeline),
                                names=('vtcorp', 'tfidf', 'tfidf_n'),
                                aligned=True)

    logging.info('Flatten (structured, composite is aligned).')
    flatten = FlattenComposite(composite, structured=True)
    flattened = flatten[composite]

    logging.info('Create writers.')
    simple_writer = HtmlVocabularyWriter(root=loader.root,
                                         prefix=loader.layout.name + '.tfidf',
                                         top_k=100,
                                         min_freq=0.001)
    composite_writer = HtmlStructuredFlattenedWriter(root=loader.root,
                                                     writers=(simple_writer,
                                                              simple_writer,
                                                              simple_writer))
    logging.info('Create introspection.')
    introspection = IntrospectionTransformer(flattened,
                                             writer=composite_writer)
    intro_pipeline = introspection[flattened]

    logging.info('Dry-run introspection, to generate files.')
    dry_run(intro_pipeline)
    iid2intro = intro_pipeline.obj.iid2introspection_filename
    firstfile = iid2intro[sorted(iid2intro.keys())[0]]

    logging.info('Opening introspection file.')
    webbrowser.open(firstfile)

    _end_time = time.clock()
    logging.info('Exiting tfidf_vs_plain.py. Total time: {0} s'
                 ''.format(_end_time - _start_time))


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
