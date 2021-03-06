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
vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}

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
    vtcorp = VTextCorpus(input=vtlist, input_root=args.root, **vtcorp_settings)

    logging.info('Build tfidf.')
    tfidf = TfidfModel(vtcorp, **tfidf_settings)
    tfidf_pipeline = tfidf[vtcorp]

    logging.info('Build normalized tfidf.')
    tfidf_n = copy.deepcopy(tfidf)
    tfidf_n.normalize = True
    tfidf_n_pipeline = tfidf_n[vtcorp]

    logging.info('Create t_composite (aligned).')
    t_composite = CompositeCorpus((vtcorp, tfidf_pipeline, tfidf_n_pipeline),
                                  names=('vtcorp', 'tfidf', 'tfidf_n'),
                                  aligned=True)

    logging.info('Flatten (structured, t_composite is aligned).')
    flatten = FlattenComposite(t_composite, structured=True)
    t_flattened = flatten[t_composite]

    logging.info('Create writers.')
    simple_writer = HtmlVocabularyWriter(root=loader.root,
                                         prefix=loader.layout.name + '.tfidf',
                                         top_k=100,
                                         min_freq=0.001)
    composite_writer = HtmlStructuredFlattenedWriter(root=loader.root,
                                                     writers=(simple_writer,
                                                              simple_writer,
                                                              simple_writer))

    logging.info('Build icorp.')
    image_file = os.path.join(loader.root,
                              loader.layout.image_vectors)
    icorp = ImagenetCorpus(image_file, **icorp_settings)
    i_ser_target = loader.pipeline_serialization_target('.img.mm_tfidf_vs_plain')
    i_serializer = Serializer(icorp, ShardedCorpus, i_ser_target,
                              **image_serialization_settings)
    icorp_serialized = i_serializer[icorp]

    logging.info('Get indices.')
    # Auxiliary mmdata corpus used to compute t2i (never iterated over?)
    aux_mmdata = CompositeCorpus((vtcorp, icorp_serialized),
                                 names=('txt', 'img'), aligned=False)
    t2i_file = os.path.join(loader.root,
                            loader.layout.textdoc2imdoc)
    t2i_indexes = compute_docname_flatten_mapping(aux_mmdata, t2i_file)
    logging.info('  Indices total: {0}'.format(len(t2i_indexes)))

    logging.info('Combine icorp with text corpora.')
    mm_composite = CompositeCorpus((t_flattened, icorp_serialized),
                                   names=('txt', 'img'),
                                   aligned=False)
    mm_flatten = FlattenComposite(mm_composite,
                                  indexes=t2i_indexes,
                                  structured=True)
    mm_flattened = mm_flatten[mm_composite]

    logging.info('Create mm_composite writers.')
    img_writer = HtmlImageWriter(root=os.path.join(loader.root,
                                                   loader.layout.img_dir))
    mm_writer = HtmlStructuredFlattenedWriter(root=loader.root,
                                              writers=(composite_writer,
                                                       img_writer))

    logging.info('Create introspection.')
    introspection = IntrospectionTransformer(mm_flattened,
                                             writer=mm_writer)
    intro_pipeline = introspection[mm_flattened]

    logging.info('Dry-run introspection, to generate files.')
    dry_run(intro_pipeline, max=100)
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
