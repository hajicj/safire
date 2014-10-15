#!/usr/bin/env python
"""Testing script for the loader-setup-learner scenario. Runs a miniature
experiment."""

import argparse
import cProfile
import logging
import os
import pstats
import StringIO
from gensim import corpora

from safire.data.loaders import MultimodalDatasetLoader
from safire.learning.models.logistic_regression import LogisticRegression
from safire.learning.learners.base_sgd_learner import BaseSGDLearner


def profile_run(learner, model_handle, dataset):
    
    pr = cProfile.Profile()
    pr.enable()

    learner.run(model_handle, dataset)

    pr.disable()
    s = StringIO.StringIO()
    sortby='tottime'
    ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
    ps.print_stats(.1)

    return s.getvalue()




def main(args):

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

    logging.info('Initializing loader...')
    loader = MultimodalDatasetLoader(args.root, args.name,
                                     text_serializer=serializer)

    logging.info('Loading dataset...')
    dataset = loader.load(text_infix=args.text_label, img_infix=args.img_label)
    dataset.set_mode(1)

    logging.info('Setting up model...')
    model_handle = LogisticRegression.setup(dataset, batch_size=args.batch_size)

    logging.info('Setting up learner...')
    learner = BaseSGDLearner(n_epochs=args.n_epochs, b_size=args.batch_size,
                             validation_frequency=args.validation_frequency)

    logging.info('Running learner with profiling...')
    profiler_results = profile_run(learner, model_handle, dataset)

    print profiler_results

    

def build_argument_parser():

    parser = argparse.ArgumentParser(description = __doc__, add_help=True)

    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory, passed to Loader.')
    parser.add_argument('-n', '--name', required=True,
                        help='The name passed to Loader.')
    parser.add_argument('--text_label', default=None, help='Text corpus label.')
    parser.add_argument('--img_label', default=None, help='Image corpus label.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='SGD batch size')
    parser.add_argument('-e', '--n_epochs', type=int, default=5,
                        help='Number of SGD epochs.')
    parser.add_argument('-f', '--validation_frequency', type=int, default=3,
                        help='Validation will be run once every -v batches.')

    parser.add_argument('--serializer', help='Use this gensim.corpora class'+
                        ' to load the serialized text corpora. Accepts: Mm,'+
                        ' Blei, SVMlight, Low; defaults to MmCorpus')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Will output INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Will output DEBUG messages.')

    return parser

def _set_logging(args):

    level = logging.WARN
    
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO

    logging.basicConfig(format='%(levelname)s : %(message)s', level=level)


####################################################

if __name__ == '__main__':

    parser = build_argument_parser()

    args = parser.parse_args()
    _set_logging(args)

    main(args)
