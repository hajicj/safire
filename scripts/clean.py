#!/usr/bin/env python
"""
``clean.py`` is a script that restores a dataset directory to its original
state: no corpora, no models, no learners, no indexes.
"""
import argparse
import logging
import os
import shutil
import sys

from safire.data.layouts import DataDirLayout

__author__ = 'Jan Hajic jr.'


def clean_dir(path, force=False):
    """Removes everything from the directory given by path."""
    if not force:
        print 'Are you sure you want to delete everything in %s? [y/n]' % path
        confirmation = sys.stdin.readline().strip()
        if confirmation not in ['y', 'Y', 'yes', 'Yes', 'YES']:
            print 'Aborting...'
            return
        else:
            print 'Proceeding...'

    shutil.rmtree(path)
    os.makedirs(path)

########################################################################


def main(args):
    logging.info('Executing clean.py...')

    layout = DataDirLayout(args.root)

    corpus_dir = os.path.join(layout.name, layout.corpus_dir)
    dataset_dir = os.path.join(layout.name, layout.dataset_dir)
    model_dir = os.path.join(layout.name, layout.model_dir)
    learner_dir = os.path.join(layout.name, layout.learner_dir)
    index_dir = os.path.join(layout.name, layout.index_dir)
    temp_dir = os.path.join(layout.name, layout.temp_dir)

    clean_dir(corpus_dir, force=args.force)
    clean_dir(dataset_dir, force=args.force)
    clean_dir(model_dir, force=args.force)
    clean_dir(learner_dir, force=args.force)
    clean_dir(index_dir, force=args.force)
    clean_dir(temp_dir, force=args.force)

    logging.info('Exiting clean.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory.')
    parser.add_argument('-n', '--name', required=False,
                        help='The name of the dataset to clean. Will only'
                             'clean out corpora, models etc. pertaining'
                             'to this dataset. [NOT IMPLEMENTED]')
    parser.add_argument('--retain', nargs='+',
                        help='Retain datasets/corpora etc. with the given'
                             'infixes. [NOT IMPLEMENTED]')

    parser.add_argument('-f', '--force', action='store_true',
                        help='If given, does not ask for confirmation when'
                             'deleting.')


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
