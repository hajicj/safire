#!/usr/bin/env python
"""
``clean.py`` is a script that restores a dataset directory to its original
state: no corpora, no models, no learners, no indexes.
"""
import argparse
import logging

from safire.data.layouts import clean_data_root


__author__ = 'Jan Hajic jr.'

########################################################################


def main(args):
    logging.info('Executing clean.py...')

    clean_data_root(args.root)

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
