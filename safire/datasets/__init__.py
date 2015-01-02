#!/usr/bin/env python
"""
``__init__.py`` is a script that ...
"""
import argparse
import logging

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing __init__.py...')

    pass

    logging.info('Exiting __init__.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

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
