#!c:\users\lenovo\canopy\user\scripts\python.exe
"""
``run.py`` is a script that takes a configuration and runs the experiment
described therein.
"""
import argparse
import logging
import time
from safire.utils.config import ConfigParser, ConfigBuilder

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing run.py...')
    _start_time = time.clock()

    logging.info('Parsing configuration {0}...'.format(args.configuration))
    cparser = ConfigParser()
    with open(args.configuration) as config_handle:
        configuration = cparser.parse(config_handle)

    if args.clear:
        configuration._builder['clear'] = 'True'

    logging.info('Building configuration {0}: creating builder...'
                 ''.format(args.configuration))
    builder = ConfigBuilder(configuration)

    if args.draw_configuration:
        logging.info('Drawing configuration dependency graph to file {0}'
                     ''.format(args.draw_configuration))
        builder.draw_dependency_graph(args.draw_configuration)

    logging.info('Building configuration {0}: running build...'
                 ''.format(args.configuration))
    outputs = builder.build()

    _end_time = time.clock()
    logging.info(
        'Exiting run.py. Total time: {0} s'.format(_end_time - _start_time))


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-c', '--configuration', action='store',
                        help='The configuration file to work with.')
    parser.add_argument('--draw_configuration', action='store',
                        help='Draws the configuration graph to this file in'
                             ' SVG format.')
    parser.add_argument('--clear', action='store_true',
                        help='If set, will set the \'clear\' configuration'
                             ' value in the _builder section to True.')

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
