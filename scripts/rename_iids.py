#!/usr/bin/env python
"""
``rename_iids.py`` is a utility that renames IIDs in a vtext-image-map
according to an IID --> representant IID mapping. Use to standardize duplicate
IIDs.
"""
import argparse
import logging

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing rename_iids.py...')

    logging.info('Loading iid --> representant map.')
    iid_map = {}
    with open(args.iid_map) as iid_map_h:
        for line in iid_map_h:
            i, r = line.strip().split()
            iid_map[i] = r

    logging.info('Processing t2i map: %s --> %s' % (args.input_t2i,
                                                    args.output_t2i))
    with open(args.input_t2i) as input_h:
        with open(args.output_t2i, 'w') as output_h:
            for line in input_h:
                t, i = line.strip().split()
                r = iid_map[i]
                output_h.write('\t'.join([t,r]) + '\n')

    logging.info('Exiting rename_iids.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-i', '--input_t2i',
                        help='The input vtext-image-map.csv file.')
    parser.add_argument('-o', '--output_t2i',
                        help='The vtext-image-map.csv file to which the output'
                             ' will be written.')
    parser.add_argument('-m', '--iid_map',
                        help='The file that contains the IID --> representant '
                             'mapping.')

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
