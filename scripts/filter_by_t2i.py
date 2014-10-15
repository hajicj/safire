#!/usr/bin/env python
"""
``filter_by_t2i.py`` is a utility script that speeds up the repetitive task of
filtering the dataset "master files" (vtlist, vtext-image-map, img.ids2files
and im.ftrs.csv) according to a given vtext-image-map.

No safire modules are used.

The script will output the master files so that they contain only the texts and
images described in the given vtext-image-map file.

TAKE CARE WHEN RUNNING THIS SCRIPT SO THAT NOTHING GETS OVERWRITTEN!!!!
(Except for re-generating files from a new vtext-image map for the given label.)
"""
import argparse
import logging
import os

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing filter_by_t2i.py...')

    # Preparing the filtering
    input_t2i = {}
    with open(args.input_t2i) as i_t2i_h:
        for line in i_t2i_h:
            t, i = line.strip().split()
            if t in input_t2i:
                input_t2i[t].append(i)
            else:
                input_t2i[t] = [i]

    retain_texts = frozenset(input_t2i.keys())
    retain_iids = frozenset([ iid
                              for iids in input_t2i.values()
                              for iid in iids ])

    logging.debug('After building t2i: first items - %s' % input_t2i.items()[:10])

    # Process t2i map
    if not args.no_t2i:
        source_t2i_file = os.path.join(args.root,
                                       args.name + '.vtext-image-map.csv')
        target_t2i_file = os.path.join(args.root,
                                       args.output_prefix + args.name + '.vtext-image-map.csv')
        logging.info('Processing t2i map: %s ---> %s' % (source_t2i_file,
                                                         target_t2i_file))

        with open(source_t2i_file) as s_t2i_h:
            with open(target_t2i_file, 'w') as t_t2i_h:
                for line in s_t2i_h:
                    t, i = line.strip().split()
                    if t in retain_texts and i in retain_iids:
                        t_t2i_h.write(line)
                    elif t in retain_texts:
                        logging.debug('Have text %s, but not image %s' % (t, i))

    # Process vtlist
    if not args.no_vtlist:
        source_vtlist = os.path.join(args.root,
                                     args.name + '.vtlist')
        target_vtlist = os.path.join(args.root,
                                     args.output_prefix + args.name + '.vtlist')
        logging.info('Processing vtlist: %s ---> %s' % (source_vtlist, target_vtlist))

        with open(source_vtlist) as s_vtl_h:
            with open(target_vtlist, 'w') as t_vtl_h:
                for line in s_vtl_h:
                    if line.strip() in retain_texts:
                        logging.debug('Retaining: %s' % line.strip())
                        t_vtl_h.write(line)


    # Process img.ids2files
    if not args.no_i2f:
        source_i2f_file = os.path.join(args.root,
                                       args.name + '.img.ids2files.csv')
        target_i2f_file = os.path.join(args.root,
                                       args.output_prefix + args.name + '.img.ids2files.csv')
        logging.info('Processing i2f map: %s ---> %s' % (source_i2f_file,
                                                         target_i2f_file))

        with open(source_i2f_file) as s_i2f_h:
            with open(target_i2f_file, 'w') as t_i2f_h:
                for line in s_i2f_h:
                    i, f = line.strip().split()
                    if i in retain_iids:
                        t_i2f_h.write(line)

    # Proces im.ftrs.csv
    if not args.no_imftrs:
        source_imftrs_file = os.path.join(args.root,
                                          args.name + '.im.ftrs.csv')
        target_imftrs_file = os.path.join(args.root,
                            args.output_prefix + args.name + '.im.ftrs.csv')
        logging.info('Processing imftrs: %s ---> %s' % (source_imftrs_file,
                                                        target_imftrs_file))
        with open(source_imftrs_file) as s_imftrs_h:
            with open(target_imftrs_file, 'w') as t_imftrs_h:
                for line in s_imftrs_h:
                    iid = line.split('\t', 1)[0]
                    if iid in retain_iids:
                        t_imftrs_h.write(line)

    logging.info('Exiting filter_by_t2i.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                                            ' the directory which is the root of a dataset.' +
                                            ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', default='',
                        help='The master files with this name will be read.')
    parser.add_argument('-i', '--input_t2i', required=True,
                        help='The vtext-image map according to which the master'
                             ' files should be filtered.')
    parser.add_argument('-o', '--output_prefix', required=True,
                        help='The prefix with which the created files will be'
                             ' prepended. HAS to be provided.')

    parser.add_argument('--no_t2i', action='store_true',
                        help='Will not process the vtext-image-map.csv file.')
    parser.add_argument('--no_vtlist', action='store_true',
                        help='Will not process the vtlist file.')
    parser.add_argument('--no_i2f', action='store_true',
                        help='Will not process the img.ids2files.csv file.')
    parser.add_argument('--no_imftrs', action='store_true',
                        help='Will not process the im.ftrs.csv file.')

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
