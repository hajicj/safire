#!c:\users\lenovo\canopy\user\scripts\python.exe
"""
generate_corpora.py : automatizes corpus generation in a new dataset.

Implements/expects basic naming conventions for infixes.

A "grid search" approach over multiple parameters is taken. Each parameter is
called an *axis* of the parameter space:

* ``pos`` - Part of Speech filtering: None, NADV, NAV, N

* ``top_k`` - frequency filtering: None, 20010/10, 10010/10, 5010/10 (always
  leaves out the 10 most frequent tokens)

* ``pfilter`` - positional filtering: None, 5, 0.5

* ``pfilter_fullfreq`` - positional filtering full freqs: None, True

* ``tfidf`` - TF-IDF transformation: None, Yes

By default, all axes are on. You can switch them off by providing the
``--defaults axisname`` option (``--defaults pfilter tfidf``, etc.).

Corpora that are found will not be overwritten, unless the ``--overwrite``
option is given.

Infix naming
==============

The infixes are assembled in the following order::

  .pos.freq.pfilter.pfilter_fullfreq.tfidf

The infixes for individual axes (if the given axis is not applied, no infix
is generated; not even the dot):

======================= ==============================================
axis                    infixes
======================= ==============================================
Part of Speech          ``.NADV``, ``.NAV``, ``.N``
Frequency filter        ``.top20010``, ``.top10010``, ``.top5010``
Positional filter       ``.pf5``, ``.pf05``
``pfilter_fullfreq``    ``.pFF``
tfidf                   ``.tfidf``
======================= ==============================================

"""
import argparse
import collections
import logging
import itertools
import dataset2corpus
from safire.data.loaders import MultimodalShardedDatasetLoader

__author__ = 'Jan Hajic jr.'

# This is what we are iterating over.
axes = collections.OrderedDict([
    ('pos', [None, 'NADV', 'NAV', 'NA', 'N']),
    ('top_k', [None, 20010, 10010, 5010, 1010]),
    ('pfilter', [None, 0.1, 0.2, 0.3, 0.5, 5, 10]),
    ('pfilter_fullfreq', [False, True]),
    ('tfidf', [True, False]),
    ('post_tfidf', [False]) # Turns out this is a bad idea.
])

label_prefixes = collections.OrderedDict([
    ('pos', ''),
    ('top_k', 'top'),
    ('pfilter', 'pf'),
    ('pfilter_fullfreq', ''),
    ('tfidf', ''),
    ('post_tfidf', '')
])

label_substitutions = collections.OrderedDict([
    ('pfilter_fullfreq', 'pff'),
    ('post_tfidf', 'ptfidf')
])

##############################################################################

def generate_args_string(*arg_dicts, **kwargs):
    """arg_dicts are dictionaries with argname : arg value pairs,
    args are named arguments."""

    # Collect args
    args = []
    for arg_dict in arg_dicts:
        args.extend(list(arg_dict.items()))

    args.extend(list(kwargs.items()))

    # Create the string forms
    args_as_strings = []
    for argname, argvalue in args:

        if argvalue is None:
            continue

        # Flags
        if isinstance(argvalue, bool):
            args_as_strings.append('--' + argname)
            continue

        # Non-flags
        args_as_strings.append('--' + argname)

        # String args are iterable, but they are single arguments...
        if isinstance(argvalue, str):
            args_as_strings.append(argvalue)
            continue

        try:
            # nargs='+'?
            iter(argvalue)
            for aval in argvalue:
                args_as_strings.append(str(aval))
        except TypeError:
            # Single-value args
            args_as_strings.append(str(argvalue))

    return args_as_strings


def generate_d2c_label(*argdicts):
    """Generates a label for the given configuration of args."""

    kwargs = {}
    for a in argdicts:
        for i in a.items():
            kwargs[i[0]] = i[1]

    #print 'label kwargs: %s' % str(kwargs)

    label_fields = ['']

    for axis in axes:
        #print 'Axis: %s' % axis
        avalue = kwargs[axis]
        prefix = label_prefixes[axis]
        if avalue:
            if isinstance(avalue, bool):
                if axis in label_substitutions:
                    axis = label_substitutions[axis]
                label_fields.append(prefix + str(axis)) # Flags
            else:
                label_fields.append(prefix + str(avalue))

    label = '.'.join(label_fields)
    return label


def main(args):

    logging.info('Executing generate_corpora.py...')

    default_values = {}
    default_axes = []
    if args.defaults:
        default_axes = args.defaults
        default_values = { a : axes[a][0] for a in default_axes }

    iter_axes = [ a for a in axes if a not in default_axes ]
    iter_values = [ axes[a] for a in iter_axes ]

    # Prepare configurations

    configurations = []

    for walk_config in itertools.product(*iter_values):
        config_dict = { iter_axes[i] : walk_config[i]
                        for i in xrange(len(walk_config)) }

        #print config_dict # DEBUG

        d2c_parser = dataset2corpus.build_argument_parser()
        d2c_label = generate_d2c_label(default_values, config_dict)

        #print 'Label: %s' % d2c_label

        d2c_args_input = generate_args_string(default_values, config_dict,
                                              discard_top=10, label=d2c_label,
                                              root=args.root, name=args.name)
                                              #verbose=args.verbose,
                                              #debug=args.debug)

        #print 'Args string: %s' % str(d2c_args_input)

        d2c_args = d2c_parser.parse_args(d2c_args_input)

        #print d2c_args # DEBUG

        configurations.append((d2c_label, d2c_args))

    logging.info('Total: %d configurations' % len(configurations))

    # Generate corpora
    dloader = MultimodalShardedDatasetLoader(args.root, args.name)

    for label, d2c_args in configurations:

        if args.clear:
            d2c_args.clear = True
            if not dloader.has_text_corpora(label):
                logging.info('Clearing: corpus %s doesn\'t exist, skipping.' % label)
                print 'Clearing: corpus %s doesn\'t exist, skipping.' % label
                continue
            if args.dry_run:
                logging.info('Would clear corpus %s.' % label)
                print 'Would clear corpus %s' % label
            else:
                logging.info('Clearing corpus %s.' % label)
                print 'Clearing corpus %s' % label
                dataset2corpus.main(d2c_args)
            continue

        if dloader.has_text_corpora(label):
            if not args.overwrite:
                logging.info('Corpus %s exists, skipping.' % label)
                print 'Corpus %s exists, skipping.' % label
            else:
                logging.info('Would overwrite existing corpus %s' % label)
                print 'Would overwrite existing corpus %s' % label
        else:
            if args.dry_run:
                logging.info('Would generate corpus %s' % label)
                print 'Would generate corpus %s' % label
            else:
                logging.info('Generating corpus %s' % label)
                dataset2corpus.main(d2c_args)



    logging.info('Exiting generate_corpora.py.')


def build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                         ' the directory which is the root of a dataset.'+
                         ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the'+
                        ' Loader. Has to correspond to the *.vtlist file name.')

    parser.add_argument('-d', '--defaults', nargs='+',
                        help='Specify which axes should use their default '
                             'values.')
    parser.add_argument('-y', '--dry_run', action='store_true',
                        help='If set, doesn\'t actually create the files, only '
                             'logs which corpora would be created.')
    parser.add_argument('-c', '--clear', action='store_true',
                        help='If set, attempts to removes the given corpora '
                             'instead of generating them.')

    parser.add_argument('--overwrite', action='store_true',
                        help='If set, will re-build existing corpora as well.')

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
