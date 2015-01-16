#!c:\users\lenovo\canopy\user\scripts\python.exe
import argparse
from copy import deepcopy
import itertools
import logging
import operator
import os
import random
import webbrowser
from safire.data.text_browser import TextBrowser
import safire.utils
from safire.data.image_browser import ImageBrowser
from safire.data.loaders import MultimodalDatasetLoader, IndexLoader, \
    ModelLoader, MultimodalShardedDatasetLoader

__author__ = 'Jan Hajic jr.'

##############################################################################

def build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                        ' the directory which is the root of a dataset.' +
                        ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the' +
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-l', '--labels', nargs='+',
                        help='The corpus labels.')
    parser.add_argument('--first_n_sentences', type=int, default=10,
                        help='Display only this many sentences from the '
                             'beginning of a text.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    return parser


def print_interactive_help():
    """Prints the help message for interactive mode."""
    print 'Image index explorer interactive mode help\n' \
          '==========================================\n' \
          '\n' \
          'Commands:\n' \
          '  h        ... help\n' \
          '  c N      ... compare representations for N-th document in vtlist\n' \
          '  q|e      ... exit (will ask for confirmation)\n' \
          '\n' \
          'On the \'c\' command, will show two columns of most similar images\n' \
          'with the similarities. Will show query image on top.'


def run_interactive(vtlist, raw_corpus, raw_browser,
                    corpora, browsers, labels):

    exit_commands = frozenset(['q', 'e'])
    compare_commands = frozenset(['c'])
    help_commands = frozenset(['h'])

    # Starting settings
    highest_scoring = 10

    exit_interactive = False
    while not exit_interactive:

        # Parse command
        user_input = raw_input('--> ')
        split_input = user_input.split(' ', 1)
        if len(split_input) > 1:
            command, options = split_input
        else:
            command = split_input[0]
            options = None


        # Execute command
        if command in help_commands:
            print_interactive_help()
            continue

        elif command in compare_commands:

            N = int(options)
            text = raw_browser.get_text(N)

            btext = text + '\n[end of text]\n'
            #print btext

            representations = []
            for label, browser in zip(labels, browsers):
                representation = browser.get_word_representation(N,
                                            highest_scoring=highest_scoring)
                # Add headers to representation
                representation = [('model', label), ('-----', '-----')] \
                                 + representation
                representations.append(representation)

            all_representations = list(itertools.chain(*representations))

            # ???
            formatted_repr = raw_browser.format_representation(
                all_representations, n_cols=len(representations))

            output = text + '\n\n' + formatted_repr
            raw_browser.text_to_window(output)


        elif command in exit_commands:
            confirmation = raw_input('-[y/n]-> ')
            if confirmation in exit_commands or confirmation == '' \
                    or confirmation == 'y':
                exit_interactive = True
                continue
        else:
            print 'Invalid command %s' % command

def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    vtlist_file = dloader.layout.vtlist
    with open(os.path.join(args.root, vtlist_file)) as vtlist_handle:
        vtlist = [ l.strip() for l in vtlist_handle ]

    # The corpus and browser used for displaying the raw texts
    raw_text_corpus = dloader.load_text_corpus()
    raw_text_browser = TextBrowser(args.root, raw_text_corpus,
                                   first_n_sentences=args.first_n_sentences)

    # The browsers from which we pull representations
    text_corpora = [ dloader.load_text_corpus(label) for label in args.labels]
    text_browsers = [ TextBrowser(args.root, corpus,
                                  first_n_sentences=args.first_n_sentences)
                      for corpus in text_corpora ]

    run_interactive(vtlist, raw_text_corpus, raw_text_browser,
                    text_corpora, text_browsers, args.labels)

    # Explicit delete
    del raw_text_browser
    for browser in text_browsers:
        del browser


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