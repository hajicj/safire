#!c:\users\lenovo\canopy\user\scripts\python.exe
import argparse
import itertools
import logging
import matplotlib.pyplot as plt
import operator
import os
import random
import numpy
from safire.data.text_browser import TextBrowser
import safire.utils
import safire.utils.matutils as matutils
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
    parser.add_argument('-i1', '--img_label_1',
                        help='Label of the first corpus/dataset/index triplet.')
    parser.add_argument('-i2', '--img_label_2',
                        help='Label of the second corpus/dataset/index triplet.'
                             'If not given, the \'compare\' command in '
                             'interactive mode will only output images from one'
                             ' index.')
    parser.add_argument('-b', '--image_browser',
                        help='Full path to the image id-filename map for the '
                             'image browser.')
    parser.add_argument('-t', '--text_browser_label',
                        help='VTextCorpus label in the given data root. If '
                             'given, will initialize a TextBrowser to inspect'
                             ' text documents related to the images. Will look'
                             ' for the DataDirLayout default vtext-image-map.csv'
                             ' file.')

    parser.add_argument('--quantitative', action='store_true',
                        help='Instead of entering interactive mode for manual '
                             'exploration, enters interactive mode for '
                             'comparing the retrieval performance of i2 vs i1.')

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
          '  c ID     ... compare most similar for given corpus ID (number)\n' \
          '  d N      ... set browser to display N most similar images.\n' \
          '  s ID     ... show most similar for given corpus ID (number)\n' \
          '  b ID N K ... show N most similar, then each k-th most similar.\n' \
          '  p ID     ... plot the sorted similarities of query results.\n' \
          '  w        ... switch on/off displaying text together with images.\n' \
          '  r        ... switch on/off randomized image ordering.\n' \
          '  q|e      ... exit (will ask for confirmation)\n' \
          '\n' \
          'On the \'c\' command, will show two columns of most similar images\n' \
          'with the similarities. Will show query image on top.'


def run_interactive(icorp_1, idata_1, index_1, icorp_2, idata_2, index_2,
                    image_browser, text_browser=None):
    """Runs an interactive index explorer script."""

    exit_commands = frozenset(['q', 'e'])
    num_best_commands = frozenset(['d'])
    buckets_commands = frozenset(['b'])
    display_commands = frozenset(['s'])
    help_commands = frozenset(['h'])
    compare_commands = frozenset(['c'])
    plot_commands = frozenset(['p'])
    with_text_commands = frozenset(['w'])
    do_shuffle_commands = frozenset(['r'])

    default_num_best = 5

    idata_1_iter = idata_1.__iter__()
    idata_1_ctr = -1
    index_1.num_best = default_num_best

    if index_2 is not None:
        idata_2_iter = idata_2.__iter__()
        idata_2_ctr = -1
        index_2.num_best = default_num_best

    # If true, will display texts together with images.
    with_text = False

    # If true, will shuffle images on display.
    do_shuffle = False

    exit_interactive = False
    while not exit_interactive:

        user_input = raw_input('--> ')
        split_input = user_input.split(' ', 1)
        if len(split_input) > 1:
            command, options = split_input
        else:
            command = split_input[0]
            options = None

        logging.debug('Command: %s, options: %s' % (str(command), str(options)))

        # Execute command
        if command in help_commands:
            print_interactive_help()
            continue

        elif command in with_text_commands:

            if with_text:
                with_text = False
            else:
                if text_browser is None:
                    print 'Error: cannot switch on text display without a ' \
                          'text browser.'
                    continue
                else:
                    with_text = True

        elif command in do_shuffle_commands:

            do_shuffle = (not do_shuffle)

        elif command in num_best_commands:

            if not options:
                print 'Error: must supply num_best command with options.'
                continue

            new_num_best = int(options)

            index_1.num_best = new_num_best+1 # First image is always query img.

            # So that we can work in single-index mode:
            if index_2 is not None:
                index_2.num_best = new_num_best

            continue

        elif command in display_commands:

            try:
                image_no = int(options)
            except ValueError:
                image_no = icorp_1.doc2id[options]
            #image_id = icorp_1.id2doc[image_no]

            logging.info('Querying for image no. %d' % image_no)

            query = idata_1.get_sparse(image_no)

            result = index_1[query] # We want to display the first image here.

            inos = map(operator.itemgetter(0), result)

            logging.info('Inos to retrieve: %s' % str(inos))
            iids = [ icorp_1.id2doc[imno] for imno in inos ]
            logging.info('IIDs to retrieve: %s' % str(iids))

            similarities = map(operator.itemgetter(1), result)

            if do_shuffle:
                iids, similarities = safire.utils.shuffle_together(iids, similarities)

            image = image_browser.build_tiled_image(iids, similarities)

            if with_text and text_browser:

                image_id = icorp_1.id2doc[image_no]
                docids = text_browser.im2text[image_id]

                logging.debug('DocIDs: %s' % str(docids))

                text_browser.show_multiple(docids)

            image.show()

        elif command in plot_commands:

            num_best = index_1.num_best
            index_1.num_best = None

            image_no = int(options)
            #image_id = icorp_1.id2doc[image_no]

            query = idata_1.get_sparse(image_no)

            result = index_1[query] # We want to display the first image here.

            # When querying without num_best set, the similarities are returned
            # as a list.
            #similarities = map(operator.itemgetter(1), result)
            sorted_similarities = sorted(result, reverse=True)

            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.plot(sorted_similarities)
            plt.title('Sorted similarities for query image %d' % image_no)
            #plt.subplot(122)
            plt.hist(sorted_similarities, bins=100, normed=False, histtype='step',
                     orientation='horizontal', color='red')
            plt.title('Smiliarities, w. histogram')

            # Plot the image features for the given image
            features = idata_1[image_no]
            plt.subplot(122)
            plt.plot(features)
            plt.hist(list(features) * 2, bins=20, normed=False, histtype='step',
                     orientation='horizontal', color='red')
            plt.title('Activations, w. histogram')

            plt.show()

            index_1.num_best = num_best


        elif command in buckets_commands:

            image_no, N, K = map(int, options.split())
            #image_id = icorp_1.id2doc[image_no]

            query = idata_1.get_sparse(image_no)

            # For bucket inspection, we want the whole result
            nb = index_1.num_best
            index_1.num_best = None

            raw_result = index_1[query]

            index_1.num_best = nb # Back to previous state...

            result = zip(range(len(raw_result)), raw_result)
            sorted_results = sorted(result, key=operator.itemgetter(1),
                                    reverse=True)

            top_results = sorted_results[:N]

            sample_step = (len(sorted_results) - (N + 1)) / K
            sample_ids = range(len(sorted_results) - 1, N, -1 * sample_step)
            sample_ids.reverse()

            sample_results = [ sorted_results[sid] for sid in sample_ids ]

            all_results = top_results + sample_results
            all_imnos, all_similarities = map(list, zip(*all_results))
            all_iids = [icorp_1.id2doc[imno] for imno in all_imnos]

            if do_shuffle:
                all_iids, all_similarities = safire.utils.shuffle_together(all_iids, all_similarities)

            image = image_browser.build_tiled_image(all_iids, all_similarities)

            if with_text and text_browser:

                image_id = icorp_1.id2doc[image_no]
                docids = text_browser.im2text[image_id]

                logging.debug('DocIDs: %s' % str(docids))

                text_browser.show_multiple(docids)

            image.show()

        elif command in compare_commands:

            if not options:
                print 'Error: must supply compare command with options.'
                continue

            if index_2 is None:
                print 'Error: cannot compare with only one index supplied.'
                continue

            image_no = int(options)
            image_id = icorp_1.id2doc[image_no]

            query_1 = idata_1.get_sparse(image_no)
            query_2 = idata_2.get_sparse(image_no)

            result_1 = index_1[query_1][1:]
            result_2 = index_2[query_2][1:]

            inos_1 = list(itertools.imap(operator.itemgetter(0), result_1))
            iids_1 = [icorp_1.id2doc[imno] for imno in inos_1]
            similarities_1 = list(itertools.imap(operator.itemgetter(1), result_1))
            images_1 = image_browser.load_images(iids_1, similarities_1)

            inos_2 = list(itertools.imap(operator.itemgetter(0), result_2))
            iids_2 = [icorp_2.id2doc[imno] for imno in inos_2]
            similarities_2 = list(itertools.imap(operator.itemgetter(1), result_2))
            images_2 = image_browser.load_images(iids_2, similarities_2)

            comparison = safire.utils.image_comparison_report(images_1, images_2)

            query_image = image_browser.load_image(image_id)
            comparison_w_header = safire.utils.add_header_image(comparison,
                                                                query_image)

            comparison_w_header.show()

            continue

        elif command in exit_commands:
            confirmation = raw_input('-[y/n]-> ')
            if confirmation in exit_commands or confirmation == '':
                exit_interactive = True
                continue

        else:
            print 'Invalid command %s' % command


def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)
    iloader = IndexLoader(args.root, args.name)

    logging.info('Loading first image corpus with label %s' % args.img_label_1)

    icorp_1 = safire.utils.transcorp.bottom_corpus(dloader.load_image_corpus(args.img_label_1))
    idata_1 = dloader.load_img(args.img_label_1)
    index_1 = iloader.load_index(args.img_label_1)

    logging.debug('icorp_1.id2doc length: %d' % len(icorp_1.id2doc))

    icorp_2 = None
    idata_2 = None
    index_2 = None

    if args.img_label_2:

        logging.info('Loading second image corpus with label %s' % args.img_label_2)

        icorp_2 = safire.utils.transcorp.bottom_corpus(dloader.load_image_corpus(args.img_label_2))
        idata_2 = dloader.load_img(args.img_label_2)
        index_2 = iloader.load_index(args.img_label_2)

        logging.debug('icorp_2.id2doc length: %d' % len(icorp_2.id2doc))

        if len(index_1) != len(index_2):
            logging.warn('Using indexes of unequal length: i1 %d, i2 %d' % (
                len(index_1), len(index_2)
                                                                           ))
        if len(idata_1) != len(idata_2):
            logging.warn('Using datasets of unequal length: i1 %d, i2 %d' % (
                len(idata_1), len(idata_2)
                                                                           ))
        if len(icorp_1) != len(icorp_2):
            logging.warn('Using corpora of unequal length: i1 %d, i2 %d' % (
                len(icorp_1), len(icorp_2)
                                                                           ))


    else:

        logging.warn('Running in single-index mode; comparison functionality not available.')

    logging.info('Loading image browser from file %s' % args.image_browser)

    # Load image browser
    image_browser = None

    if args.image_browser:
        if not os.path.isfile(args.image_browser):
            rel_browser_map = os.path.join(args.root, args.image_browser)
            if os.path.isfile(rel_browser_map):
                logging.warning('Using image browser map relative to input '
                'root, are you sure? (%s)' % rel_browser_map)
                args.image_browser = rel_browser_map
            else:
                raise ValueError('Image browser map not found!')
        with open(args.image_browser) as image_browser_handle:
            image_browser = ImageBrowser(args.root, image_browser_handle)

    print 'Image browser: %s' % image_browser

    text_browser = None

    # Load text browser and text-image map
    if args.text_browser_label:
        vtcorp = dloader.load_text_corpus(args.text_browser_label)

        # Populate text-image maps
        text2im_file = os.path.join(args.root, dloader.layout.textdoc2imdoc)
        text2im = {}
        im2text = {}
        with open(text2im_file) as t2i_handle:

            for line in t2i_handle:
                text_id, img_id = line.strip().split()
                if text_id in text2im:
                    text2im[text_id].append(img_id)
                else:
                    text2im[text_id] = [img_id]

                if img_id in im2text:
                    im2text[img_id].append(text_id)
                else:
                    im2text[img_id] = [text_id]

        text_browser = TextBrowser(args.root, vtcorp,
                                   text2im=text2im, im2text=im2text,
                                   first_n_sentences=10)

    if args.quantitative:
        n_samples = 1000
        num_best = 10

        logging.info('Performing quantitative evaluation of i2 predicting i1 '
                     'with 1000 samples, 10 best')

        choose_from = range(min(len(index_1), len(index_2)))
        random.shuffle(choose_from)
        sample_imnos = choose_from[:n_samples]

        index_1.num_best = num_best
        index_2.num_best = num_best

        qr_1 = [ index_1[idata_1.get_sparse(imno)] for imno in sample_imnos ]
        qr_2 = [ index_2[idata_2.get_sparse(imno)] for imno in sample_imnos ]
        qresults = zip(qr_1, qr_2)

        precisions = [ matutils.precision(qr2, qr1) for qr1, qr2 in qresults ]
        recalls = [ matutils.recall(qr2, qr1) for qr1, qr2 in qresults ]
        f_scores = [ matutils.f_score(qr2, qr1) for qr1, qr2 in qresults ]

        avg_prec = numpy.average(precisions)
        avg_rec = numpy.average(recalls)
        avg_fsc = numpy.average(f_scores)

        print 'Averages:\n\tPrec: %.3f\n\t Rec: %.3f\n\tF.sc: %.3f' % (
            avg_prec, avg_rec, avg_fsc
        )

        return

    else:
        # Run interactive part
        run_interactive(icorp_1, idata_1, index_1,
                        icorp_2, idata_2, index_2,
                        image_browser, text_browser)


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