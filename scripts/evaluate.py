#!/usr/bin/env python
"""
``evaluate.py`` is a script that evaluates the retrieval performance of two
systems.

It takes as input two retrieval files: the True file and the Predicted file.
A retrieval file can be in one of three formats:

* ``result`` -- three columns: text, images and similarities. Images and
  similarities are ``;``-delimited multi-value.

* ``t2i`` -- like ``result``, but without the similarities column

* ``vtext-image-map`` -- two columns, but only one image per column (if more
  images for one texts, the text is on multiple lines).

Each retrieval file has two columns, with the first column being the text ID
and the second column the ';'-separated retrieved image IDs.

Computes first of all all-N accuracy: the proportion of texts for which at least
one of the retrieved items is at least one of the original items.

"""
import argparse
import logging

__author__ = 'Jan Hajic jr.'


def parse_t2i_csv(handle, image_delimiter=';'):
    """Returns a dictionary. Keys are texts, values are lists images.
    Raises a ValueError if a text is present more than once."""
    output = {}
    for line in handle:
        text, images = line.strip().split()
        if text in output:
            raise ValueError('Text %s already in output!' % text)
        output[text] = images.split(image_delimiter)

    return output

def parse_results_csv(handle, image_delimiter=';'):
    """Returns a dictionary. Keys are texts, values are lists images.
    Raises a ValueError if a text is present more than once. Expects
    the output generated by ``run.py``."""
    output = {}
    for line in handle:
        text, images, similarities = line.strip().split()
        if text in output:
            raise ValueError('Text %s already in output!' % text)
        output[text] = images.split(image_delimiter)

    return output

def parse_vtext_image_map(handle):
    """Like parse_t2i_csv, but from vtext-image-map, not eval results file."""
    output = {}
    for line in handle:
        text, image = line.strip().split()
        if text in output:
            output[text].append(image)
        else:
            output[text] = [image]
    return output

def main(args):
    logging.info('Executing evaluate.py...')

    with open(args.true) as true_handle:
        if args.t_asmap:
            true = parse_vtext_image_map(true_handle)
        else:
            if not args.t_nosim:
                true = parse_results_csv(true_handle)
            else:
                true = parse_t2i_csv(true_handle)

    with open(args.prediction) as prediction_handle:
        if args.p_asmap:
            prediction = parse_vtext_image_map(prediction_handle)
        else:
            if not args.p_nosim:
                prediction = parse_results_csv(prediction_handle)
            else:
                prediction = parse_t2i_csv(prediction_handle)

    # Evaluate all-N accuracy
    hits = []
    total_texts_skipped = 0
    for t in prediction:
        # Sanity check
        if t not in true:
            if args.ignore_missing_texts:
                logging.debug('Missing text in true: %s' % t)
                total_texts_skipped += 1
                continue
            else:
                raise ValueError('Cannot measure performance, missing text in true: %s' % t)

        predicted = prediction[t]
        to_hit = set(true[t])
        has_hit = False
        for p in predicted:
            if p in to_hit:
                has_hit = True
                break
        if has_hit:
            hits.append(1.0)
        else:
            hits.append(0.0)

    logging.info('Evaluation: missing texts - %d out of %d' % (total_texts_skipped, len(prediction)))
    accuracy = sum(hits) / float(len(hits))
    print 'Accuracy: %.3f' % accuracy

    logging.info('Exiting evaluate.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store',
                        help='The file with the True text-image mapping.')
    parser.add_argument('--t_asmap', action='store_true',
                        help='Will parse the True input as a vtext-image-map, '
                             'not an eval map.')
    parser.add_argument('--t_nosim', action='store_true',
                        help='If set, will expect a third column of'
                             'similarities in the true file.')

    parser.add_argument('-p', '--prediction', action='store',
                        help='The file with the predicted text-image mapping.')
    parser.add_argument('--p_asmap', action='store_true',
                        help='Will parse the predicted input as a vtext-image-map, '
                             'not an eval map.')
    parser.add_argument('--p_nosim', action='store_true',
                        help='If set, will expect a third column of'
                             'similarities in the predicted file.')

    parser.add_argument('--ignore_missing_texts', action='store_true',
                        help='If set, will simply not count any predicted text'
                             ' that is not in True.')


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
