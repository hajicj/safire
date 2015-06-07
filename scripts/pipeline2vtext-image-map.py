#!/usr/bin/env python
"""
``pipeline2vtext-image-map.py`` is a script that exports the results of a given
retrieval pipeline into a CSV file. To evaluate the results, the generated
vtext-image-map should then be passed to the evaluate.py script.

"""
import argparse
import logging
import time
from safire.utils.config import build_pipeline
from safire.utils.transcorp import get_id2doc_obj, id2doc

__author__ = 'Jan Hajic jr.'


def main(args):
    logging.info('Executing pipeline2vtext-image-map.py...')
    _start_time = time.clock()

    pipeline = build_pipeline(args.configuration, args.pipeline)

    # Extract image id2doc mapping. This is tricky: we need to get
    # the id2doc mapping from the corpus that was used to build the
    # image similarity index.
    #
    # You can either supply the data manually through the --img_id2doc_source
    # option, or the script will try to auto-resolve:
    #  - find the last SimilarityTransformer in the pipeline,
    #  - find its index,
    #  - find its corpus,
    #  - get id2doc obj from its corpus
    # The auto-resolve is not yet implemented, though.
    img_id2doc = get_id2doc_obj(pipeline.get_object(args.img_id2doc_source))

    # Now run through the retrieval results and output t2i
    logging.info('Starting retrieval and vtext-image map generation...')
    _ret_start_time = time.clock()
    with open(args.output, 'w') as output_handle:
        for text_iid, ret in pipeline:
            # Text id2doc mapping doesn't have to be initialized prior to
            # extracting retrieval results.
            text_doc = id2doc(pipeline, text_iid)
            img_docs = [img_id2doc[iid] for iid, sim in ret]
            for img_doc in img_docs:
                output_handle.write('\t'.join([text_doc, img_doc] + '\n'))

    logging.info('Retrieval and vtext-image map generation done in {0:.2f} s'
                 ''.format(time.clock() - _ret_start_time))

    _end_time = time.clock()
    logging.info(
        'Exiting pipeline2vtext-image-map.py. Total time: {0:.2f} s'.format(
            _end_time - _start_time))


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-c', '--configuration',
                        help='Get pipeline with t2i results from this '
                             'configuration file.')
    parser.add_argument('-p', '--pipeline',
                        help='Extract this blockname as the t2i pipeline '
                             'output.')
    parser.add_argument('-o', '--output',
                        help='The file to which the t2i map should be written.')

    parser.add_argument('--img_id2doc_source',
                        help='Get the image id2doc mapping from this pipeline'
                             ' object. (Should be the corpus from which the'
                             ' similarity index used for retrieval was built.)')

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
