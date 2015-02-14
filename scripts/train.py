#!/usr/bin/env python
"""
``train.py`` is a script that builds a processing pipeline.

Intended as a substitute for dataset2corpus.py plus pretrain.py. Implements
basic pipeline building.
"""
import argparse
import logging
from safire.data.layouts import DataDirLayout
import safire

__author__ = 'Jan Hajic jr.'

sample_config = '''
# This is a sample YAML string that configures a pipeline on texst data.
# The script is, however, agnostic to how the configuration is passed -- all
# it uses is the Argparse.Namespace object, which can be filled from the
# command line. This is to facilitate usage with external experiment managers
# such as UFAL's eman that work with generic scripts and variables.
#
---
# Where is our dataset (and Waldo)?
&root root: test
# This is a special value that will point us to safire test data

name: test-data
# The name of the dataset within the data root.

&vtlist vtlist: auto
# Again, a special value that derives the corpus automatically.

# Loading block. Our sample configuration loads a new VTextCorpus.
block :
    name : read-vtext
    type : safire.data.vtextcorpus.VTextCorpus
    args :
        input : *vtlist
...
'''

'''
# class ConfigBlock(object):
#     """Defines a configuration block. The blocks are the blueprint for the
#     pipeline. The block also defines a build() method, which recursively builds
#     the 'apply_to' blocks and then adds itself.
#     """
#     def __init__(self, name, apply_to, cls, **kwargs):
#         """
#         :param name: The name of this block.
#
#         :param apply_to: Another config block to which it should be applied.
#
#         :param cls: The transformer class that should act as the block's obj.
#
#         :param kwargs: Initialization arguments for the given class.
#
#         :return:
#         """
#         self.apply_to = apply_to
#         self.name = name
#         self.cls = cls,
#         for arg in kwargs:
#             self.__setattr__(arg, kwargs[arg])
#
# layout = DataDirLayout('test-data')
# root = safire.get_test_data_root()
#
# blocks = {
#     'read-vtext': read_vtext,
#     'tfidf': tfidf,
#     'freqfilter': freqfilter,
#     'tanh': tanh_text,
#     'serialize_text': serialize_text,
# }
#
# configuration = {
#     'root':   root,
#     'layout': layout,
#     'blocks': blocks,
# }
'''


def main(args):
    logging.info('Executing train.py...')

    # Workflow:

    configuration = parse_configuration(args)
    # The configuration specifies which blocks the pipeline should be built
    # from.

    pipeline = load_pipeline(configuration)
    # If no previously built pipeline is given, we expect the first block
    # to read data. (VTextCorpus, ImagenetCorpus, etc.)
    # (Shouldn't loading be done by the first block?)

    for block in configuration['blocks']:
        pipeline = apply_block(pipeline, block)

    pipeline.save()
    # Saves the pipeline, not the data! Saving the data is a Serialization
    # op.
    #
    # Actually, even similarity retrieval can be thought of as a pipeline
    # block: a transformation from the space of the "query features" to the
    # pairwise similarity space between possible responses. "Training" the
    # transformation means building the similarity index from the database
    # of responses, applying the block means that inputs to the block will
    # be interpreted as queries.

    logging.info('Exiting train.py.')


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
