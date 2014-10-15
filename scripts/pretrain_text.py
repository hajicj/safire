#!/usr/bin/env python
"""
Trains an unsupervised layer on text data.

Use to build individual layers of the text stack. Choose your labels carefully!
There is no automated mechanism as of yet to keep track of what is what.

You can build layers on top of already trained layers by simply selecting the
labels for the previous layers. Unless ``--no_transform_corpus`` and
``--no_transform_dataset`` flags are given, the source corpus and dataset
will be at the end run through the trained model and saved with the
``--transformation_label`` label."""

__author__ = 'Jan Hajic jr.'

import argparse
import logging
import safire.learning.models as models
from safire.learning.models import DenoisingAutoencoder
from safire.data.loaders import MultimodalDatasetLoader, \
    MultimodalShardedDatasetLoader
from safire.data.loaders import ModelLoader
from safire.learning.interfaces import SafireTransformer
from safire.learning.learners import BaseSGDLearner

from train_multimodal_model import check_model_dataset_compatibility

def _build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                        ' the directory which is the root of a dataset.' +
                        ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the' +
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-t', '--text_label', help='The text corpus label from'
                                                  'which to load data.')
    parser.add_argument('-l', '--transformation_label', action='store',
                        help='The output label. This is to help distinguish ' +
                        'models made with different options. Controls saving names,'
                        'both for the model and for the transformed corpus.')
    parser.add_argument('--n_out', type=int, default=1000,
                        help='The number of model output neurons.')
    parser.add_argument('-m', '--model', default='DenoisingAutoencoder',
                        help='The name of the model class to use. Default is'
                             'DenoisingAutoencoder.')

    # Learner settings
    parser.add_argument('--batch_size', type=int, default=100,
                        help='How many data points will be used to perform one'
                             'gradient update.')
    parser.add_argument('--learning_rate', type=float, default=0.13,
                        help='How fast will the gradient update be.')
    parser.add_argument('--n_epochs', type=int, default='3',
                        help='How many training epochs should be taken.')
    parser.add_argument('--validation_frequency', type=int, default='4',
                        help='Run validation once every X training batches.')

    # Model settings
    parser.add_argument('--reconstruction',
                        help='The reconstruction measure to use. Currently '
                             'supported is \'cross-entropy\' and \'mse\'.')

    # Learner settings
    parser.add_argument('--shared', action='store_true',
                        help='If set, dataset batches will be theano shared'
                             'variables.')
    parser.add_argument('--track_weights', action='store_true',
                        help='Useful for debugging. Will print out a submatrix'
                             'of the model weights each iteration, to see that'
                             'they are changing.')
    parser.add_argument('--track_weights_change', action='store_true',
                        help='Useful for debugging. Will print out the three'
                             'largest weight changes between iterations.')
    parser.add_argument('--save_every', type=int,
                        help='Saves the intermediate model every k-th epoch.')
    parser.add_argument('--no_overwrite_intermediate_saves', action='store_true',
                        help='If set and saving is set, will not overwrite '
                             'the intermediate saved models and retain all of '
                             'them, not just the last one.')
    parser.add_argument('--resume', action='store_true',
                        help='Attempt to resume training.')


    parser.add_argument('--no_corpus_transform', action='store_true',
                        help='If set, will not save the transformed input'
                             ' corpus.')
    parser.add_argument('--no_dataset_transform', action='store_true',
                        help='If set, will not save the transformed dataset.')
    parser.add_argument('--no_save_transformer', action='store_true',
                        help='If set, will not save the transformer. (Useful'
                             'for debugging purposes.)')
    parser.add_argument('--no_save_learner', action='store_true',
                        help='If set, will not save the learner. Saving the '
                             'learner is useful if you will want to resume '
                             'training later.')

    parser.add_argument('--profile_training', action='store_true',
                        help='If set, will profile the training procedure.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')
    parser.add_argument('--heavy_debug', action='store_true',
                        help='Turn on a brutal amount of debugging from inside'
                             ' the compiled theano functions.')


    return parser


def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))
    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)

    logging.info('Loading text dataset with label %s' % args.text_label)
    tdataset = dloader.load_text(args.text_label)

    logging.info('Setting up model handle with output dimension %d' % args.n_out)
    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError('Invalid model specified: %s' % args.model)

    check_model_dataset_compatibility(tdataset, model_class)

    if issubclass(model_class, models.Autoencoder):
        model_handle = model_class.setup(tdataset, n_out=args.n_out,
                                         heavy_debug=args.heavy_debug,
                                         reconstruction=args.reconstruction)
    else:
        model_handle = model_class.setup(tdataset, n_out=args.n_out,
                                         heavy_debug=args.heavy_debug)

    logging.info('Setting up learner...')
    learner = BaseSGDLearner(n_epochs=args.n_epochs,
                             b_size=args.batch_size,
                             validation_frequency=args.validation_frequency,
                             track_weights=args.track_weights,
                             track_weights_change=args.track_weights_change)

    # Intermediate model saving during training
    if args.save_every:

        learner_saving_overwrite = not args.no_overwrite_intermediate_saves
        learner.set_saving(infix=args.transformation_label,
                           model_loader=mloader,
                           save_every=args.save_every,
                           overwrite=learner_saving_overwrite)

    logging.info('Setting up and training transformer...')
    transformer = SafireTransformer(model_handle, tdataset, learner,
                                    attempt_resume=args.resume,
                                    profile_training=args.profile_training)

    logging.info('Saving transformer with label %s' % args.transformation_label)
    mloader.save_transformer(transformer, args.transformation_label)

    logging.info('Creating transformed corpus with label %s' % args.transformation_label)
    vtcorp = dloader.load_text_corpus(args.text_label)
    transformed_vtcorp = transformer[vtcorp]

    if not args.no_corpus_transform:

        logging.info('Saving transformed corpus with label %s.' % args.transformation_label)
        dloader.serialize_text_corpus(transformed_vtcorp,
                                      args.transformation_label)
        dloader.save_text_corpus(transformed_vtcorp, args.transformation_label)

    if not args.no_dataset_transform:

        logging.info('Saving transformed dataset with label %s.' % args.transformation_label)
        dloader.build_text(transformed_vtcorp, args.transformation_label)




if __name__ == '__main__':

    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
