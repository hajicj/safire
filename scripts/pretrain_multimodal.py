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
from safire.learning.models import DenoisingAutoencoder, \
    check_model_dataset_compatibility
from safire.data.loaders import MultimodalDatasetLoader, \
    MultimodalShardedDatasetLoader, LearnerLoader
from safire.data.loaders import ModelLoader
from safire.learning.interfaces import SafireTransformer
from safire.learning.learners import BaseSGDLearner


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
    parser.add_argument('-i', '--img_label', help='The image corpus label from'
                                                  'which to load data.')

    parser.add_argument('-j', '--joint_label', action='store',
                        help='The output label. This is to help distinguish ' +
                        'models made with different options. Controls saving names,'
                        'both for the model and for the transformed corpus.'
                        '!!!!! For the multimodal corpus, USE ONLY THE JOINT '
                        'LAYER LABEL as the joint label will be automatically '
                        'generated using the text and image labels and this one.')
    parser.add_argument('--n_out', type=int, default=1000,
                        help='The number of model output (joint layer) neurons.')
    parser.add_argument('-m', '--model', default='DenoisingAutoencoder',
                        help='The name of the model class to use. Default is'
                             'DenoisingAutoencoder.')

    # Learner settings
    parser.add_argument('--batch_size', type=int, default=100,
                        help='How many data points will be used to perform one'
                             'gradient update.')
    parser.add_argument('--learning_rate', type=float, default=0.13,
                        help='How fast will the gradient update be.')
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='How many training epochs should be taken.')
    parser.add_argument('--validation_frequency', type=int, default=4,
                        help='Run validation once every X training batches.')

    # Monitoring settings
    parser.add_argument('--profile_training', action='store_true',
                        help='If set, will profile the training procedure.')
    parser.add_argument('--plot_transformation', action='store_true',
                        help='If set, will plot a sample of the learned '
                             'transformation after each epoch.')
    parser.add_argument('--plot_weights', action='store_true',
                        help='If set, will plot the weights after each epoch.')

    # Model settings
    parser.add_argument('--reconstruction',
                        help='The reconstruction measure to use. Currently '
                             'supported is \'cross-entropy\' and \'mse\'.'
                             ' Only valid for autoencoder models.')

    # Saving.
    parser.add_argument('--no_save', action='store_true',
                        help='If set, will not save anything.')
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
    mdataset = dloader.load(text_infix=args.text_label,
                            img_infix=args.img_label)

    logging.info('Setting up model handle with output dimension %d' % args.n_out)
    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError('Invalid model specified: %s' % args.model)

    check_model_dataset_compatibility(mdataset, model_class)

    if issubclass(model_class, models.Autoencoder):
        model_handle = model_class.setup(mdataset, n_out=args.n_out,
                                         heavy_debug=args.heavy_debug,
                                         reconstruction=args.reconstruction)
    else:
        model_handle = model_class.setup(mdataset, n_out=args.n_out,
                                         heavy_debug=args.heavy_debug)

    logging.info('Setting up learner...')
    lloader = LearnerLoader(args.root, args.name)
    # No resuming here (yet)...
    learner = BaseSGDLearner(n_epochs=args.n_epochs,
                             b_size=args.batch_size,
                             validation_frequency=args.validation_frequency,
                             plot_transformation=args.plot_transformation,
                             plot_weights=args.plot_weights)

    logging.info('Setting up and training transformer...')
    transformer = SafireTransformer(model_handle, mdataset, learner,
                                    profile_training=args.profile_training)

    transformation_label = args.text_label + '-' + args.img_label + '-' + args.joint_label

    logging.info('Creating transformed corpus with label %s' % transformation_label)
    vtcorp = dloader.load_text_corpus(args.text_label)
    transformed_vtcorp = transformer[vtcorp]

    if args.no_save:
        args.no_corpus_transform = True
        args.no_dataset_transform = True
        args.no_save_transformer = True
        args.no_save_learner = True

    if not args.no_save_learner:

        logging.info('Saving learner with label %s' % transformation_label)
        lloader.save_learner(learner, transformation_label)

    if not args.no_save_transformer:

        logging.info('Saving transformer with label %s' % transformation_label)
        mloader.save_transformer(transformer, transformation_label)

    if not args.no_corpus_transform:

        logging.info('Saving transformed joint repr. as corpus with label %s.' % transformation_label)
        dloader.serialize_text_corpus(transformed_vtcorp,
                                      transformation_label)
        dloader.save_text_corpus(transformed_vtcorp, transformation_label)

    if not args.no_dataset_transform:

        logging.info('Saving transformed joint repr. as dataset with label %s.' % transformation_label)
        dloader.build_text(transformed_vtcorp, transformation_label)




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
