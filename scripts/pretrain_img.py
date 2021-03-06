#!/usr/bin/env python
"""
Trains an unsupervised layer on image data.

Use to build individual layers of the image stack. Choose your labels carefully!
There is no automated mechanism as of yet to keep track of what is what.

You can build layers on top of already trained layers by simply selecting the
labels for the previous layers. Unless ``--no_transform_corpus`` and
``--no_transform_dataset`` flags are given, the source corpus and dataset
will be at the end run through the trained model and saved with the
``--transformation_label`` label.
"""
import argparse
import logging
import matplotlib.pyplot as plt
import operator
import theano
import theano.compile.pfunc
from safire.data.loaders import MultimodalShardedDatasetLoader, ModelLoader, \
    LearnerLoader
from safire.learning.interfaces import SafireTransformer
from safire.learning.learners import BaseSGDLearner
import safire.learning.models as models
from safire.learning.models import check_model_dataset_compatibility
from safire.utils import ReLU, cappedReLU, build_cappedReLU

print theano.config.compute_test_value

__author__ = 'Jan Hajic jr.'



def init_activation(activation_str):
    """Resolves which activation function to use."""
    activation = theano.tensor.tanh
    if activation_str == 'tanh':
        activation = theano.tensor.tanh
    elif activation_str == 'sigmoid':
        activation = theano.tensor.nnet.sigmoid
    elif activation_str == 'softmax':
        activation = theano.tensor.nnet.softmax
    elif activation_str == 'relu': ## Returns NaN for some reason???
        activation = build_cappedReLU(2.0)

    return activation


def init_dataset_args(args):
    """Initializes args for dataset loading."""
    d_args = {}
    if args.save_dataset:
        d_args['save_dataset'] = args.save_dataset
    if args.load_dataset:
        d_args['load_dataset'] = args.load_dataset
    if args.try_loading:
        d_args['try_loading'] = True

    return d_args


def _build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', action='store', default=None,
                        required=True, help='The path to'+
                        ' the directory which is the root of a dataset.' +
                        ' (Will be passed to a Loader as a root.)')
    parser.add_argument('-n', '--name', help='The dataset name passed to the' +
                        ' Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-i', '--img_label', help='The image corpus label from'
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
    parser.add_argument('--activation', default='tanh',
                        help='The activation function to use. Currently'
                             'supported: \'tanh\', \'sigmoid\','
                             '\'softmax\' and \'relu\'.')
    parser.add_argument('--reconstruction', default='cross-entropy',
                        help='The reconstruction measure to use. Currently '
                             'supported is \'cross-entropy\' and \'mse\'.')
    parser.add_argument('--corruption', type=float, default=0.3,
                        help='Corruption level for denoising autoencoders.')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='Sparsity target for sparse denoising '
                             'autoencoders.')

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

    parser.add_argument('--profile_training', action='store_true',
                        help='If set, will profile the training procedure.')
    parser.add_argument('--plot_monitors', action='store',
                        help='If set, will plot the training and validation '
                             'errors to this file.')
    parser.add_argument('--plot_transformation', action='store_true',
                        help='If set, will plot a sample of the learned '
                             'transformation after each epoch.')
    parser.add_argument('--plot_weights', action='store_true',
                        help='If set, will plot the weights after each epoch.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')
    parser.add_argument('--heavy_debug', action='store_true',
                        help='Turn on a brutal amount of debugging from inside'
                             ' the compiled theano functions.')

    return parser

##############################################################################

def main(args):

    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    mdloader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)

    logging.info('Loading sharded dataset with label %s' % args.img_label)

    idataset = mdloader.load_img(args.img_label)

    logging.info('Setting up SdA handle with output dimension %d' % args.n_out)

    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError('Invalid model specified: %s' % args.model)

    check_model_dataset_compatibility(idataset, model_class)

    activation = init_activation(args.activation)
    model_init_args = {
        'heavy_debug' : args.heavy_debug,
        'activation' : activation,
    }
    if args.model == 'DenoisingAutoencoder':
        model_init_args['corruption_level'] = args.corruption
    if args.model == 'SparseDenoisingAutoencoder' :
        model_init_args['corruption_level'] = args.corruption
        model_init_args['sparsity_target'] = args.sparsity

    model_handle = model_class.setup(idataset, n_out=args.n_out,
                                     **model_init_args)



    logging.info('Setting up learner...')

    lloader = LearnerLoader(args.root, args.name)

    learner = None
    if args.resume:
        try:
            learner = lloader.load_learner(args.transformation_label)
        except Exception:
            logging.warn('Could not load learner for resuming training, will'
                         'start again. (Infix: %s)' % args.transformation_label)

    if not learner:
        learner = BaseSGDLearner(n_epochs=args.n_epochs,
                                 b_size=args.batch_size,
                                 validation_frequency=args.validation_frequency,
                                 track_weights=args.track_weights,
                                 track_weights_change=args.track_weights_change,
                                 plot_transformation=args.plot_transformation,
                                 plot_weights=args.plot_weights)

    # Intermediate model saving during training
    if args.save_every:

        learner_saving_overwrite = not args.no_overwrite_intermediate_saves
        learner.set_saving(infix=args.transformation_label,
                           model_loader=mloader,
                           save_every=args.save_every,
                           overwrite=learner_saving_overwrite)

    logging.info('Setting up and training transformer...')

    transformer = SafireTransformer(model_handle, idataset, learner,
                                    attempt_resume=args.resume,
                                    profile_training=args.profile_training)

    if args.no_save:
        args.no_corpus_transform = True
        args.no_dataset_transform = True
        args.no_save_transformer = True
        args.no_save_learner = True

    if not args.no_save_learner:

        logging.info('Saving learner with label %s' % args.transformation_label)
        lloader.save_learner(learner, args.transformation_label)

    if args.plot_monitors:

        logging.info('Plotting monitors to %s' % args.plot_monitors)
        plt.figure()
        monitor = learner.monitor
        training_cost = monitor['training_cost']
        validation_cost = monitor['validation_cost']

        tc_x = map(operator.itemgetter(0), training_cost)
        tc_y = map(operator.itemgetter(1), training_cost)
        vc_x = map(operator.itemgetter(0), validation_cost)
        vc_y = map(operator.itemgetter(1), validation_cost)

        plt.plot(tc_x, tc_y, 'b')
        plt.plot(vc_x, vc_y, 'g')

        plt.savefig(args.plot_monitors)


    if not args.no_save_transformer:

        logging.info('Saving transformer with label %s' % args.transformation_label)
        mloader.save_transformer(transformer, args.transformation_label)

    logging.info('Creating transformed corpus with label %s' % args.transformation_label)

    icorp = mdloader.load_image_corpus(args.img_label)
    transformed_icorp = transformer[icorp]

    if not args.no_corpus_transform:

        logging.info('Saving transformed corpus with label %s.' % args.transformation_label)

        mdloader.serialize_image_corpus(transformed_icorp,
                                      args.transformation_label)
        mdloader.save_image_corpus(transformed_icorp, args.transformation_label)

    if not args.no_dataset_transform:

        logging.info('Saving transformed dataset with label %s.' % args.transformation_label)

        mdloader.build_img(transformed_icorp, args.transformation_label)

###############################################################################

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
