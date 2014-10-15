#!/usr/bin/env python

"""Helper script that builds the test-data LogisticRegression transformer
and index. Intended to be run from the safire/source directory."""
import argparse
import copy
import logging
import safire

from gensim.similarities import Similarity
import theano
theano.config.exception_verbosity = 'high'

from safire.data.loaders import MultimodalDatasetLoader, MultimodalShardedDatasetLoader
from safire.data.loaders import ModelLoader
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from safire.learning.interfaces.safire_transformer import SafireTransformer
import safire.learning.models as models

__author__ = 'Jan Hajic jr.'

############################

# Default values for multilayer perceptron layer parameters.
hidden_layer_defaults = {
    'activation' : theano.tensor.tanh
}

# Default values for pre-training layer parameters.
pretraining_defaults = {
    'corruption_level' : 0.3,
    'tied_weights' : True,
    'activation' : theano.tensor.tanh
}


def _create_transformer(*args, **kwargs):
    """Wrapper around transformer training for profiling purposes."""
    transformer = SafireTransformer(*args, **kwargs)
    return transformer

###############################################

# Functions for loading individual layers

def layer_loading_infixes(load_layers):
    """Parses the layers-to-load infixes. Returns a dict such that
    i-th layer infix has key i."""
    output_infixes = {}
    for i, ll in enumerate(load_layers):
        if ll != 'None':
            output_infixes[i] = ll
    return output_infixes

def layer_loading_init_args(layer_infixes, model_loader):
    """Loads for each layer with infix specified by layer_infixes the
    appropriate init args. Returns the dict with init args as values."""
    layer_init_args = {}
    for layer in layer_infixes:
        model = model_loader.load_model(layer_infixes[layer])
        init_args = model._init_args_snapshot()
        layer_init_args[layer] = init_args
    return layer_init_args

###################################################

def set_dataset_mode(args, dataset):

    if args.mode == 'unsupervised':
        dataset.set_mode(0)
        dataset.n_out = args.n_out
    elif args.mode == 'supervised':
        dataset.set_mode(1)
    elif args.mode == 'supervised-reverse':
        dataset.set_mode(2)
    else:
        raise ValueError('Incorrect mode set: %s' % args.mode)

def check_model_dataset_compatibility(dataset, model_class):
    if issubclass(model_class, models.BaseSupervisedModel):
        if dataset.mode == 0:
            raise ValueError('Cannot run supervised training with unsupervised'+
                             ' dataset mode. (Model %s, dataset mode %s' %
                             (str(model_class), str(dataset.mode)))

###############################################################################

def parse_model_init_args(args):
    """Creates a dict that can be passed like kwargs to the model's setup()
    method.

    :param args: The argparse.Namespace that holds script command-line args.

    :returns: The init arg dict.
    """
    init_args = copy.copy(hidden_layer_defaults)

    if hasattr(args, 'activation'):
        if args.activation == 'sigmoid':
            init_args['activation'] = theano.tensor.nnet.sigmoid
        elif args.activation == 'tanh':
            init_args['activation'] = theano.tensor.tanh
        else:
            raise ValueError('Invalid activation given: ' +
                             '%s, use \'sigmoid\' or \'tanh\'.' % args.activation)

    # Special case...
    if args.model == 'MultilayerPerceptron':
        if 'activation' in init_args:
            init_args['hidden_activation'] = init_args['activation']
            #init_args['log_regression_params'] = { 'activation' : init_args['activation'] }
            del init_args['activation']

    if hasattr(args, 'n_layers'):
        init_args['n_layers'] = args.n_layers

    if hasattr(args, 'hidden_sizes'):
        init_args['n_hidden_list'] = args.hidden_sizes

    if hasattr(args, 'L1_w'):
        init_args['L1_w'] = args.L1_w

    if hasattr(args, 'L2_w'):
        init_args['L1_w'] = args.L2_w


    return init_args


###############################################################################
def main(args):

    logging.info('Initializing loaders (root: %s, name: %s)' % (
                 args.root, args.name))

    # Initialize loaders
    dloader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)

    logging.info('Loading dataset (text label: %s, image label: %s)' % (
                 args.text_label, args.img_label))

    # Setup and train model
    dataset = dloader.load(text_infix=args.text_label,
                           img_infix=args.img_label)

    # Correctly initialize dataset mode (and n_out, for unsupervised settings)
    set_dataset_mode(args, dataset)

    logging.info('Dataset dimension: n_in = %d, n_out = %d' % (
        dataset.n_in, dataset.n_out))

    logging.info('Initializing model class %s setting up handle.' % args.model)

    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError('Invalid model specified: %s' % args.model)

    check_model_dataset_compatibility(dataset, model_class)

    model_init_args = parse_model_init_args(args)

    # Parsing intermediate layer loading
    if hasattr(args, 'load_layers'):
        l_infixes = layer_loading_infixes(args.load_layers)
        l_init_args = layer_loading_init_args(l_infixes, mloader)

        if model_class == 'MultilayerPerceptron': # This will usually be the case.
            hparams = [ {} for _ in xrange(args.n_layers) ]
            for l in l_init_args:
                hparams[l] = l_init_args[l]
            model_init_args['hidden_layer_params'] = hparams

    model_handle = model_class.setup(dataset,
                                     batch_size=args.batch_size,
                                     learning_rate=args.learning_rate,
                                     **model_init_args)

    logging.info('Initializing learner...')

    learner = BaseSGDLearner(n_epochs=args.n_epochs,
                             b_size=args.batch_size,
                             validation_frequency=args.validation_frequency)

    #if args.pretraining:
    #    #pretraining_args = parse_pretraining_args(args) # TODO
    #    #pretraining_handles = setup_pretraining_handles(pretraining_args)
    #
    # TODO: pretraining execution. Currently not implemented.
    #

    logging.info('Initializing and training transformer...')

    # Train & save
    if args.no_training:
        transformer = SafireTransformer(model_handle)
    else:
        if args.profile_training:
            report, transformer = safire.utils.profile_run(_create_transformer,
                                                model_handle, dataset, learner)
        else:
            transformer = SafireTransformer(model_handle, dataset, learner)

    logging.info('Saving transformer...')

    mloader.save_transformer(transformer, args.transformer_label)

###############################################################################

def _build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    # Dataset identification
    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory, passed to Loader.')
    parser.add_argument('-n', '--name', required=True,
                        help='The name passed to Loader.')
    parser.add_argument('-t', '--text_label', default=None,
                        help='Label of text corpus to use for training the ' +
                             'model. Used to load an existing text corpus' +
                             ' (serialized) in the multimodal dataset.')
    parser.add_argument('-i', '--img_label', default=None,
                        help='Label of image corpus to use for building the ' +
                             'model. (Important especially in supervised ' +
                             'training.)')

    parser.add_argument('--mode', default='supervised',
                        help='Which training mode should be used. The choice '
    'is: \'supervised\', for training on texts with images as response, '
    '\'unsupervised\' for training on texts and images together, and '
    '\'supervised-reverse\', for training on images with texts as response.')
    parser.add_argument('--n_out', default=None,
                        help='If unsupervised training mode is used, the number'
                             'of output neurons needs to be set.')

    # Model settings
    parser.add_argument('-m', '--model', default='MultilayerPerceptron',
                        help='Name of model class. In order to use multiple'
                             'layers, a MultilayerPerceptron should be used.')
    parser.add_argument('--n_layers', default=0, type=int,
                        help='No. of hidden layers.')
    parser.add_argument('--hidden_sizes', default=[], type=int, nargs='+',
                        help='The sizes of hidden layers.')
    parser.add_argument('--activation', default='tanh',
                        help='Accepts \'sigmoid\', \'tanh\' and \'softmax\'.')
    parser.add_argument('--layer_labels', default=None, nargs='+',
                        help='For loading layers. If the i-th layers should be'
                             ' loaded, the model infix for the layer should be'
                             ' the i-th argument to --layer_labels. If a model'
                             ' should not be loaded for a layer, give the'
                             ' layer label as None (the script will parse it'
                             ' correctly).')

    # Pretraining settings
    parser.add_argument('-p', '--pretraining', default=None,
                        help='If set, gives the unsupervised model class used'
                             ' for pre-training. Currently, all layers are'
                             ' pre-trained using the same default model:'
                             ' Denoising Autoencoder')
    #parser.add_argument('--pmodel', default='DenoisingAutoencoder',
    #                    help='Which model class to use for pre-training.')


    # TODO:
    parser.add_argument('-c', '--config', default=None,
                        help='Configuration file label. The configuration file'
                             'is a Python file that contains')

    # Saving the trained model
    parser.add_argument('-l', '--transformer_label', default=None,
                        help='The label under which to save the trained ' +
                             'SafireTransformer object.')
    parser.add_argument('--no_training', action='store_true',
                        help='Doesn\'t train model, only performs initialization.'
                             ' Useful for debugging save/load or dimension'
                             ' problems.')

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
    parser.add_argument('--profile_training', action='store_true', help='Turn '
                        'on profiling for model training.')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on'+
                        ' INFO logging messages.')
    parser.add_argument('--debug', action='store_true', help='Turn on debug '+
                        'prints.')

    return parser

##############################################################################

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


