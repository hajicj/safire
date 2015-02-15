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
from gensim.utils import SaveLoad
import matplotlib.pyplot as plt
import operator
import os
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.datasets.transformations import FlattenComposite
from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.utils.transcorp import dimension, smart_cast_dataset, \
    log_corpus_stack, get_id2word_obj, docnames2indexes, \
    compute_docname_flatten_mapping
import safire
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
import time
import theano
import theano.compile.pfunc
from safire.data.loaders import MultimodalShardedDatasetLoader, ModelLoader, \
    LearnerLoader
from safire.learning.interfaces import SafireTransformer
from safire.learning.learners import BaseSGDLearner
import safire.learning.models as models
from safire.learning.models import check_model_dataset_compatibility
from safire.utils import ReLU, cappedReLU, build_cappedReLU, abstanh, \
    profile_run, parse_textdoc2imdoc_map

print theano.config.compute_test_value

__author__ = 'Jan Hajic jr.'


def init_activation(activation_str):
    """Resolves which activation function to use."""
    activation = theano.tensor.tanh
    if activation_str == 'tanh':
        activation = theano.tensor.tanh
    elif activation_str == 'abstanh':
        activation = abstanh
    elif activation_str == 'sigmoid':
        activation = theano.tensor.nnet.sigmoid
    elif activation_str == 'hard_sigmoid':
        activation = theano.tensor.nnet.hard_sigmoid
    elif activation_str == 'ultra_fast_sigmoid':
        activation = theano.tensor.nnet.ultra_fast_sigmoid
    elif activation_str == 'softmax':
        activation = theano.tensor.nnet.softmax
    elif activation_str == 'softplus':
        activation = theano.tensor.nnet.softplus
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
    parser.add_argument('-i', '--img_label',
                        help='The image corpus label from which to load data. '
                             'Only one of -t, -i can be specified.')
    parser.add_argument('-t', '--text_label',
                        help='The text corpus label from which to load data. '
                             'Only one of -t, -i can be specified.')
    parser.add_argument('--mm_label',
                        help='If an already flattened multimodal dataset is'
                             ' available, use that dataset right away and do'
                             ' not unnecesarily do the combination step.'
                             ' If mm_label is used, img_label and text_label'
                             ' must *not* be used.')
    parser.add_argument('-l', '--transformation_label', action='store',
                        help='The output label. This is to help distinguish ' +
                        'models made with different options. Controls saving names,'
                        'both for the model and for the transformed corpus.')

    parser.add_argument('--w2v', action='store',
                        help='Path to the word2vec dict.')

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
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='How many training epochs should be taken.')
    parser.add_argument('--validation_frequency', type=int, default=100,
                        help='Run validation once every X training batches.')

    # Model settings
    parser.add_argument('--activation', default='sigmoid',
                        help='The activation function to use. Currently'
                             'supported: \'tanh\', \'sigmoid\','
                             '\'softmax\' and \'relu\'.')
    parser.add_argument('--backward_activation', default='sigmoid',
                        help='The backward activation function to use. Currently'
                             ' supported: see --activation.')
    parser.add_argument('--reconstruction', default='cross-entropy',
                        help='The reconstruction measure to use. Currently '
                             'supported is \'cross-entropy\', \'mse\''
                             ' and \'exaggerated-mse\'.')
    parser.add_argument('--corruption', type=float, default=0.3,
                        help='Corruption level for denoising autoencoders.')

    # Sparsity parameters
    parser.add_argument('--sparsity', type=float, default=None,
                        help='Sparsity target for sparse denoising '
                             'autoencoders and sparse RBMs. Operates on'
                             'features.')
    parser.add_argument('--output_sparsity', type=float, default=None,
                        help='Sparsity target for sparse denoising '
                             'autoencoders and sparse RBMs. Operates on'
                             'data point representations.')
    parser.add_argument('--CD_k', type=int, default=1,
                        help='For RBMs: how many pre-training steps should be '
                             'taken.')
    parser.add_argument('--CD_use_sample', action='store_true',
                        help='Use the sample as the negative gradient'
                             ' particle, not the mean.')
    parser.add_argument('--prefer_extremes', type=float,
                        help='Adds an extremes-preference cost: negative log'
                             'of 2 * distance from 0.5 (so that this cost is 0'
                             ' at 1.0 and 0.0).')
    parser.add_argument('--noisy_input', type=float,
                        help='Adds small noise sampled uniformly from up to'
                             'this number to each visible input.')

    # Weight decay
    parser.add_argument('--bias_decay', type=float, default=0.0,
                        help='A decay coefficient for bias parameters only '
                             '(weights stay untouched).')
    parser.add_argument('--L1_norm', type=float, default=0.0,
                        help='Weight of L1-regularization.')
    parser.add_argument('--L2_norm', type=float, default=0.0,
                        help='Weight of L2-regularization.')

    parser.add_argument('--feature_centering', action='store_true',
                        help='If set, will initialize a centering parameter'
                             ' that will subtract the mean of each feature'
                             ' from the input.')

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
    parser.add_argument('--no_overwrite_intermediate_saves',
                        action='store_true',
                        help='If set and saving is set, will not overwrite '
                             'the intermediate saved models and retain all of '
                             'them, not just the last one.')
    parser.add_argument('--resume', action='store_true',
                        help='Attempt to resume training. [BUGS]')

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

    parser.add_argument('--serialize', action='store_true',
                        help='If set, will serialize the pipeline built by '
                             'adding the SafireTransformer block to the input '
                             'pipeline. The serialization will be done by '
                             'ShardedCorpus and the name will be based on the'
                             '--transformation_label argument. Note that if '
                             'this is given, the pipeline will be saved '
                             'including the Serializer block.')

    parser.add_argument('--profile_main', action='store_true',
                        help='If set, will profile the main() function.')
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
    parser.add_argument('--plot_every', action='store', type=int, default=10,
                        help='Plot transformations/weights every X epochs.')
    parser.add_argument('--plot_on_init', action='store_true',
                        help='Plot transformation/weights at initialization.')

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

    if args.root == 'test':
        args.root = safire.get_test_data_root()
        args.name = 'test-data'

    # Initializing loaders
    logging.info('Initializing loaders with root %s, name %s' % (
        args.root, args.name))

    mdloader = MultimodalShardedDatasetLoader(args.root, args.name)
    mloader = ModelLoader(args.root, args.name)

    # Loading datasets
    if args.mm_label and (args.img_label or args.text_label):
        raise ValueError('Cannot specify both mm_label and'
                         ' img_label/text_label.')

    if not args.img_label and not args.text_label and not args.mm_label:
        raise ValueError('Must specify text/image label or both or mm_label.')

    if args.img_label and args.text_label:
        logging.info('Will train a multimodal model: text label {0}, image '
                     'label {1}.'.format(args.img_label, args.text_label))

        logging.info('Assuming')
        #raise ValueError('Can only specify one of text and image label.')

    # Need to refactor dataset loading.
    # ...no more difference in principle between image labels and text labels.
    if args.img_label:
        logging.info('Loading image dataset with img. label {0}'
                     ''.format(args.img_label))
        pipeline_fname = mdloader.pipeline_name(args.img_label)

        #  - load the pipeline
        img_pipeline = SaveLoad.load(fname=pipeline_fname)
        # cast to Dataset
        img_pipeline = Dataset(img_pipeline)

    if args.text_label:
        logging.info('Loading text dataset with text label {0}'
                     ''.format(args.text_label))
        pipeline_fname = mdloader.pipeline_name(args.text_label)

        #  - load the pipeline
        text_pipeline = SaveLoad.load(fname=pipeline_fname)
        # - Cast to dataset
        text_pipeline = Dataset(text_pipeline, ensure_dense=True)

        # This is specifically a text transformation.
        if args.w2v:
            logging.info('Building and applying word2vec sampler. Note that '
                         'this will mean no serialization is performed after'
                         ' flattening, in case this is applied in a multimodal'
                         ' setting.')
            w2v_trans = Word2VecTransformer(args.w2v,
                                            get_id2word_obj(text_pipeline))
            w2v_sampler = Word2VecSamplingDatasetTransformer(w2v_trans)

            text_pipeline = w2v_sampler[text_pipeline]

    if (not args.text_label) and args.img_label:
        pipeline = img_pipeline
    elif args.text_label and (not args.img_label):
        pipeline = text_pipeline
    elif args.text_label and args.img_label:
        logging.info('Combining text and image sources into a multimodal '
                     'pipeline.')
        logging.info('Text pipeline:\n{0}'.format(log_corpus_stack(text_pipeline)))
        logging.info('Image pipeline:\n{0}'.format(log_corpus_stack(img_pipeline)))

        # - Combine into CompositeDatasest
        mm_composite_dataset = CompositeDataset((text_pipeline, img_pipeline),
                                                names=('txt', 'img'),
                                                aligned=False)
        # - Flatten the dataset
        #    - Load flatten indices
        t2i_file = os.path.join(mdloader.root,
                                mdloader.layout.textdoc2imdoc)
        # t2i_map = parse_textdoc2imdoc_map(t2i_file)
        # t2i_list = [[text, image]
        #             for text in t2i_map
        #             for image in t2i_map[text]]
        # Sorting the indices is an optimization for underlying ShardedCorpus
        # serializers.
        t2i_indexes = compute_docname_flatten_mapping(mm_composite_dataset,
                                                      t2i_file)

        #    - Initialize flattening transformer
        flatten = FlattenComposite(mm_composite_dataset, indexes=t2i_indexes)

        #    - Apply
        pipeline = flatten[mm_composite_dataset]

        if not args.w2v:
            #    - Serialize, because multimodal indexed retrieval is *slow*
            serialization_name = mdloader.pipeline_serialization_target(
                args.text_label + '__' + args.img_label)
            logging.info('Serializing flattened multimodal data to {0}.'
                         ''.format(serialization_name))

            logging.debug('Pre-serialization pipeline: {0}'
                          ''.format(log_corpus_stack(pipeline)))
            serializer = Serializer(pipeline, ShardedCorpus, serialization_name,
                                    dim=dimension(pipeline), gensim=False)
            pipeline = serializer[pipeline]
        else:
            logging.warn('Word2vec sampling active, cannot serialize flattened'
                         'corpus.')

    if args.mm_label:
        logging.info('Loading multimodal pipeline with label {0}'
                     ''.format(args.mm_label))
        pipeline_name = mdloader.pipeline_name(args.mm_label)
        pipeline = SaveLoad.load(pipeline_name)

    logging.info('Loaded pipeline:\n{0}'.format(log_corpus_stack(pipeline)))

    #  - cast to dataset
    dataset = smart_cast_dataset(pipeline, test_p=0.1, devel_p=0.1,
                                 ensure_dense=True)

    logging.info('Setting up %s handle with output dimension %d' % (args.model,
                                                                    args.n_out))
    # Loading model class
    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError('Invalid model specified: %s' % args.model)

    check_model_dataset_compatibility(dataset, model_class)

    # Setting up model initialization arguments
    activation = init_activation(args.activation)
    if not args.backward_activation:
        args.backward_activation = args.activation
    backward_activation = init_activation(args.backward_activation)

    model_init_args = {
        'heavy_debug': args.heavy_debug,
        'activation': activation,
        'backward_activation': backward_activation
    }
    if args.model == 'DenoisingAutoencoder':
        model_init_args['corruption_level'] = args.corruption
        model_init_args['reconstruction'] = args.reconstruction
        model_init_args['L1_norm'] = args.L1_norm
        model_init_args['L2_norm'] = args.L2_norm
        model_init_args['bias_decay'] = args.bias_decay
        model_init_args['sparsity_target'] = args.sparsity
        model_init_args['output_sparsity_target'] = args.output_sparsity

    if args.model == 'SparseDenoisingAutoencoder':
        model_init_args['corruption_level'] = args.corruption
        model_init_args['sparsity_target'] = args.sparsity
        model_init_args['reconstruction'] = args.reconstruction

    if args.model == 'RestrictedBoltzmannMachine' or args.model == 'ReplicatedSoftmax':
        model_init_args['sparsity_target'] = args.sparsity
        model_init_args['output_sparsity_target'] = args.output_sparsity
        model_init_args['CD_k'] = args.CD_k
        model_init_args['bias_decay'] = args.bias_decay
        model_init_args['CD_use_mean'] = not args.CD_use_sample
        model_init_args['prefer_extremes'] = args.prefer_extremes
        model_init_args['L1_norm'] = args.L1_norm
        model_init_args['L2_norm'] = args.L2_norm
        model_init_args['noisy_input'] = args.noisy_input

    logging.info('\nModel init args:' +
                 u'\n'.join([u'  {0}: {1}'.format(k, v)
                             for k, v in model_init_args.items()]))

    # Set up model
    model_handle = model_class.setup(dataset, n_out=args.n_out,
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
        learner = BaseSGDLearner(
            n_epochs=args.n_epochs,
            b_size=args.batch_size,
            validation_frequency=args.validation_frequency,
            track_weights=args.track_weights,
            track_weights_change=args.track_weights_change,
            plot_transformation=args.plot_transformation,
            plot_weights=args.plot_weights,
            plot_every=args.plot_every,
            plot_on_init=args.plot_on_init)

    # Intermediate model saving during training
    if args.save_every:

        learner_saving_overwrite = not args.no_overwrite_intermediate_saves
        learner.set_saving(infix=args.transformation_label,
                           model_loader=mloader,
                           save_every=args.save_every,
                           overwrite=learner_saving_overwrite)

    logging.info('Setting up and training transformer...')

    # Training starts here.
    transformer = SafireTransformer(model_handle, dataset, learner,
                                    attempt_resume=args.resume,
                                    profile_training=args.profile_training)

    # Training is done at this point.

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

    logging.info('Creating transformed corpus with label {0}'
                 ''.format(args.transformation_label))
    # This applies the transformation to the input corpus.
    pipeline = transformer[pipeline]

    # Serialization (this should be wrapped in some utility function?)
    # Doesn't always have to happen. (Difference from dataset2corpus.)
    if args.serialize:
        serializer_class = ShardedCorpus
        data_name = mdloader.pipeline_serialization_target(args.transformation_label)
        serialization_start_time = time.clock()
        logging.info('Starting serialization: {0}'
                     ''.format(serialization_start_time))
        serializer_block = Serializer(pipeline, serializer_class,
                                      data_name,
                                      dim=dimension(pipeline))
        serialization_end_time = time.clock()
        logging.info('Serialization finished: {0}'
                     ''.format(serialization_end_time))

        pipeline = serializer_block[pipeline]

    # Now we save the pipeline. This is analogous to the Dataset2Corpus step.
    # In this way, also, the learned transformation is stored and can be
    # recycled, and other handles can be derived from the sftrans.model_handle.
    pipeline_savename = mdloader.pipeline_name(args.transformation_label)
    logging.info('    Pipeline name: {0}'.format(pipeline_savename))

    pipeline.save(pipeline_savename)

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

    if args.profile_main:
        report, _ = profile_run(main, args)
        print report.getvalue()
    else:
        main(args)
