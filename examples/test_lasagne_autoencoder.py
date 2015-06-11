"""Sandbox test that doesn't correspond to anything in Safire. Just playing
with Lasagne and unit testing seemed like a good way of doing this.
"""
import logging
import unittest
import os
import matplotlib.pyplot as plt
import numpy
from safire.learning.interfaces.safire_transformer import SafireTransformer
from safire.learning.interfaces.model_handle import ModelHandle
from safire.learning.learners.base_sgd_learner import BaseSGDLearner
from safire.utils.transcorp import dimension
from safire.learning.interfaces.lasagne_integration import LasagneSetup
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset
from safire.utils.transformers import GeneralFunctionTransform
from safire.data.imagenetcorpus import ImagenetCorpus
import theano
import theano.tensor as TT

try:
    import lasagne
except ImportError:
    logging.warn('Unable to import Lasagne (not installed?), skipping lasagne'
                 ' integration unittests.')
    lasagne = None

__author__ = 'Jan Hajic jr'

##############################################################################


def build_autoencoder(input_pipeline, batch_size=100):
    """The `get_autoencoder()` function builds the autoencoder model."""
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, dimension(input_pipeline))
    )
    l_hidden = lasagne.layers.DenseLayer(
        l_in,
        num_units=1000,
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='bottleneck_layer'
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden,
        num_units=l_in.shape[1],
        nonlinearity=lasagne.nonlinearities.tanh,
        # W=l_hidden.W.T  # Tied weights so far not working
        name='reconstruction_layer'
    )
    return l_out


def get_loss(autoencoder, X_tensor_type=TT.matrix):
    """This function generates the loss expressions for an autoencoder
    architecture, one for training (non-deterministic), one for transformation
    runs (deterministic)."""
    objective = lasagne.objectives.Objective(
        autoencoder,
        loss_function=lasagne.objectives.mse)

    X_batch = X_tensor_type('x')
    loss_train = objective.get_loss(input=X_batch, target=X_batch)
    loss_run = objective.get_loss(X_batch, target=X_batch,
                                  deterministic=True)

    return loss_train, loss_run

##############################################################################


class TestLasagneAutoencoder(unittest.TestCase):

    SAFIRE_ROOT = os.path.join(os.getenv('SAFIRE_DATA'), 'safire')

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        cls.data_root = os.path.join(os.getenv('SAFIRE_DATA'), 'safire')
        cls.loader = MultimodalShardedDatasetLoader(cls.data_root,
                                                    'safire-notabloid')

        image_file = os.path.join(cls.SAFIRE_ROOT,
                                  cls.loader.layout.image_vectors)
        img_pipeline = ImagenetCorpus(image_file,
                                      delimiter=';',
                                      dim=4096,
                                      label='')

        itanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        img_pipeline = itanh[img_pipeline]

        serializer = Serializer(img_pipeline, ShardedCorpus,
                                cls.loader.pipeline_serialization_target(
                                    '.icorp'))
        cls.img_pipeline = serializer[img_pipeline]

        autoencoder = build_autoencoder(cls.img_pipeline, batch_size=100)
        X_batch = TT.matrix('X')
        X_batch.tag.test_value = numpy.ones((10, dimension(cls.img_pipeline)),
                                            dtype=theano.config.floatX)

        objective = lasagne.objectives.Objective(
            autoencoder,
            loss_function=lasagne.objectives.mse)

        loss_train = objective.get_loss(input=X_batch, target=X_batch)
        loss_monitor = objective.get_loss(X_batch, target=X_batch,
                                          deterministic=True)

        theano.printing.pydotprint(loss_monitor, 'temp.png')

        # TODO: get named middle layer's ouptut expression."""
        # run_expr = lasagne.layers.???

        updater = lasagne.updates.nesterov_momentum
        updater_settings = {'learning_rate': 0.001,
                            'momentum': 0.9}

        lasagne_setup = LasagneSetup()

        cls.setup_handles = lasagne_setup.setup(
            autoencoder,
            loss_train, [X_batch],
            updater=updater, updater_kwargs=updater_settings,
            monitor_expr=loss_monitor, monitor_inputs=[X_batch],
            heavy_debug=True
        )

    def test_setup_works(self):

        setup_handles = self.setup_handles
        self.assertIsInstance(setup_handles, dict)
        self.assertTrue('train' in setup_handles)
        self.assertTrue('validate' in setup_handles)
        self.assertTrue('test' in setup_handles)
        self.assertTrue('run' in setup_handles)
        self.assertIsInstance(setup_handles['train'], ModelHandle)
        self.assertIsInstance(setup_handles['validate'], ModelHandle)
        self.assertIsInstance(setup_handles['test'], ModelHandle)
        self.assertIsInstance(setup_handles['run'], ModelHandle)
        
    def test_training(self):
        learner = BaseSGDLearner(n_epochs=10, b_size=100,
                                 validation_frequency=10,
                                 plot_transformation=True,
                                 plot_every=11)

        print 'Starting training..'
        trained_sftrans = SafireTransformer(self.setup_handles['run'],
                                            self.setup_handles,
                                            dataset=Dataset(self.img_pipeline,
                                                            devel_p=0.1,
                                                            test_p=0.1),
                                            learner=learner)

        monitor_values = learner.monitor['validation_cost']
        plt.plot(monitor_values)
        plt.show()

        # Bogus assertion
        self.assertIsInstance(trained_sftrans, SafireTransformer)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestLasagneAutoencoder)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
