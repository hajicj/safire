"""Tests the lasagne integration module."""
import logging
import numpy
import os
import sys
import theano.tensor as TT
from safire.learning.learners import BaseSGDLearner
from safire.datasets.dataset import Dataset
from safire.datasets.dataset import SupervisedDataset
from safire.data.composite_corpus import CompositeCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.document_filter import DocumentFilterTransform
from safire.data.filters.frequency_filters import zero_length_filter
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.data.word2vec_transformer import Word2VecTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.vtextcorpus import VTextCorpus
from safire.learning.interfaces.model_handle import ModelHandle
from safire.learning.interfaces.safire_transformer import SafireTransformer
from safire.utils.transcorp import get_id2word_obj, dimension, \
    compute_docname_flatten_mapping
from safire.utils.transformers import GeneralFunctionTransform, \
    ReorderingTransform

try:
    import lasagne
except ImportError:
    logging.warn('Unable to import Lasagne (not installed?), skipping lasagne'
                 ' integration unittests.')


__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase
from safire.learning.interfaces.lasagne_integration import LasagneSetup

vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}
vtlist_fname = 'test-data.vtlist'

BATCH_SIZE = 1

@unittest.skipIf('lasagne' not in sys.modules, 'Unable to import Lasagne.')
class TestLasagneIntegration(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=True, no_datasets=True):
        super(TestLasagneIntegration, cls).setUpClass(clean_only=clean_only,
                                                      no_datasets=no_datasets)

        cls.edict_pkl_fname = os.path.join(cls.data_root, 'test-data.edict.pkl')
        cls.e_matrix_fname = os.path.join(cls.data_root, 'test-data.emtr.pkl')

        cls.loader = MultimodalShardedDatasetLoader(cls.data_root,
                                                    'test-data')

        # Text pipeline
        # -------------
        cls.vtlist = os.path.join(cls.loader.root, cls.loader.layout.vtlist)
        cls.token_vtcorp = VTextCorpus(cls.vtlist, input_root=cls.data_root,
                                       tokens=True,
                                       **vtcorp_settings)
        cls.token_vtcorp.dry_run()
        cls.token_vtcorp.lock()

        text_pipeline = cls.token_vtcorp

        word2vec = Word2VecTransformer(cls.edict_pkl_fname,
                                       get_id2word_obj(cls.token_vtcorp))
        cls.token_vtcorp.dry_run()
        text_pipeline = word2vec[text_pipeline]

        word2vec_miss_filter = DocumentFilterTransform(zero_length_filter)
        text_pipeline = word2vec_miss_filter[text_pipeline]

        ttanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        text_pipeline = ttanh[text_pipeline]

        serializer = Serializer(text_pipeline, ShardedCorpus,
                                cls.loader.pipeline_serialization_target(
                                    '.tokenw2v'))
        cls.text_pipeline = serializer[text_pipeline]

        # Image pipeline
        # --------------

        image_file = os.path.join(cls.data_root,
                                  cls.loader.layout.image_vectors)
        img_pipeline = ImagenetCorpus(image_file,
                                      delimiter=';',
                                      dim=4096,
                                      label='')

        itanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        img_pipeline = itanh[img_pipeline]

        print '--- Serializing icorp.. ---'
        serializer = Serializer(img_pipeline, ShardedCorpus,
                                cls.loader.pipeline_serialization_target(
                                    '.icorp'))
        cls.img_pipeline = serializer[img_pipeline]
        mmdata = CompositeCorpus((cls.text_pipeline, cls.img_pipeline),
                                 names=('txt', 'img'),
                                 aligned=False)
        t2i_file = os.path.join(cls.loader.root,
                                cls.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)
        t_mapping, i_mapping = zip(*t2i_indexes)
        t_reorder = ReorderingTransform(t_mapping)
        cls.t_reordered = t_reorder[cls.text_pipeline]
        i_reorder = ReorderingTransform(i_mapping)
        cls.i_reordered = i_reorder[cls.img_pipeline]

    def setUp(self):
        # Create the Lasagne model (let's train text features from pictures...)
        self.model = self._setUpLasagneModel()

    def _setUpLasagneModel(self):
        l_in = lasagne.layers.InputLayer(
            shape=(BATCH_SIZE, dimension(self.img_pipeline)),
        )
        l_hidden1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
        l_hidden1_dropout = lasagne.layers.DropoutLayer(
            l_hidden1,
            p=0.5,
        )
        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
        l_hidden2_dropout = lasagne.layers.DropoutLayer(
            l_hidden2,
            p=0.5,
        )
        l_out = lasagne.layers.DenseLayer(
            l_hidden2_dropout,
            num_units=dimension(self.text_pipeline),
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        return l_out

    def test_lasagne_setup(self):

        data = SupervisedDataset((Dataset(self.t_reordered),
                                  Dataset(self.i_reordered)))

        # Prepare loss function. Code taken from lasagne/examples/mnist.py
        X_batch = TT.matrix('X')
        y_batch = TT.matrix('y')
        batch_index = TT.iscalar('batch_index')
        # batch_slice = slice(batch_index * BATCH_SIZE,
        #                     (batch_index + 1) * BATCH_SIZE)

        print '--- Initializing objective... ---'
        objective = lasagne.objectives.Objective(
            self.model,
            loss_function=lasagne.objectives.mse)
        loss_train = objective.get_loss(X_batch, target=y_batch)
        loss_monitor = objective.get_loss(X_batch, target=y_batch,
                                          deterministic=True)

        print '--- Initializing updater... ---'
        updater = lasagne.updates.nesterov_momentum
        updater_settings = {'learning_rate': 0.001,
                            'momentum': 0.9}

        print '--- Initializing LasagneSetup... ---'
        lasagne_setup = LasagneSetup()

        print '--- Running setup... ---'
        setup_handles = lasagne_setup.setup(
            self.model,
            loss_train, [X_batch, y_batch],
            updater=updater, updater_kwargs=updater_settings,
            monitor_expr=loss_monitor, monitor_inputs=[X_batch, y_batch])

        self.assertIsInstance(setup_handles, dict)
        self.assertTrue('train' in setup_handles)
        self.assertTrue('validate' in setup_handles)
        self.assertTrue('test' in setup_handles)
        self.assertTrue('run' in setup_handles)
        self.assertIsInstance(setup_handles['train'], ModelHandle)
        self.assertIsInstance(setup_handles['validate'], ModelHandle)
        self.assertIsInstance(setup_handles['test'], ModelHandle)
        self.assertIsInstance(setup_handles['run'], ModelHandle)

        print '--- Applying sftrans without training... ---'
        sftrans = SafireTransformer(setup_handles['run'])
        i2t_pipeline = sftrans[self.img_pipeline]

        output = [i for i in i2t_pipeline]
        self.assertEqual(len(output), len(self.img_pipeline))
        self.assertEqual(dimension(i2t_pipeline), dimension(self.text_pipeline))
        print output[0]

        print '--- Training a sftrans... ---'
        learner = BaseSGDLearner(n_epochs=10, b_size=1, validation_frequency=1)
        trained_sftrans = SafireTransformer(setup_handles['run'],
                                            setup_handles,
                                            dataset=data,
                                            learner=learner)

        trained_i2t_pipeline = trained_sftrans[self.img_pipeline]
        output = [i for i in trained_i2t_pipeline]
        self.assertEqual(len(output), len(self.img_pipeline))
        self.assertEqual(dimension(i2t_pipeline), dimension(self.text_pipeline))
        print output[0]

        print '   Learner validation costs: {0}' \
              ''.format(learner.monitor['validation_cost'])


############################################################################


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestLasagneIntegration)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
