import logging
import os
from safire.data.loaders import MultimodalShardedDatasetLoader, ModelLoader
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import MultilayerPerceptron
from test import SafireTestCase

__author__ = 'Jan Hajic jr.'

import unittest


class TestMultilayerPerceptron(SafireTestCase):

    @classmethod
    def setUpClass(cls):

        super(TestMultilayerPerceptron, cls).setUpClass()

        cls.dloader = MultimodalShardedDatasetLoader(cls.data_root, 'test-data')
        cls.mloader = ModelLoader(cls.data_root, 'test-data')

        cls.dataset = cls.dloader.load()
        cls.dataset.set_mode(1)

        cls.init_args = {
            'n_layers' : 3,
            'n_hidden_list' : [ 100, 10, 4096 ]
        }

        cls.temp_files = []

        cls.test_infix = '.test_label'

    @classmethod
    def tearDownClass(cls):

        for f in cls.temp_files:
            os.remove(f)

        super(TestMultilayerPerceptron, cls).tearDownClass()

    def setUp(self):

        self.model_handle = MultilayerPerceptron.setup(self.dataset,
                                                       **self.init_args)

    def test_mlp_training(self):

        pre_training_weights = [ layer.W.get_value(borrow=True)[0,0]
                        for layer in self.model_handle.model_instance.layers]

        learner = BaseSGDLearner(n_epochs=3, b_size=2, validation_frequency=4)

        learner.run(self.model_handle, self.dataset)

        post_training_weights = [ layer.W.get_value(borrow=True)[0,0]
                        for layer in self.model_handle.model_instance.layers]

        self.assertNotEqual(pre_training_weights, post_training_weights)

    def test_partial_saveload(self):

        model = self.model_handle.model_instance

        partial_savename_base = os.path.join(self.data_root,
                        self.dloader.layout.get_model_file(self.test_infix))

        layer_savenames = []
        for i in xrange(model.n_layers):

            layer_savename = partial_savename_base + '.' + str(i)
            model.save_layer(i, layer_savename)
            layer_savenames.append(layer_savename)
            self.temp_files.append(layer_savename)

        self.assertEqual(model.n_layers, len(layer_savenames))

        # Test loading the intermediate models
        #model_export = model._export_pickleable_object()


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestMultilayerPerceptron)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

