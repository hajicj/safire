import logging
import os
from safire.data.loaders import MultimodalShardedDatasetLoader, \
    LearnerLoader, ModelLoader
from safire.learning.learners import BaseSGDLearner
from safire.learning.models import MultilayerPerceptron
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr.'

import unittest


class TestBaseSGDLearner(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestBaseSGDLearner, cls).setUpClass()

        cls.dloader = MultimodalShardedDatasetLoader(cls.data_root, 'test-data')
        cls.mloader = ModelLoader(cls.data_root, 'test-data')
        cls.lloader = LearnerLoader(cls.data_root, 'test-data')

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

        super(TestBaseSGDLearner, cls).tearDownClass()

    def setUp(self):

        self.model_handle = MultilayerPerceptron.setup(self.dataset,
                                                       **self.init_args)
        self.learner = BaseSGDLearner(n_epochs=3, b_size=2, validation_frequency=4)

    def tearDown(self):

        self.learner.clear_intermediate()

    def test_intermediate_save_on_training(self):

        pre_training_weights = [ layer.W.get_value(borrow=True)[0,0]
                        for layer in self.model_handle.model_instance.layers]


        self.learner.set_saving(self.mloader, save_every=1, infix=None)

        self.learner.run(self.model_handle, self.dataset)

        post_training_weights = [ layer.W.get_value(borrow=True)[0,0]
                        for layer in self.model_handle.model_instance.layers]

        self.assertNotEqual(pre_training_weights, post_training_weights)
        # Have the intermediate files been created?
        for fname in self.learner.intermediate_fnames:
            self.assertTrue(os.path.isfile(fname))

        self.learner.run(self.model_handle, self.dataset, resume=True)

        self.assertEqual(6, len(self.learner.intermediate_fnames))

    def test_learner_saveload(self):

        self.learner.set_saving(self.mloader, save_every=1, infix=None)
        self.learner.run(self.model_handle, self.dataset)

        self.lloader.save_learner(self.learner, '.test_label')

        loaded_learner = self.lloader.load_learner('.test_label')

        loaded_learner.run(self.model_handle, self.dataset, resume=True,
                           force_resume=True)

        self.assertEqual(6, len(loaded_learner.intermediate_fnames))


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestBaseSGDLearner)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

