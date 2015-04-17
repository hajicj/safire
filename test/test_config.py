import os
import collections
import webbrowser
from gensim.interfaces import TransformedCorpus
import pprint
from safire.data.layouts import init_data_root
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.serializer import SwapoutCorpus
from safire.introspection.interfaces import IntrospectionTransformer
from safire.utils.config import ConfigParser, Configuration, ConfigBuilder
from safire.utils.transcorp import dry_run, bottom_corpus, log_corpus_stack

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase


class TestConfig(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestConfig, cls).setUpClass(clean_only, no_datasets)
        cls.config_file = os.path.join(cls.data_root, 'test_config.ini')
        cls.complex_config_file = os.path.join(cls.data_root, 'test_complex_config.ini')
        cls.training_config_file = os.path.join(cls.data_root, 'test_training_config.ini')
        cls.retrieval_config_file = os.path.join(cls.data_root, 'test_retrieval_config.ini')
        cls.t2i_config_file = os.path.join(cls.data_root, 'test_t2i_config.ini')

    def setUp(self):
        self.setUpClass(clean_only=False, no_datasets=False)

    def tearDown(self):
        init_data_root(self.data_root, overwrite=True)

    def test_parser(self):
        parser = ConfigParser()
        with open(self.config_file) as config_handle:
            conf = parser.parse(config_handle)

        # Correct type returned?
        self.assertIsInstance(conf, Configuration)

        # All objects parsed?
        self.assertIsInstance(conf.objects, collections.OrderedDict)
        self.assertEqual(len(conf.objects), 4)
        for name, obj in conf.objects.items():
            self.assertTrue('_class' in obj)
            if '_dependencies' in obj:
                self.assertIsInstance(obj['_dependencies'], list)
            if '_access_deps' in obj:
                self.assertIsInstance(obj['_access_deps'], dict)
                for dep in obj['_access_deps']:
                    self.assertTrue(dep in obj['_dependencies'])

        # Check for presence of special sections
        self.assertTrue(hasattr(conf, '_info'))
        self.assertTrue(hasattr(conf, '_assembly'))
        self.assertTrue(hasattr(conf, '_builder'))
        self.assertTrue(hasattr(conf, '_loader'))
        self.assertTrue(hasattr(conf, '_persistence'))

    def test_builder(self):
        cparser = ConfigParser()
        with open(self.config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        # Testing the special sections
        _info = builder._info
        self.assertTrue(hasattr(_info, 'name'))
        self.assertEqual(_info.name, 'test_config')
        _loader = builder._loader
        self.assertTrue(isinstance(_loader, MultimodalShardedDatasetLoader))
        self.assertEqual(_loader.root, 'test-data/')
        self.assertEqual(_loader.layout.name, 'test-data')
        _assembly = builder._assembly
        self.assertTrue(hasattr(_assembly, '_1_'))
        self.assertEqual(_assembly._1_, 'vtcorp')
        self.assertTrue(hasattr(_assembly, '_2_'))
        self.assertEqual(_assembly._2_, 'tfidf[_1_]')
        self.assertTrue(hasattr(_assembly, '_3_'))
        self.assertEqual(_assembly._3_, 'serializer[_2_]')
        _persistence = builder._persistence
        self.assertTrue(hasattr(_persistence, 'loader'))
        for blockname, label in zip(['_1_', '_2_', '_3_'],
                                    ['vtcorp', 'tfidf', 'tfidf.serialized']):
            self.assertTrue(hasattr(_persistence, blockname))
            self.assertEqual(getattr(_persistence, blockname), label)

        # Testing the dependency graph
        deps_graph = builder.deps_graph
        self.assertEqual(len(deps_graph), 7)
        self.assertTrue(deps_graph['tokenfilter'] == set())
        self.assertTrue(deps_graph['vtcorp'] == set(['tokenfilter']))
        self.assertTrue(deps_graph['_1_'] == set(['vtcorp']))
        self.assertTrue(deps_graph['tfidf'] == set(['_1_']))
        self.assertTrue(deps_graph['_2_'] == set(['_1_', 'tfidf']))
        self.assertTrue(deps_graph['serializer'] == set(['_2_']))
        self.assertTrue(deps_graph['_3_'] == set(['_2_', 'serializer']))

        # Testing object accesibility checks
        self.assertTrue(builder.is_dep_obj_accessible('vtcorp', 'tokenfilter'))
        self.assertTrue(builder.is_dep_obj_accessible('_1_', 'vtcorp'))
        self.assertTrue(builder.is_dep_obj_accessible('_2_', '_1_'))
        self.assertTrue(builder.is_dep_obj_accessible('_2_', 'tfidf'))
        self.assertFalse(builder.is_dep_obj_accessible('tokenfilter', 'vtcorp'))
        self.assertFalse(builder.is_dep_obj_accessible('_1_', '_2_'))
        self.assertFalse(builder.is_dep_obj_accessible('_3_', '_1_'))
        self.assertFalse(builder.is_dep_obj_accessible('serializer', 'vtcorp'))

        self.assertTrue(builder._uses_block('tfidf', '_1_'))
        self.assertTrue(builder._uses_block('serializer', '_2_'))
        self.assertFalse(builder._uses_block('_2_', 'serializer'))

        # Testing the dependency graph sort
        sorted_deps = builder.sorted_deps
        expected_sort_order = ['tokenfilter', 'vtcorp', '_1_', 'tfidf', '_2_',
                               'serializer', '_3_']
        for sorted_key, expected_key in zip(sorted_deps, expected_sort_order):
            self.assertEqual(sorted_key, expected_key)

        # Testing persistence resolution
        will_load, will_be_loaded, will_init = builder.resolve_persistence()
        self.assertEqual(len(will_load), 0)
        self.assertEqual(len(will_be_loaded), 0)
        self.assertEqual(len(will_init), 7)

        self.assertEqual(builder.get_loading_filename('vtcorp'),
                         _loader.pipeline_name('vtcorp'))

        # Testing the build itself
        output_objects = builder.build()
        print output_objects
        self.assertTrue('_3_' in output_objects)
        pipeline = output_objects['_3_']
        self.assertIsInstance(pipeline, SwapoutCorpus)

        # Save the outputs.
        builder.run_saving()

        # Re-test persistence resolution after build+save (i.e. when the files
        # should *all* be available)
        will_load, will_be_loaded, will_init = builder.resolve_persistence()
        self.assertEqual(len(will_load), 1)
        self.assertEqual(len(will_be_loaded), 6)
        self.assertEqual(len(will_init), 0)

    def test_builder_complex(self):
        cparser = ConfigParser()
        with open(self.complex_config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        outputs = builder.build()
        print outputs

        pipeline = outputs['_12_']
        self.assertIsInstance(pipeline, TransformedCorpus)
        introspection = pipeline.obj
        self.assertIsInstance(introspection, IntrospectionTransformer)

        dry_run(pipeline, max=100)
        iid2intro = introspection.iid2introspection_filename
        firstfile = iid2intro[sorted(iid2intro.keys())[0]]
        self.assertTrue(os.path.isfile(firstfile))
        for filename in iid2intro.keys():
            self.assertTrue(os.path.isfile(filename))
        # webbrowser.open(firstfile)

    def test_builder_training(self):
        cparser = ConfigParser()
        with open(self.training_config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        outputs = builder.build()

        self.assertTrue('_4_' in outputs)
        pipeline = outputs['_4_']
        self.assertIsInstance(pipeline, SwapoutCorpus)
        icorp = bottom_corpus(pipeline)
        self.assertIsInstance(icorp, ImagenetCorpus)
        self.assertEqual(len(pipeline), len(icorp))

    def test_builder_retrieval(self):
        cparser = ConfigParser()
        with open(self.retrieval_config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        outputs = builder.build()

        self.assertTrue('_retrieved_' in outputs)
        pipeline = outputs['_retrieved_']
        print log_corpus_stack(pipeline)
        retrieval_results = [x for x in pipeline]
        pprint.pprint(retrieval_results)
        self.assertEqual(len(retrieval_results), len(pipeline))

    def test_builder_t2i(self):
        cparser = ConfigParser()
        with open(self.t2i_config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        outputs = builder.build()

        self.assertTrue('_aggregated_similarities_' in outputs)
        pipeline = outputs['_aggregated_similarities_']
        print log_corpus_stack(pipeline)
        retrieval_results = [x for x in pipeline]
        pprint.pprint(retrieval_results)
        self.assertEqual(len(retrieval_results), len(pipeline))


    def test_autodetect_dependencies(self):

        cparser = ConfigParser()
        with open(self.complex_config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)

        for obj_name, obj in builder.configuration.objects.items():
            deps = builder.deps_graph[obj_name]
            autodeps = builder.autodetect_dependencies(obj)
            self.assertEqual(deps, autodeps)

        with open(self.config_file) as config_handle:
            conf2 = cparser.parse(config_handle)
        builder2 = ConfigBuilder(conf2)
        for obj_name, obj in builder2.configuration.objects.items():
            deps = builder2.deps_graph[obj_name]
            autodeps = builder2.autodetect_dependencies(obj)
            self.assertEqual(deps, autodeps)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestConfig)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
