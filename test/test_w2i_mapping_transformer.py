import unittest
import os
import collections
import numpy
from safire.data.composite_corpus import CompositeCorpus, Zipper
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.data.vtextcorpus import VTextCorpus
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.utils.transcorp import compute_docname_flatten_mapping, \
    get_id2word_obj, token_iid2word, get_id2doc_obj
from safire.utils.transformers import W2IMappingTransformer, \
    GeneralFunctionTransform
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr'

##############################################################################

vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}
vtlist_fname = 'test-data.vtlist'

freqfilter_settings = {'k': 110,
                       'discard_top': 10}

tanh = 0.5

serialization_vtname = 'serialized.vt.shcorp'
serialization_iname = 'serialized.i.shcorp'

##############################################################################


class TestW2IMappingTransformer(SafireTestCase):
    @classmethod
    def setUpClass(cls, clean_only=True, no_datasets=True):
        super(TestW2IMappingTransformer, cls).setUpClass(clean_only=True,
                                                         no_datasets=True)

    def setUp(self):
        # Prepare text pipeline and image pipeline. (Token-based setup, although
        # doc-based setup would be more effective.)
        vtcorp = VTextCorpus(self.vtlist,
                             input_root=self.data_root,
                             tokens=True,
                             **vtcorp_settings)
        vtcorp.dry_run()
        # Need to serialize because the t2i mapping will need to recover an
        # iid2word mapping, so the corpus has to be indexable by iid.
        t_serializer = Serializer(vtcorp, ShardedCorpus,
                                  self.loader.pipeline_serialization_target(
                                      '.tcorp'),
                                  gensim_serialization=True,
                                  gensim_retrieval=True)
        self.vtcorp_serialized = t_serializer[vtcorp]

        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        self.icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')
        self.icorp.dry_run()

        # Here we should prepare the t2i mapping.
        mmcorp = CompositeCorpus((self.vtcorp_serialized, self.icorp),
                                 names=('txt', 'img'),
                                 aligned=False)
        t2i_file = os.path.join(self.data_root,
                                self.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmcorp,
                                                      t2i_file)

        # We'd normally get a composite dataset, but we don't need that, we
        # just need the t2i mapping.
        t2i_indexes_dict = collections.defaultdict(list)
        for t, i in t2i_indexes:
            t2i_indexes_dict[t].append(i)

        # We now need the token_iid --> token map.
        id2word = get_id2word_obj(self.vtcorp_serialized)
        t2i_mapping = {}
        for token_iid, img_iids in t2i_indexes_dict.items():
            # Here is why we needed to serialize the vtcorp:
            token = token_iid2word(token_iid, self.vtcorp_serialized, id2word)
            t2i_mapping[token] = img_iids

        self.w2i = W2IMappingTransformer(t2i_mapping, aggregation='hard')
        self.w2i_soft = W2IMappingTransformer(t2i_mapping, aggregation='soft')

    def test_init(self):

        qtanh = GeneralFunctionTransform(numpy.tanh, multiplicative_coef=0.4)
        query_vtcorp = qtanh[self.vtcorp_serialized]
        item = query_vtcorp[0]

        query_w2i = get_id2word_obj(query_vtcorp)
        self.w2i.set_id2word(query_w2i)
        self.w2i_soft.set_id2word(query_w2i)

        output = self.w2i[item]
        output_soft = self.w2i_soft[item]

        print output
        print output_soft
        word = self.w2i.runtime_id2word[query_vtcorp[0][0][0]]
        qv_id2doc = get_id2doc_obj(query_vtcorp)
        for i, x in enumerate(query_vtcorp):
            if self.w2i.runtime_id2word[x[0][0]] == word:
                print 'Found {0} in doc {1}'.format(word, qv_id2doc[i])

        icorp_id2doc = get_id2doc_obj(self.icorp)
        for i, x in output:
            icorp_doc = icorp_id2doc[i]
            print 'Found {0} with image {1}'.format(word, icorp_doc)



        self.assertEqual(len(output), len(output_soft))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestW2IMappingTransformer)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
