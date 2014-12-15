import os
import sys
import unittest

import theano.tensor as TT

from safire.datasets.word2vec_transformer import \
    Word2VecSamplingDatasetTransformer
from safire.utils.transcorp import get_id2word_obj
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.word2vec_transformer import Word2VecTransformer

__author__ = 'Jan Hajic jr.'

class TestWord2VecDatasetTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.testdir = os.path.dirname(__file__)
        cls.data_root = os.path.join(cls.testdir, 'test-data')

        cls.loader = MultimodalShardedDatasetLoader(cls.data_root, 'test-data')
        cls.infix = ''

        if not os.getenv('HOME'):
            homepath = os.getenv('USERPROFILE')
        else:
            homepath = os.getenv('HOME')

        cls.w2v_data_root = os.path.join(homepath, 'word2vec')
        cls.edict_pkl_file = os.path.join(cls.w2v_data_root,
                                          'ces_wiki.edict.pkl')
        cls.e_matrix_file = os.path.join(cls.w2v_data_root,
                                         'ces_wiki.emtr.pkl')

    def setUp(self):
        self.corpus = self.loader.load_text_corpus(self.infix)
        self.id2word = get_id2word_obj(self.corpus)
        self.w2v_transformer = Word2VecTransformer(self.edict_pkl_file,
                                                   self.id2word)

    def test_get_batch_sample(self):

        w2v = Word2VecSamplingDatasetTransformer(self.w2v_transformer)
        dataset = self.loader.load_text(self.infix)

        batch = dataset.train_X_batch(0, 2)
        batch_sample = w2v.get_batch_sample(batch)

        self.assertEqual(TT.sum(batch_sample), batch.shape[0])





if __name__ == '__main__':
    unittest.main()
