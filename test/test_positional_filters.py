import itertools

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase
import safire.data.filters.positional_filters as filters

sentences = [['This', 'is', 'a', 'sentence', '.'],
             ['This', 'is', 'another', 'such', 'sentence', '.'],
             ['This', 'is', 'not', 'just', 'any', 'sentence', '!'],
             ['This', 'sentence', 'is', 'longer', 'than', 'the', 'others', '.'],
             ['I', 'do', 'not', 'like', 'this', 'sentence', '.'],
             ['I', 'think', 'this', 'will', 'be', 'the', 'last', 'one', '.']]


class MyTestCase(SafireTestCase):

    def test_first_k(self):

        first_k = filters.first_k(sentences, 2)
        self.assertEqual(len(first_k), 2)
        self.assertEqual(first_k, sentences[:2])

    def test_words_from_first_k(self):

        words = filters.words_from_first_k(sentences, 1)
        self.assertEqual(len(words), len(sentences))
        self.assertEqual(len(itertools.chain(*words)), 19)





if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(MyTestCase)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
