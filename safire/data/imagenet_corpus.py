#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from gensim import matutils
from gensim.corpora.indexedcorpus import IndexedCorpus

import utils as safutils
from dataset import Dataset
from vtextcorpus import VTextCorpus
from corpus_dataset import UnsupervisedCorpusDataset
from corpus_dataset import UnsupervisedVTextCorpusDataset


class ImagenetCorpus(IndexedCorpus):
    """The ImgnetCorpus is capable of reading the results of running
    images through the ImageNet convolutional network, giving out 4096-dim
    vectors of floats.

    Through a set of
    common document IDs, it's possible to link the image vectors to text
    documents."""
    
    def __init__(self, input):
        """Initializes the ImageNet image vector corpus."""
        pass
