#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import logging

from gensim import matutils
from gensim.interfaces import CorpusABC

logger = logging.getLogger(__name__)

class ImagenetCorpus(CorpusABC):
    """The ImgnetCorpus is capable of reading the results of running
    images through the ImageNet convolutional network, giving out 4096-dim
    vectors of floats.

    Through a set of
    common document IDs, it's possible to link the image vectors to text
    documents in a :class:`MultimodalDataset`."""

    def __init__(self, input, delimiter=None, dim=4096, eps=1e-9,
                 doc2id=None, id2doc=None, gzipped=False, label=''):
        """Initializes the ImageNet image vector corpus.

        :param input: The input for an ImagenetCorpus is a handle
            with the ImageNet result file open. Alternately, a filename may be
            supplied.

            The file format is::

              docname [tab] x1;x2;x3;....;x4096

            where ``;`` is the delimiter for the vector values and ``docname``
            is whichever ID the picture is given. This ID is then used when
            pairing images to documents in a multimodal corpus.

        :param delimiter: The delimiter of the vector value columns. If
            left at ``None``, python's default ``split()`` is used.

        :param dim: The dimension of the image vector. Default is 4096.

        :param eps: The minimum required value for a feature to be included
            in the sparse output. Default is 1e-9.

        :param doc2id: If specified, the corpus will use the given map
            from document names to their IDs (order in the corpus).

        :param id2doc: If specified, the corpus will use the given map
            from document IDs (order in the corpus) to their names
            (as given in the input file). While doc2id is a dictionary,
            id2doc is an array.

        :param gzipped: If set to true, expects ``input`` to be a filename
            and the input ImageNet result file to be gzipped.

        :param label: An optional descriptive label of the corpus (could for
            instance describe the dimension, or gzipped state). Used by Loader
            classes to determine what the ImagenetCorpus export file names
            should be.
        """
        self.__do_cleanup = False

        self.input = input

        self.delimiter = delimiter
        self.dim = dim
        self.eps = eps

        self.label = label

        if doc2id is None:
            doc2id = {}
        if id2doc is None:
            id2doc = []

        self.doc2id = doc2id
        self.id2doc = id2doc

        self.n_processed = 0


    def __iter__(self):
        """The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each
        document.
        """
        for i, image in enumerate(self.get_images()):
            logging.debug('__iter__ Yielding image no. %d' % i)
            yield matutils.full2sparse(image, self.eps)

    def reset(self):
        logging.info('Resetting corpus as if it never iterated.')
        self.doc2id = {}
        self.id2doc = []
        logging.debug('Old n_processed: %d' % self.n_processed)
        self.n_processed = 0

    def get_images(self):
        """One iteration of get_images should yield one document, which means
        one line of input.
        """
        input_handle = self.input
        if isinstance(self.input, str):
            input_handle = open(self.input)

        self.reset()

        for imno, imline in enumerate(input_handle):
            logger.debug('Processing image no. %d' % imno)

            docname, feature_str = imline.strip().split('\t', 1)

            features = map(float, feature_str.split(self.delimiter))

            if len(features) != self.dim:
                raise ValueError('Invalid input data: data dimension %d does not correspond to declared dimension %d (on line %d of input, with docno %s)' % (len(features), self.dim, imno))

            self.doc2id[docname] = imno
            self.id2doc.append(docname)

            self.n_processed += 1

            yield features

        if isinstance(self.input, str):
            input_handle.close()

    def __len__(self):
        return self.n_processed

    def __del__(self):
        if self.__do_cleanup:
            self.input.close()
