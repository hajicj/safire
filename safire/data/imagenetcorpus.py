#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import logging
import collections
from gensim import matutils
from gensim.interfaces import CorpusABC
import safire.utils

logger = logging.getLogger(__name__)


class ImagenetCorpus(CorpusABC):
    """The ImgnetCorpus is capable of reading the results of running
    images through the ImageNet convolutional network, giving out 4096-dim
    vectors of floats.

    Through a set of
    common document IDs, it's possible to link the image vectors to text
    documents in a :class:`MultimodalDataset`."""

    def __init__(self, input, delimiter=None, dim=4096, eps=1e-9,
                 doc2id=None, id2doc=None, gzipped=False,
                 include_docnames=None, exclude_docnames=None,
                 label=''):
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

        :param include_docnames: A set of image docnames (like '0000000323.jpg')
            that should be returned. If the given input line starts with
            a docname that is *not* in this set, it will be skipped.

            If a string is supplied, it is interpreted as a filename and the
            docnames are read as lines from the given file.

        :param exclude_docnames: A set of image docnames (like '0000000323.jpg')
            that should not be returned. If the given input line starts with
            a docname that is in this set, it will be skipped.

            If a string is supplied, it is interpreted as a filename and the
            docnames are read as lines from the given file.

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
            doc2id = collections.defaultdict(set)
        if id2doc is None:
            id2doc = collections.defaultdict(str)

        self.doc2id = doc2id
        self.id2doc = id2doc

        if isinstance(include_docnames, str):
            self.include_docnames = safire.utils.file_lines_as_set(include_docnames)
        else:
            self.include_docnames = include_docnames
        if isinstance(exclude_docnames, str):
            self.exclude_docnames = safire.utils.file_lines_as_set(exclude_docnames)
        else:
            self.exclude_docnames = exclude_docnames

        ### DEBUG
        # logging.debug('Include_docnames: {0}'.format(self.include_docnames))
        # logging.debug('Exclude_docnames: {0}'.format(self.exclude_docnames))

        self.n_processed = 0

    def __iter__(self):
        """The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each
        document.
        """
        for i, image in enumerate(self.get_images()):
            logging.debug('__iter__ Yielding image no. {0}'.format(i))
            yield matutils.full2sparse(image, self.eps)

    def reset(self):
        """Sets corpus to "clean" state -- as if it was never iterated over."""
        logging.info('Resetting corpus as if it never iterated.')
        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)
        logging.debug('Old n_processed: {0}'.format(self.n_processed))
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
            logger.debug('Processing image no. {0}'.format(imno))

            docname, feature_str = imline.strip().split('\t', 1)
            if self.exclude_docnames:
                if docname in self.exclude_docnames:
                    logging.debug('Skipping docname {0}: found in '
                                  'self.exlude_docnames'.format(docname))
                    continue
            if self.include_docnames:
                if docname not in self.include_docnames:
                    logging.debug('Skipping docname {0}: not found in '
                                  'self.inlude_docnames'.format(docname))
                    continue

            features = map(float, feature_str.split(self.delimiter))

            if len(features) != self.dim:
                raise ValueError('Invalid input data: data dimension {0}'
                                 ' does not correspond to declared dimension {1}'
                                 ' (on line {2} of input, with docno {3})'
                                 ''.format(len(features), self.dim, imno))

            self.doc2id[docname].add(self.n_processed)
            # Used to be: self.doc2id[docname].add(imno)
            #  but this led to problems when an item was skipped but
            #  imno incremented anyway
            self.id2doc[len(self.id2doc)] = docname

            self.n_processed += 1

            yield features

        if isinstance(self.input, str):
            input_handle.close()

        ### DEBUG
        logging.debug('ImagenetCorpus total processed: {0}'
                      ''.format(self.n_processed))

    def __len__(self):
        return self.n_processed

    def __del__(self):
        if self.__do_cleanup:
            self.input.close()

    def dry_run(self):
        """Iterates through the corpus once."""
        for _ in self:
            pass