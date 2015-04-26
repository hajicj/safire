"""
This module contains classes that ...
"""
import logging
import copy
import gensim
import itertools
import numpy
import safire.datasets.transformations
import safire.utils.transcorp
from safire.utils import IndexedTransformedCorpus, flatten_composite_item

__author__ = "Jan Hajic jr."


class Zipper(gensim.interfaces.TransformationABC):
    """This transformer combines several pipelines into one. Applying a Zipper
    to a tuple of pipelines creates a CompositeCorpus block.

    While a Zipper instance should theoretically be independent of what it is
    actually zipping and this should be determined at the time of the ``_apply``
    call, in practice, we need to instantiate the zipper with a specific
    combination of pipelines in mind to avoid a multi-parameter ``__getitem__``
    call. Therefore, once a Zipper is created, it already knows its output
    dimension.

    The Zipper works like this::

    >>> x = [[1, 10], [2, 20], [3, 30]]
    >>> y = [[-1, -10], [-2, -20], [-3, -30]]
    >>> z = Zipper((x, y))
    >>> composite = z._apply((x, y))
    >>> composite[0]
    ([1, 10], [-1, -10])
    >>> composite[0:2]
    [([1, 10], [-1, -10]), ([2, 20], [-2, -20])]

    The Zipper by default returns output structured in tuples that
    correspond to the structure of the input combination of pipelines.
    This can be changed using the ``flatten`` parameter::

    >>> z_flatten = Zipper((x, y), flatten=True)
    >>> composite_flat = z_flatten._apply((x, y))
    >>> composite_flat[0:2]
    [[1, 10, -1, -10], [2, 20, -2, -20]]

    **The Zipper expects all its input pipelines to be of the same length.**
    (The :class:`ReorderingCorpus` is useful to meet this condition.)

    Zippers are an essential part of Safire. Do get to know them.
    """
    def __init__(self, corpora, flatten=False, dim=None, names=None):
        # Check that 'corpora' is a tuple
        if not isinstance(corpora, tuple):
            raise TypeError('Input to zipper must be a tuple of corpora, got:'
                            ' {0} with type {1}'.format(corpora, type(corpora)))

        self.flatten = flatten
        self.names = names

        # Generate proposed dim from corpora
        proposed_dim = tuple(safire.utils.transcorp.dimension(c) for c in corpora)
        if dim is not None and dim != proposed_dim:
            logging.warn('Supplied dimension {0} and proposed dimension {1} do '
                         'not match, defaulting to supplied dim.'
                         ''.format(dim, proposed_dim))
        if dim is None:
            dim = proposed_dim

        self.orig_dim = dim

        if self.flatten:
            flat_dim = safire.datasets.transformations.FlattenedDatasetCorpus.flattened_dimension(dim)
            dim = flat_dim
        self.dim = dim

    def __getitem__(self, item):
        # The check for _apply is quite fragile.
        if isinstance(item[0], gensim.interfaces.CorpusABC):
            return self._apply(item)

        if self.flatten:
            if isinstance(item[0], numpy.ndarray):
                # Only works for ndarrays so far, support for gensim not implemented
                output = list(flatten_composite_item(item))
                output = numpy.hstack(output)
            else:
                output = self.flatten_gensim(item, self.orig_dim)
            return output
        else:
            return item

    def _apply(self, corpora):
        return CompositeCorpus(corpora, dim=self.dim, names=self.names)

    @staticmethod
    def flatten_gensim_and_dim(vectors, dim):
        """Flattens the given gensim vectors, using the supplied dim.
        Works on both non-recursive and recursive composite vectors.

        >>> v1 = [(0, 1), (1, 2), (4, 1)]
        >>> v2 = [(1, 1), (2, 4), (3, 1)]
        >>> v3 = [(0, 2), (3, 1)]
        >>> dim = ((5, 5), 4)
        >>> vectors = ((v1, v2), v3)
        >>> Zipper.flatten_gensim_and_dim(vectors, dim)
        ([(0, 1), (1, 2), (4, 1), (6, 1), (7, 4), (8, 1), (10, 2), (13, 1)], 14)

        """
        # logging.warn('Vectors: {0}, dim: {1}'.format(vectors, dim))
        if isinstance(dim, int):
            return vectors, dim
        if len(vectors) != len(dim):
            raise ValueError('Item length {0} and dimension length {1} do not '
                             'match.'.format(vectors, dim))

        output = []
        dim_offset = 0
        for item, d in itertools.izip(vectors, dim):
            to_add, to_add_dim = Zipper.flatten_gensim_and_dim(item, d)

            # logging.warn('To add: {0}, dim_offset: {1}'.format(to_add, dim_offset))
            for key, value in to_add:
                output.append((key + dim_offset, value))
            dim_offset += to_add_dim

        total_dim = dim_offset
        return output, total_dim

    @staticmethod
    def flatten_numpy(vectors):
        flattened_vectors = list(flatten_composite_item(vectors))
        output = numpy.hstack(flattened_vectors)
        return output

    @staticmethod
    def flatten_gensim(vectors, dim):
        """Given a composite item made of gensim vectors and the dimension of the
        composite item, outputs a flattened version."""
        output, total_dim = Zipper.flatten_gensim_and_dim(vectors, dim)
        return output


class CompositeCorpus(IndexedTransformedCorpus):
    """Allows combining pipelines from multiple sources into one, like a more
    flexible version of ``itertools.izip()``. A CompositeCorpus can either be
    created directly, or through a :class:`Zipper` transformer. [NOT
    IMPLEMENTED]

    Also allows naming pipelines (this is useful for train/dev/test splits and
    features/targets splits).

    Initialized with a tuple of pipelines, indexing is available:

    >>> from safire.datasets.dataset import DatasetABC
    >>> features = DatasetABC([[1], [2], [3]], dim=1)
    >>> targets = DatasetABC([[-1], [-2], [-3]], dim=1)
    >>> composite = CompositeCorpus((features, targets), names=('features', 'targets'))
    >>> composite[1:3]
    ([[2], [3]], [[-2], [-3]])
    >>> composite['targets'][:2]
    [[-1], [-2]]
    >>> composite.dim
    (1, 1)

    Can also be recursive:

    >>> recursive = CompositeCorpus((composite, composite), names=('first', 'second'))
    >>> recursive.dim
    ((1, 1), (1, 1))
    >>> recursive[1:3]
    (([[2], [3]], [[-2], [-3]]), ([[2], [3]], [[-2], [-3]]))

    However, it only currently supports building this tree-like structure one
    by one. Trying ``composite = CompositeDataset(((data1, data2), data3))``
    will fail.
    """
    def __init__(self, corpora, dim=None, names=None,
                 aligned=True):
        """Initializes a CompositeCorpus.

        :param corpora:

        :param dim:

        :param names:

        :type aligned: bool
        :param aligned: If set, will expect that all the individual datasets
            from ``corpora`` have the same length. If unset, will not check this
            and advertise the length of the first given dataset as its length;
            only do this if you are flattening the dataset immediately after
            initialization (and using indexes to flatten)!

        """
        self.aligned = aligned
        # Check lengths??
        self.length = len(corpora[0])  # TODO: This is very ugly.

        self.corpus = corpora
        self.obj = None  # No obj so far, waiting to implement ZipPipelines.
        self.chunksize = None

        derived_dim = self.derive_dimension(corpora)
        if dim is None:
            dim = self.derive_dimension(corpora)
        else:
            if dim != derived_dim:
                logging.warn('Supplied dimension {0} inconsistent with '
                             'dimension {1} derived from given corpora. '
                             'Using supplied dimension (and hoping you know '
                             'what you are doing).'
                             ''.format(dim, derived_dim))
        self.dim = dim

        if self.aligned:
            for d in corpora:
                if len(d) != self.length:
                    raise ValueError('All composite corpus components must '
                                     'have the same length. (Lengths: '
                                     '{0}) Are you sure the CompositeCorpus'
                                     'should be aligned?'
                                     ''.format(tuple((len(d) for d in corpora)))
                    )

        if names:
            if len(names) != len(corpora):
                raise ValueError('Corpus names too many or too few'
                                 ' ({0}) for {1} component'
                                 ' corpora.'.format(len(names),
                                                    len(corpora)))
        else:
            names = []
        self.names = names
        self.names_dict = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, item):
        """Retrieval from a composite corpus has several modes:

        >>> from safire.datasets.dataset import DatasetABC
        >>> features = DatasetABC([[1], [2], [3]], dim=1)
        >>> targets = DatasetABC([[-1], [-2], [-3]], dim=1)
        >>> composite = CompositeCorpus((features, targets), names=('features', 'targets'))
        >>> composite[1:3]
        ([[2], [3]], [[-2], [-3]])
        >>> composite.__getitem__((1, 2))
        ([2], [-3])

        """
        try:
            # For retrieving a different index from each data point
            if isinstance(item, tuple):
                return tuple([d[item[i]] for i, d in enumerate(self.corpus)])
            else:
                return tuple([d[item] for d in self.corpus])
        except (TypeError, IndexError):
            if isinstance(item, str):
                return self.corpus[self.names_dict[item]]
            else:
                raise

    def __len__(self):
        # Ugly hack - returns a structure instead of a number... doesn't work
        # with test_p and devel_p, though, so I disabled it for now.
        # if not self.aligned:
        #     return tuple([len(d) for d in self.data])
        return self.length

    @staticmethod
    def derive_dimension(corpora):
        return tuple(safire.utils.transcorp.dimension(d) for d in corpora)

    def as_composite_dim(self, idx):
        """Formats the given index as the composite dimension. Suppose self.dim
        is ``{10, (10, 20), 50)``, then ``as_composite_dim(4)`` will return
        ``(4, (4, 4), 4)``.

        Used as a utility function when creating indexes for flattening an
        aligned CompositeCorpus.

        :param idx: An integer.
        """
        output = []
        for component in self.corpus:
            if isinstance(component, CompositeCorpus):
                output.append(component.as_composite_dim(idx))
            else:
                output.append(idx)
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)


# There's a problem with id2doc.
#
# Flattening a CompositeCorpus: same as flattening a CompositeDataset, as the
# CompositeCorpus already guarantees a dimension.