"""
This module contains classes that ...
"""
import logging
import copy
import safire.utils.transcorp
from safire.utils import IndexedTransformedCorpus

__author__ = "Jan Hajic jr."


class CompositeCorpus(IndexedTransformedCorpus):
    """Allows combining pipelines from multiple sources into one, like a more
    flexible version of ``itertools.izip()``. A CompositeCorpus can either be
    created directly, or through a :class:`ZipPipelines` transformer. [NOT
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
        self.length = len(corpora[0])  # TODO: This is very temporary.

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
        # with test_p and devel_p, though, so I disabled it temporarily.
        #if not self.aligned:
        #    return tuple([len(d) for d in self.data])
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



# Flattening a CompositeCorpus: same as flattening a CompositeDataset, as the
# CompositeCorpus already guarantees a dimension.