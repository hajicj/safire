#!/usr/bin/env python
"""This module provides facilities for loading data as Dataset objects.

The loaders work by first processing data into gensim-style corpora,
then providing access to the serialized corpora using the corpus dataset
classes defined in the :mod:`corpus_dataset` module.

The loaders all expect the data dir layout defined by the :class:`DataDirLayout`
class.

All loaders are initialized by a root and name. The root is the data directory
where the loader should operate, the name is the dataset name (used in various
prefixes). This scheme allows holding multiple selections from one set of raw
data, although it can lead to clutter. However, aside from elementary files
preparation (see the :class:`DataDirLayout` documentation), the loaders should
handle all further communication with the dataset(s).

The individual experiments in a dataset and the corresponding data are
distinguished by user-supplied labels that serve as infixes within a standard
naming scheme (again defined by the layout class).

There are currently four loaders available: one for multimodal datasets,
one for models/transformers, one for similarity indexes and one for sharded
datasets (see :class:`ShardedDataset`).
"""

import copy
import logging
import os

import gensim
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities import Similarity

from safire.datasets.corpus_dataset import UnsupervisedVTextCorpusDataset, \
    UnsupervisedImagenetCorpusDataset
from safire.datasets.sharded_dataset import ShardedDataset
from safire.datasets.sharded_multimodal_dataset import ShardedMultimodalDataset, \
    UnsupervisedShardedImagenetCorpusDataset, \
    UnsupervisedShardedVTextCorpusDataset
from safire.learning.interfaces import ModelHandle
from safire.learning.interfaces.safire_transformer import SafireTransformer
from safire.learning.learners import BaseSGDLearner
from safire.learning.models.base_model import BaseModel
from safire.data.vtextcorpus import VTextCorpus
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.multimodal_dataset import MultimodalDataset
from .layouts import DataDirLayout


logger = logging.getLogger(__name__)

default_label = None

#: Default image corpus arguments. Works for the UFAL ImageNet vector inputs.
default_icorp_args = {
    'delimiter': ';',
    'dim': 4096,
    'label': default_label}
# TODO: Move these defaults to somewhere more principled.

#: Default text corpus arguments. They are already encoded within
#  the VTextCorpus __init__() default values.
default_vtcorp_args = {}


class MultimodalDatasetLoader(object):
    """A class that works as an interface for loading a multimodal dataset
    using a :class:`VTextCorpus` and :class:`ImagenetCorpus`.

        >>> loader = MDloader('/path/to/data/root', name)
        >>> safire_multimodal_dataset = loader.load()

    The ``name`` parameter is a name of the dataset. **It has to reflect the
    names used in the data directory itself;** the dependency is defined by
    the :class:`DataDirLayout` class. Currently, the name is expected in the
    following places in the data root:

    * The ``vtlist`` file passed to VTextCorpus to operate on has the name
      ``name.vtlist``

    * The corpus file names start with ``name``

    Typically, ``name`` for a ``/path/to/data/root`` would be ``root``.

    Loading relies on the gensim-style corpora already being generated. If not
    present, they first need to be generated. Because that is a time-consuming
    procedure, the loader class will not generate them automatically. Instead,
    the following workflow is available:

        >>> loader = MDLoader('/path/to/data/root', name)
        >>> loader.has_corpora()
        False
        >>> dataset = loader.load()
        ValueError: Corpora unavailable in SAFIRE dataset at /path/to/safire/root
        >>> loader.generate_corpora(safire_root)
        [...takes a long time..]
        >>> loader.has_corpora()
        True
        >>> dataset = loader.load()

    All loading/corpus-building/corpus-checking functionality can further be
    called separately for text and image data.

    """

    def __init__(self, root, name, sentences=False, text_loader=VTextCorpus,
                 img_loader=ImagenetCorpus, text_serializer=MmCorpus,
                 img_serializer=MmCorpus):
        """Initializes the mulitmodal dataset loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are the text/ and img/ directories, a ``*.vtlist`` file,
            an Imagenet output file in the format described in
            :class:`ImagenetCorpus` and a text-image mapping as described in
            :class:`MultimodalDataset`.

            By default, the output corpora will be serialized/saved into a
            ``corpora/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        :type sentences: bool
        :param sentences: Defines whether sentences should be used as documents
            in corpusifying the text data

        :type text_loader: gensim.interfaces.CorpusABC
        :param text_loader: Corpus class used to load the text data.

        :type img_loader: gensim.interfaces.CorpusABC
        :param img_loader: Corpus class used to load the image data.
        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' . root)

        self.root = root

        self.sentences = sentences
        self.layout = DataDirLayout(name)

        # Default serializers.
        self.text_serializer = text_serializer
        self.img_serializer = img_serializer

        self._verify_layout()

        if not text_loader:
            text_loader = VTextCorpus

        if not img_loader:
            img_loader = ImagenetCorpus

        self.text_loader = text_loader
        self.img_loader = img_loader

    def _verify_layout(self):
        """Check that the SAFIRE root given at initialization contains all
        required directories and files. Only called at initialization.

        Does not check for corpora.

        Will raise a ``ValueError`` if the required components are missing,
        instead of returning True/False, because a valid SAFIRE object cannot
        exist unless the requirements checked by ``_verify`` are met.

        :raises: ValueError
        """
        files = frozenset(os.listdir(self.root))

        if self.layout.img_dir not in files:
            raise ValueError("Image dir %s missing in dataset root (%s), available: %s" % (
                                                 self.layout.img_dir,
                                                 self.root,
                                                 str(files)))
        if self.layout.text_dir not in files:
            raise ValueError('Text dir %s missing in dataset root (%s), available:\n%s' % (
                                                 self.layout.text_dir,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.vtlist not in files:
            raise ValueError('Vtlist %s missing in dataset root (%s), available:\n%s' % (
                                                 self.layout.vtlist,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.image_vectors not in files:
            raise ValueError('Image vectors %s missing in dataset root (%s), available:\n%s' % (
                                                 self.layout.image_vectors,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.textdoc2imdoc not in files:
            raise ValueError('Vtlist %s missing in dataset root (%s), available:\n%s' % (
                                                 self.layout.textdoc2imdoc,
                                                 self.root,
                                                 '\n'.join(files)))

    def has_corpora(self, infix=None):
        """Checks whether corpora for loading the SAFIRE multimodal dataset
        have been generated and are in the right place."""
        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)
        files = os.listdir(corpus_dir)

        for corpus in self.layout.required_corpus_names(infix):
            if corpus not in files:
                logger.info('Corpus %s not found in corpus directory %s.' % (
                            corpus,
                            corpus_dir))
                logger.debug('   Available :\n%s' % '\t\t\n'.join(files))
                return False

        return True

    def has_text_corpora(self, infix=None):
        """Checks whether text corpora for loading the given multimodal dataset
        have been generated and are in the right place.

        Analogous methods for the text and image corpora separately are provided
        as ``has_text_corpora()`` and ``has_img_corpora()``.
        """
        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)
        files = os.listdir(corpus_dir)

        for corpus in self.layout.required_text_corpus_names(infix):
            if corpus not in files:
                logger.info('Corpus %s not found in corpus directory %s.' % (
                            corpus,
                            corpus_dir))
                logger.debug('   Available :\n%s' % '\t\t\n'.join(files))
                return False

        return True

    def get_text_corpus(self, vtext_corpus_args=None):
        """Returns the VTextCorpus object correctly initialized to the Loader's
        layout. As opposed to ``build_text_corpus``, does NOT perform
        serialization, only creates the corpus bound to the loader's data dir.

        :type vtext_corpus_args: dict
        :param vtext_corpus_args: A dictionary passed as kwargs to the
            constructor of ``VTextCorpus``. If there is an ``input_root``
            member, it will be ignored.

            .. note::

                The text corpus file names will have an infix based on the
                ``label`` member of this dictionary (if present).

        :rtype: safire.data.vtextcorpus.VTextCorpus
        :returns: A :class:`VTextCorpus` initialized according to the loader's
            layout.
        """
        if vtext_corpus_args is None:
            vtext_corpus_args = {}

        # Watch out for args root mismatch!
        if 'input_root' in vtext_corpus_args \
                and vtext_corpus_args['input_root'] != self.root:
            logging.warn(
                'Building corpora: mismatched vtext args input root (%s) and loader input root (%s).' % (
                vtext_corpus_args['input_root'], self.root))

        # When filling in input root, we don't want a side-effect to occur
        # to the input argument.
        vtargs = copy.deepcopy(vtext_corpus_args)
        vtargs['input_root'] = self.root

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        # Text corpus building/serialization
        full_vtlist_path = os.path.join(self.root, self.layout.vtlist)
        text_corpus = VTextCorpus(full_vtlist_path, **vtargs)

        return text_corpus

    def get_default_text_corpus(self):
        """Returns the default image corpus initialized to the loader's layout.
        """
        return self.get_text_corpus(default_vtcorp_args)

    def has_image_corpora(self, infix=None):
        """Checks whether text corpora for loading the given multimodal dataset
        have been generated and are in the right place.

        Analogous methods for the text and image corpora separately are provided
        as ``has_text_corpora()`` and ``has_img_corpora()``.
        """
        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)
        files = os.listdir(corpus_dir)

        for corpus in self.layout.required_img_corpus_names(infix):
            if corpus not in files:
                logger.info('Corpus %s not found in corpus directory %s.' % (
                            corpus,
                            corpus_dir))
                logger.debug('   Available :\n%s' % '\t\t\n'.join(files))
                return False

        return True

    def get_image_corpus(self, image_corpus_args=None):
        """Returns the ImagenetCorpus object correctly initialized to the Loader's
        layout. As opposed to ``build_image_corpus``, does NOT perform
        serialization.

        :type image_corpus_args: dict
        :param image_corpus_args: A dictionary passed as kwargs to the
            constructor of ``VTextCorpus``. If there is an ``input_root``
            member, it will be ignored.

            .. note::

                The text corpus file names will have an infix based on the
                ``label`` member of this dictionary (if present).

        :rtype: safire.data.vtextcorpus.VTextCorpus
        :returns: A :class:`VTextCorpus` initialized according to the loader's
            layout.
        """
        if image_corpus_args is None:
            image_corpus_args = {}

        # When filling in input root, we don't want a side-effect to occur
        # to the input argument.
        imargs = copy.deepcopy(image_corpus_args)

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        # Text corpus building/serialization
        full_ivectors_path = os.path.join(self.root, self.layout.image_vectors)
        img_corpus = ImagenetCorpus(full_ivectors_path, **imargs)

        return img_corpus

    def get_default_image_corpus(self):
        """Returns the default image corpus initialized to the loader's layout.
        """
        return self.get_image_corpus(default_icorp_args)

    def build_image_corpora(self, img_corpus_args, serializer=None):
        """Creates all *image* corpora necessary for the creation of the
        :class:`MultimodalDataset` over the given data.

        The corpora will be located in ``self.layout.corpus_dir`` in the data
        directory. There will be two corpora, three files in total:

        * ``MmCorpus`` for image data

        * The associated corpus index for image data

        * ``ImagenetCorpus`` for image metadata

        :type img_corpus_args: dict
        :param img_corpus_args: A dictionary passed as kwargs to the
            constructor of ``ImagenetCorpus``. If there is an ``input_root``
            member, it will be ignored.

            .. note::

                The image corpus file names will have an infix based on the
                ``label`` member of this dictionary (if present).

        :type serializer: class froom gensim.interfaces.CorpusABC
        :param serializer: The corpus class that should be used to
            serialize the corpus data. Defaults to ``img_serializer`` passed
            in ``__init__``.

        """
        if not serializer:
            serializer = self.img_serializer

        # When filling in input root, we don't want a side-effect to occur
        # to the input argument.
        imargs = copy.deepcopy(img_corpus_args)

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        # Image corpus building/serialization
        img_full_path = os.path.join(self.root, self.layout.image_vectors)

        img_corpus = ImagenetCorpus(img_full_path, **imargs)
        img_name_infix = self.generate_icorp_name_infix(img_corpus)

        if self.has_image_corpora(img_name_infix):
            logging.warn('Text corpora for given infix %s exist; overwriting.' % img_name_infix)

        img_corpus_names = self.layout.required_img_corpus_names(img_name_infix)
        img_data_name = img_corpus_names[0]
        img_obj_name = img_corpus_names[2]

        serializer.serialize(os.path.join(corpus_dir, img_data_name),
                             img_corpus)

        img_corpus.save(os.path.join(corpus_dir, img_obj_name))

    def build_default_image_corpora(self, serializer=None):
        """Builds (incl. serialization) the default image corpus initialized
        to the loader's layout. You still have to choose the serializer."""
        self.build_image_corpora(default_icorp_args,
                                 serializer=serializer)

    def build_text_corpora(self, vtext_corpus_args, serializer=None):
        """Creates all *text* corpora necessary for the creation of the
        :class:`MultimodalDataset` over the given data.

        The corpora will be located in ``self.layout.corpus_dir`` in the data
        directory. There will be four corpora, six files in total:

        * ``MmCorpus`` for text data

        * The associated corpus index for text data

        * ``VTextCorpus`` for text data

        .. note::

          Is NOT capable of doing frequency filtering and tf-idf transform,
          because these are *transformations*, not VTextCorpus properties.

        :type vtext_corpus_args: dict
        :param vtext_corpus_args: A dictionary passed as kwargs to the
            constructor of ``VTextCorpus``. If there is an ``input_root``
            member, it will be ignored.

            .. note::

                The text corpus file names will have an infix based on the
                ``label`` member of this dictionary (if present).

        :type serializer: class froom gensim.interfaces.CorpusABC
        :param serializer: The corpus class that should be used to
            serialize the corpus data. Defaults to ``text_serializer`` passed
            in ``__init__``.

        """
        if not serializer:
            serializer = self.text_serializer

        # Watch out for args root mismatch!
        if 'input_root' in vtext_corpus_args \
                and vtext_corpus_args['input_root'] != self.root:
            logging.warn(
                'Building corpora: mismatched vtext args input root (%s) and loader input root (%s).' % (
                    vtext_corpus_args['input_root'], self.root))

        # When filling in input root, we don't want a side-effect to occur
        # to the input argument.
        vtargs = copy.deepcopy(vtext_corpus_args)
        vtargs['input_root'] = self.root

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)
        # Text corpus building/serialization
        vtlist_full_path = os.path.join(self.root, self.layout.vtlist)

        text_corpus = VTextCorpus(vtlist_full_path, **vtargs)
        text_name_infix = self.generate_tcorp_name_infix(text_corpus)

        if self.has_text_corpora(text_name_infix):
            logging.warn('Text corpora for given infix %s exist; overwriting.' % text_name_infix)

        text_corpus_names = self.layout.required_text_corpus_names(text_name_infix)
        text_data_name = text_corpus_names[0]
        text_obj_name = text_corpus_names[2]

        serializer.serialize(os.path.join(corpus_dir, text_data_name),
                             text_corpus)

        text_corpus.save(os.path.join(corpus_dir, text_obj_name))

    def build_default_text_corpora(self, serializer=None):
        """Builds (incl. serialization) the default image corpus initialized
        to the loader's layout. You still have to choose the serializer."""
        self.build_text_corpora(default_vtcorp_args,
                                serializer=serializer)

    def build_corpora(self, vtext_corpus_args={}, img_corpus_args={},
                      text_serializer=None, img_serializer=None):
        """Creates all corpora necessary for the creation of the
        MultimodalDataset over the given data.

        The corpora will be located in ``self.layout.corpus_dir`` in the data
        directory. There will be four corpora, six files in total:

        * ``MmCorpus`` for text data

        * The associated corpus index for text data

        * ``VTextCorpus`` for text data

        And an analogous trio for image data, with an ``ImagenetCorpus``
        instead of a ``VTextCorpus`` holding the image corpus metadata.

        Building text corpora only or image corpora only can be handled
        using :func:`build_text_corpora` or :func:`build_image_corpora`.

        :type vtext_corpus_args: dict
        :param vtext_corpus_args: A dictionary passed as kwargs to the
            constructor of ``VTextCorpus``. If there is an ``input_root``
            member, it will be ignored.

            .. note::

                The text corpus file names will have an infix based on the
                ``label`` member of this dictionary (if present).

        :type img_corpus_args: dict
        :param img_corpus_args: A dictionary passed as kwargs to the constructor
            of ``ImagenetCorpus``.

            .. note::

                The image corpus file names will have an infix based on the
                ``label`` member of this dicitonary (if present).

        :type text_serializer: class froom gensim.interfaces.CorpusABC
        :param text_serializer: The corpus class that should be used to
            serialize the text corpus data. Defaults to ``text_serializer``
            passed in ``__init__``.

        :type img_serializer: class froom gensim.interfaces.CorpusABC
        :param img_serializer: The corpus class that should be used to
            serialize the image corpus data. Defaults to ``img_serializer``
            passed in ``__init__``.
        """

        self.build_text_corpora(vtext_corpus_args, text_serializer)

        self.build_image_corpora(img_corpus_args, img_serializer)

    def load(self, infix=None, text_infix=None, img_infix=None,
             text_serializer=MmCorpus, img_serializer=MmCorpus):
        """Creates the SAFIRE MultimodalDataset. If corpora are not generated,
        raises a ValueError.

        :type infix: str
        :param infix: The infix (corpus label) of the desired text
            and image corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags or scaling down image
            dimensionality.)

            If ``infix`` is supplied, ``text_infix`` and ``img_infix`` are
            overriden by ``infix``.

        :type text_infix: str
        :param text_infix: The infix (corpus label) of the desired text
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type img_infix: str
        :param img_infix: The infix (corpus label) of the desired image
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type text_serializer: class froom gensim.interfaces.CorpusABC
        :param text_serializer: The corpus class that should be used to
            de-serialize the given text corpus data. Defaults to
            ``text_serializer`` passed in ``__init__``.

        :type img_serializer: class froom gensim.interfaces.CorpusABC
        :param img_serializer: The corpus class that should be used to
            de-serialize the given image corpus data. Defaults to
            ``img_serializer`` passed in ``__init__``.

        :rtype: safire.data.multimodal_dataset.MultimodalDataset
        :returns: A :class:`MultimodalDataset` representation of the
            dataset.

        :raises: ValueError
        """
        if not text_serializer:
            text_serializer = self.text_serializer
        if not img_serializer:
            img_serializer = self.img_serializer

        if infix:
            text_infix = infix
            img_infix = infix

        if not self.has_text_corpora(text_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))

        if not self.has_image_corpora(img_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))


        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        text_corpora_fnames = self.layout.required_text_corpus_names(text_infix)
        text_data = os.path.join(corpus_dir, text_corpora_fnames[0])
        text_obj = os.path.join(corpus_dir, text_corpora_fnames[2])
            # Skipping the '.mmcorp.index' file

        img_corpora_fnames = self.layout.required_img_corpus_names(img_infix)
        img_data = os.path.join(corpus_dir, img_corpora_fnames[0])
        img_obj = os.path.join(corpus_dir, img_corpora_fnames[2])

        textdoc2imdoc = os.path.join(self.root, self.layout.textdoc2imdoc)

        dataset = MultimodalDataset(text_data, text_obj, img_data, img_obj,
                                    aligned=False,
                                    textdoc2imdoc=textdoc2imdoc,
                                    text_serializer=self.text_serializer,
                                    img_serializer=self.img_serializer)

        return dataset

    def load_text(self, text_infix=None, dataset_init_args={}):
        """Loads a text-only dataset using the class
        :class:`UnsupervisedVTextCorpusDataset`.

        :type text_infix: str
        :param text_infix: The infix (corpus label) of the desired text
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type dataset_init_args: dict
        :param dataset_init_args: Further init args that will be passed to the
            dataset constructor.

        :rtype: safire.data.corpus_dataset.UnsueprvisedVTextCorpusDataset
        :returns: The :class:`UnsupervisedVTextCorpusDataset` built from the
            given text corpora.

        """
        if not self.has_text_corpora(text_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        text_corpora_fnames = self.layout.required_text_corpus_names(text_infix)
        text_data = os.path.join(corpus_dir, text_corpora_fnames[0])
        text_obj = os.path.join(corpus_dir, text_corpora_fnames[2])
            # Skipping the '.mmcorp.index' file

        dataset = UnsupervisedVTextCorpusDataset(text_data, text_obj,
                                                 **dataset_init_args)

        return dataset

    def load_img(self, img_infix=None, dataset_init_args={}):
        """Loads an image-only dataset using the class
        :class:`UnsupervisedImagenetCorpusDataset`.

        :type img_infix: str
        :param img_infix: The infix (corpus label) of the desired text
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type dataset_init_args: dict
        :param dataset_init_args: Further init args that will be passed to the
            dataset constructor.

        :rtype: safire.data.corpus_dataset.UnsueprvisedImagenetCorpusDataset
        :returns: The :class:`UnsupervisedImagenetCorpusDataset` built from the
            given text corpora.

        """
        if not self.has_image_corpora(img_infix):
            raise ValueError('Image corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), img_infix))


        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        img_corpora_fnames = self.layout.required_img_corpus_names(img_infix)
        img_data = os.path.join(corpus_dir, img_corpora_fnames[0])
        img_obj = os.path.join(corpus_dir, img_corpora_fnames[2])

        logging.debug('Loading UnsupervisedImagenetCorpusDataset with'
                      ' data %s, obj %s' % (img_data, img_obj))

        dataset = UnsupervisedImagenetCorpusDataset(img_data, img_obj,
                                                    **dataset_init_args)

        return dataset

    def generate_icorp_name_infix(self, img_corpus):
        """Retrieves the infix of the standard name given by the Loader to the
        image corpus it builds.

        The label was supplied manually to the corpus using the ``label``
        init argument."""
        return img_corpus.label

    def generate_tcorp_name_infix(self, text_corpus):
        """Retrieves the infix of the standard name given by the Loader to the
        text corpus it builds.

        The label was supplied manually to the corpus using the ``label``
        init argument."""
        return text_corpus.label

    def has_text_corpus(self, infix=None):
        """Checks whether a saved text corpus with the given infix exists
        in the data dir.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)
        corpus_file = self.layout.get_text_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        return os.path.isfile(corpus_full_path)

    def save_text_corpus(self, corpus, infix=None):
        """Saves a VTextCorpus object to the data dir.

        :type corpus: VTextCorpus
        :param corpus: The corpus which should be saved.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if self.has_text_corpus(infix):
            logging.warn('Overwriting text corpus for infix %s.' % infix)

        corpus_file = self.layout.get_text_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        corpus.save(corpus_full_path)

    def serialize_text_corpus(self, corpus, infix=None):
        """Serializes the contents of an image corpus.

        :type corpus: ImagenetCorpus or other corpus
        :param corpus: The corpus which should be saved.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if self.has_text_corpus(infix):
            logging.warn('Overwriting image corpus for infix %s.' % infix)

        corpus_file = self.layout.required_text_corpus_names(infix)[0]
        corpus_full_path = os.path.join(self.root, self.layout.corpus_dir,
                                        corpus_file)

        self.text_serializer.serialize(corpus_full_path, corpus)

    def load_text_corpus(self, infix=None):
        """Loads a VTextCorpus object previously saved in the data dir.

        :type infix: str
        :param infix: The infix (model label) of the desired model.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if not self.has_text_corpus(infix):
            available_corpora = os.listdir(os.path.join(self.root,
                                                        self.layout.corpus_dir))
            raise ValueError('Text corpus in loader name %s for infix %s not available. Files:\n%s' % (self.layout.name, infix, '\n'.join(available_corpora)))

        corpus_file = self.layout.get_text_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        corpus = gensim.utils.SaveLoad.load(corpus_full_path)

        return corpus

    def has_image_corpus(self, infix=None):
        """Checks whether a saved image corpus with the given infix exists
        in the data dir.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)
        corpus_file = self.layout.get_image_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        return os.path.isfile(corpus_full_path)

    def save_image_corpus(self, corpus, infix=None):
        """Saves a ImagenetCorpus object to the data dir.

        :type corpus: ImagenetCorpus
        :param corpus: The corpus which should be saved.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if self.has_image_corpus(infix):
            logging.warn('Overwriting image corpus for infix %s.' % infix)

        corpus_file = self.layout.get_image_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        corpus.save(corpus_full_path)

    def serialize_image_corpus(self, corpus, infix=None):
        """Serializes the contents of an image corpus.

        :type corpus: ImagenetCorpus or other corpus
        :param corpus: The corpus which should be saved.

        :type infix: str
        :param infix: The desired infix (model label) of the corpus.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if self.has_image_corpus(infix):
            logging.warn('Overwriting image corpus for infix %s.' % infix)

        corpus_file = self.layout.required_img_corpus_names(infix)[0]
        corpus_full_path = os.path.join(self.root, self.layout.corpus_dir,
                                        corpus_file)

        self.img_serializer.serialize(corpus_full_path, corpus)

    def load_image_corpus(self, infix=None):
        """Loads an ImagenetCorpus object previously saved in the data dir.

        :type infix: str
        :param infix: The infix (model label) of the desired model.
            (Infixes serve to differentiate saved corpora
            obtained with different methods - part of speech filters,
            frequency filters, etc.)
        """
        infix = self.__default_infix(infix)

        if not self.has_image_corpus(infix):
            raise ValueError('Image corpus '
                             'for infix %s not available.' % infix)

        corpus_file = self.layout.get_image_corpus_file(infix)
        corpus_full_path = os.path.join(self.root, corpus_file)

        # This should really be an ImagenetCorpus, but just in case...
        corpus = ImagenetCorpus.load(corpus_full_path)

        return corpus

    @classmethod
    def __default_infix(cls, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix


class ModelLoader(object):
    """This class is responsible for loading models (and objects acting like
    models) from a data directory. It mirrors the function of
    :class:`MultimodalDatasetLoader`, but is much less complicated, as it
    doesn't have to deal with multiple synchronized corpora, required files,
    etc.

    The primary way of retrieving models is loading handles:

    >>> mLoader = ModelLoader('path/to/dataset/root', 'name')
    >>> model_handle = mLoader.load_handle('.NADV.top10000.sda3')

    This will load a model handle built from a corpus filtered for autosemantic
    parts of speech and limited to the 10000 most frequent lemmas, with a model
    trained as a Stacked Denoising Autoencoder no. 3 in some experimental setup
    against the default image corpus for the dataset (assuming the descriptors
    are sensible - the developer is responsible for naming the corpora and
    models appropriately).

    For completeness, you can also retrieve models without handles:

    >>> model = mLoader.load_model('.NADV.top1000.sda3')

    and build a handle later:

    >>> model_handle = model.__class__.setup(dataset, ...)

    Handles and models can also be saved and loaded:

    >>> mLoader.save_model(model, '.NADV.top1000.sda3')
    >>> mLoader.save_handle(model_handle, '.NADV.top1000.sda3')

    >>> loaded_model = mLoader.load_model('.NADV.top1000.sda3')
    >>> loaded_handle = mLoader.load_handle('.NADV.top1000.sda3')

    and so can SafireTransformers:

    >>> transformer = SafireTransformer(model_handle)
    >>> mLoader.save_transformer(transformer, '.NADV.top1000.sda3')
    >>> loaded_transformer = mLoader.load_transformer('.NADV.top1000.sda3')

    The naming conventions and directories within a data directory are described
    by the :class:`DataDirLayout` class.
    """
    def __init__(self, root, name):
        """Initializes the model loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are described by the :class:`DataDirLayout` class.

            By default, the output corpora will be serialized/saved into a
            ``models/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' . root)

        self.root = root
        self.layout = DataDirLayout(name)

        self._verify_layout()

    def _verify_layout(self):
        """Tests whether the data dir has the required layout for model/model
        handle loading."""

        files = os.listdir(self.root)

        if self.layout.model_dir not in files:
            raise ValueError('Missing model directory %s in dataset root (%s)' %
                             (self.layout.model_dir, self.root))

    def model_full_path(self, infix=None):
        """Returns the full path to a model file for the given infix.
        (The model itself doesn't have to exist.)"""
        return os.path.join(self.root,
                            self.layout.get_model_file(infix))

    def has_model(self, infix=None):
        """Checks whether a model with the given infix exists.

        :type infix: str
        :param infix: The model infix to check (like: ``.NADV.top1000.sda3``)
        """
        infix = self.__default_infix(infix)

        return self.__has('m', infix)

    def save_model(self, model, infix):
        """Saves a model instance to file."

        :type model: safire.learning.models.BaseModel
        :param model: A model to save in the data dir.

        :type infix: str
        :param infix: The infix (model label) of the model to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_model(infix):
            logging.warn('Overwriting existing model: %s' %
                         self.layout.get_model_name(infix))

        model_file = self.layout.get_model_file(infix)
        model_full_path = os.path.join(self.root, model_file)

        model.save(model_full_path)

    def save_model_layer(self, model, layer, infix):
        """Saves a layer from the given model with the given infix.
        """
        infix = self.__default_infix(infix)

        if self.has_model(infix):
            logging.warn('Overwriting existing model: %s' %
                         self.layout.get_model_name(infix))

        model_file = self.layout.get_model_file(infix)
        model_full_path = os.path.join(self.root, model_file)

        model.save_layer(layer, model_full_path)

    def load_model(self, infix=None):
        """Loads a model object for the dataset with the specified infix.

        We do not know the class of the model in advance, so we use the hack
        in the models' ``save`` mechanism to derive the class from the pickled
        object itself.

        :type infix: str
        :param infix: The infix (model label) of the desired model.
            (Infixes serve to differentiate saved models
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_model(infix):
            raise ValueError('Cannot find model %s with infix %s in data dir %s.' % (
                self.layout.get_model_name(infix), infix, self.root))

        model_file = self.layout.get_model_file(infix)
        model_full_path = os.path.join(self.root, model_file)

        pickled_model = BaseModel._load_pickleable_object(model_full_path)

        model_class = pickled_model['class']
        model_init_args = pickled_model['init_args']

        model = model_class(**model_init_args)

        return model

    def has_handle(self, infix=None):
        """Checks whether a model handle with the given infix exists.

        :type infix: str
        :param infix: The model handle infix to check
            (like: ``.NADV.top1000.sda3``)
        """
        infix = self.__default_infix(infix)

        return self.__has('h', infix)

    def save_handle(self, model_handle, infix):
        """Saves a model instance to file."

        :type model_handle: safire.learning.interfaces.ModelHandle
        :param model_handle: A model handle to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the model handle to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_handle(infix):
            logging.warn('Overwriting existing handle: %s' %
                         self.layout.get_handle_name(infix))

        handle_file = self.layout.get_handle_file(infix)
        handle_full_path = os.path.join(self.root, handle_file)

        model_handle.save(handle_full_path)

    def load_handle(self, infix=None):
        """Loads a model handle object for the dataset with the specified infix.

        :type infix: str
        :param infix: The infix (model label) of the desired model.
            (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_handle(infix):
            raise ValueError('Cannot find handle %s with infix %s in data dir %s.' % (
                self.layout.get_handle_name(infix), infix, self.root))

        handle_file = self.layout.get_handle_file(infix)
        handle_full_path = os.path.join(self.root, handle_file)

        handle = ModelHandle.load(handle_full_path)

        return handle

    def has_transformer(self, infix=None):
        """Checks whether a transformer with the given infix exists.

        :type infix: str
        :param infix: The transformer infix to check (like:
            ``.NADV.top1000.sda3``)
        """
        infix = self.__default_infix(infix)

        return self.__has('t', infix)

    def save_transformer(self, transformer, infix=None):
        """Saves a model instance to file."

        :type transformer: safire.learning.interfaces.SafireTransformer
        :param transformer: A transformer to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the transformer to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_transformer(infix):
            logging.warn('Overwriting existing transformer: %s' %
                         self.layout.get_transformer_name(infix))

        transformer_file = self.layout.get_transformer_file(infix)
        transformer_full_path = os.path.join(self.root, transformer_file)

        transformer.save(transformer_full_path)

    def load_transformer(self, infix=None):
        """Loads a model handle object for the dataset with the specified infix.

        :type infix: str
        :param infix: The infix (model label) of the desired model.
            (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_transformer(infix):
            raise ValueError('Cannot find transformer %s with infix %s in data dir %s.' % (
                self.layout.get_transformer_name(infix), infix, self.root))

        transformer_file = self.layout.get_transformer_file(infix)
        transformer_full_path = os.path.join(self.root, transformer_file)

        transformer = SafireTransformer.load(transformer_full_path)

        return transformer

    def __has(self, obj_type, infix):
        """Checks whether a given saved object exists. Shortcut for checking
        models, handles and transformers separately.

        :type obj_type: str
        :param obj_type: 'm', 'h' or 't', depending on whether we're checking for
            a model, handle or transformer.

        :type infix: str
        :param infix: The infix to check.

        :rtype: Boolean
        :returns: True or False.
        """
        filename = ''
        if obj_type == 'm':
            filename = self.layout.get_model_file(infix)
        elif obj_type == 'h':
            filename = self.layout.get_handle_file(infix)
        elif obj_type == 't':
            filename = self.layout.get_transformer_file(infix)

        full_path = os.path.join(self.root, filename)

        return os.path.isfile(full_path)

    def __default_infix(self, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix

    def load_mlp_hparams(self, n_layers, infixes):
        """Assembles the ``hparams`` init arg for a MLP setup by loading
        the specified models' init args when requested/available.

        Note that if you want to load the default empty-infix model,
        you have to supply an empty infix explicitly (``''``), otherwise
        no init args will be loaded.
        """
        assert len(n_layers) == len(infixes)
        hparams = []
        for infix in infixes:
            if infix is None:
                hparams.append({})
            else:
                model = self.load_model(infix)
                init_args = model._init_args_snapshot()
                hparams.append(init_args)
        return hparams


class IndexLoader(object):
    """This class is responsible for loading indexes from a data directory.
    It mirrors the function of :class:`MultimodalDatasetLoader`, but deals with
    index structures saved/loaded w.r.t. the data dir.

    >>> iLoader = IndexLoader('path/to/dataset/root', 'name')
    >>> index = iLoader.load_index('.i4096')

    This will load an index built from a 4096-dimensional image dataset.

    Also enables getting the 'right' string for the ``output_prefix`` string
    of the gensim :class:`gensim.similarities.Similarity` class.

    >>> iLoader.output_prefix('.i4096')
    'indexes/name.i4096.iidx'
    """
    def __init__(self, root, name):
        """Initializes the index loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are described in the :class:`DataDirLayout` class.

            By default, the output indexes will be serialized/saved into an
            ``indexes/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' . root)

        self.root = root
        self.layout = DataDirLayout(name)

        self._verify_layout()

    def _verify_layout(self):
        """Tests whether the data dir has the required layout for model/model
        handle loading."""

        files = os.listdir(self.root)

        if self.layout.index_dir not in files:
            raise ValueError('Missing index directory %s in dataset root (%s)' %
                             (self.layout.index_dir, self.root))

    def has_index(self, infix=None):
        """Checks whether a saved index object with the given infix exists in
        the dataset.

        :type infix: str
        :param infix: The infix (index label) of the desired index.
            (Infixes serve to differentiate saved indexes
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        """
        infix = self.__default_infix(infix)

        index_file = self.layout.get_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        return os.path.isfile(index_full_path)

    def has_text_index(self, infix=None):
        """Checks whether a saved text index object with the given infix exists
        in the dataset.

        :type infix: str
        :param infix: The infix (index label) of the desired index.
            (Infixes serve to differentiate saved indexes
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        """
        infix = self.__default_infix(infix)

        index_file = self.layout.get_text_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        return os.path.isfile(index_full_path)

    def save_index(self, index, infix=''):
        """Saves a model instance to file."

        :type index: safire.learning.interfaces.SafireTransformer
        :param index: A transformer to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the transformer to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_index(infix):
            logging.warn('Overwriting existing index: %s' %
                         self.layout.get_index_name(infix))

        index_file = self.layout.get_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        index.save(index_full_path)

    def load_index(self, infix=None):
        """Loads a similarity index object for the dataset with the specified
        infix.

        :type infix: str
        :param infix: The infix (index label) of the desired index.
            (Infixes serve to differentiate saved indexes
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_index(infix):
            raise ValueError('Cannot find index %s with infix %s in data dir %s.' % (
                self.layout.get_index_name(infix), infix, self.root))

        index_file = self.layout.get_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        index = Similarity.load(index_full_path)

        return index

    def save_text_index(self, index, infix=''):
        """Saves a text index instance to the appropriate file."

        :type index: safire.learning.interfaces.SafireTransformer
        :param index: A transformer to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the transformer to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_text_index(infix):
            logging.warn('Overwriting existing index: %s' %
                         self.layout.get_text_index_name(infix))

        index_file = self.layout.get_text_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        index.save(index_full_path)

    def load_text_index(self, infix=None):
        """Loads a similarity index object for the dataset with the specified
        infix.

        :type infix: str
        :param infix: The infix (index label) of the desired index.
            (Infixes serve to differentiate saved indexes
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_text_index(infix):
            raise ValueError('Cannot find index %s with infix %s in data dir %s.' % (
                self.layout.get_text_index_name(infix), infix, self.root))

        index_file = self.layout.get_text_index_file(infix)
        index_full_path = os.path.join(self.root, index_file)

        index = Similarity.load(index_full_path)

        return index

    def output_prefix(self, infix=None):
        """Builds the output_prefix parameter for a Similarity object
        so that the index can then be correctly saved."""
        infix = self.__default_infix(infix)

        index_file = self.layout.get_index_file(infix)
        return os.path.join(self.root, index_file)

    def text_output_prefix(self, infix=None):
        """Builds the output_prefix parameter for a Similarity object
        for the text modality so that the index can then be correctly saved."""
        infix = self.__default_infix(infix)

        index_file = self.layout.get_text_index_file(infix)
        return os.path.join(self.root, index_file)

    def __default_infix(self, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix


class ShardedDatasetLoader(object):
    """This class is responsible for loading sharded datasets from a data directory.
    It mirrors the function of :class:`MultimodalDatasetLoader`, but deals with
    sharded dataset structures saved/loaded w.r.t. the data dir.

    >>> sdLoader = ShardedDatasetLoader('path/to/dataset/root', 'name')
    >>> index = sdLoader.load_dataset('.SDA-1000.img')

    This will load an image dataset corresponding to the original data
    transformed by a 1000-neuron denoising autoencoder.

    Also enables getting the 'right' string for the ``output_prefix`` string
    of the gensim :class:`gensim.similarities.Similarity` class.

    >>> sdLoader.output_prefix('.i4096')
    'path/to/dataset/root/datasets/name.SDA-1000.img.shdat'


    """
    def __init__(self, root, name):
        """Initializes the index loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are described in the :class:`DataDirLayout` class.

            By default, the output indexes will be serialized/saved into an
            ``indexes/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' . root)

        self.root = root
        self.layout = DataDirLayout(name)

        self._verify_layout()

    def _verify_layout(self):
        """Tests whether the data dir has the required layout for model/model
        handle loading."""

        files = os.listdir(self.root)

        if self.layout.dataset_dir not in files:
            raise ValueError('Missing dataset directory %s in dataset root (%s)' %
                             (self.layout.dataset_dir, self.root))

    def has_dataset(self, infix=None):
        """Checks whether a saved index object with the given infix exists in
        the dataset.

        :type infix: str
        :param infix: The infix (dataset label) of the desired dataset.
            (Infixes serve to differentiate saved datasets
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        """
        infix = self.__default_infix(infix)

        dataset_file = self.layout.get_dataset_file(infix)
        dataset_full_path = os.path.join(self.root, dataset_file)

        return os.path.isfile(dataset_full_path)

    def save_dataset(self, dataset, infix=''):
        """Saves a model instance to file."

        :type dataset: safire.data.sharded_dataset.ShardedDataset
        :param dataset: A sharded dataset to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the dataset to save. Pick
            a sensible one. (Infixes serve to differentiate saved model handles
            obtained with different methods, such as Logistic Regression or
            Denoising Autoencoders.)

        """
        infix = self.__default_infix(infix)

        if self.has_dataset(infix):
            logging.warn('Overwriting existing index: %s' %
                         self.layout.get_dataset_name(infix))

        dataset_file = self.layout.get_dataset_file(infix)
        dataset_full_path = os.path.join(self.root, dataset_file)

        dataset.save(dataset_full_path)

    def load_dataset(self, infix=None):
        """Loads a similarity index object for the dataset with the specified
        infix.

        :type infix: str
        :param infix: The infix (index label) of the desired index.
            (Infixes serve to differentiate saved indexes
            obtained from different image corpora, like a full 4096-dimensional
            Imagenet feature corpus or a lower-dimensional encoding.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_dataset(infix):
            raise ValueError('Cannot find index %s with infix %s in data dir %s.' % (
                self.layout.get_dataset_name(infix), infix, self.root))

        dataset_file = self.layout.get_dataset_file(infix)
        dataset_full_path = os.path.join(self.root, dataset_file)

        index = ShardedDataset.load(dataset_full_path)

        return index

    def output_prefix(self, infix=None):
        """Builds the output_prefix parameter for a ShardedDataset object
        so that the dataset can then be correctly saved/loaded."""
        infix = self.__default_infix(infix)

        dataset_file = self.layout.get_dataset_file(infix)
        return os.path.join(self.root, dataset_file)

    def __default_infix(self, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix


class MultimodalShardedDatasetLoader(MultimodalDatasetLoader):
    """Implements all dataset loader functionality, but returns the sharded
    dataset version instead of indexed corpus-based datasets.

    Note the ``output_prefix`` methods, which are used to initialize dataset
    filenames AND the directory to which the shards will be written: these give
    the correct string to pass to the ShardedDataset constructor as the
     ``output_prefix`` parameter. The idiom for creating a ShardedDatasets is:

    >>> loader = MultimodalShardedDatasetLoader(root, name)
    >>> corpus = loader.load_text_corpus(label)
    >>> output_prefix = loader.text_output_prefix(label)
    >>> dataset = ShardedDataset(output_prefix, corpus, ...)
    >>> dataset.save()
    """

    def __init__(self, root, name, sentences=False, text_loader=VTextCorpus,
                 img_loader=ImagenetCorpus, text_serializer=MmCorpus,
                 img_serializer=MmCorpus):
        """Initializes the sharded mulitmodal dataset loader.

        :type root: str
        :param root: The root directory where the safire data lies. Expected
            contents are the text/ and img/ directories, a ``*.vtlist`` file,
            an Imagenet output file in the format described in
            :class:`ImagenetCorpus` and a text-image mapping as described in
            :class:`MultimodalDataset`.

            By default, the output corpora will be serialized/saved into a
            ``corpora/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        :type sentences: bool
        :param sentences: Defines whether sentences should be used as documents
            in corpusifying the text data

        :type text_loader: gensim.interfaces.CorpusABC
        :param text_loader: Corpus class used to load the text data.

        :type img_loader: gensim.interfaces.CorpusABC
        :param img_loader: Corpus class used to load the image data.
        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' % root)

        self.root = root

        self.sentences = sentences
        self.layout = DataDirLayout(name)

        # Default serializers.
        self.text_serializer = text_serializer
        self.img_serializer = img_serializer

        self._verify_layout()

        if not text_loader:
            text_loader = VTextCorpus

        if not img_loader:
            img_loader = ImagenetCorpus

        self.text_loader = text_loader
        self.img_loader = img_loader

    def _verify_layout(self):

        super(MultimodalShardedDatasetLoader, self)._verify_layout()

        files = frozenset(os.listdir(self.root))

        if self.layout.dataset_dir not in files:
            raise ValueError('Dataset dir %s missing in dataset root (%s), available:\n%s' % (
                                                 self.layout.text_dir,
                                                 self.root,
                                                 '\n'.join(files)))

    def text_output_prefix(self, infix=None):
        """Builds the output_prefix parameter for a text ShardedDataset object
        so that the index can then be correctly saved/loaded"""
        infix = self.__default_infix(infix)

        dataset_file = self.layout.get_dataset_file(infix + self.layout.text_corpname)
        return os.path.join(self.root, dataset_file)

    def img_output_prefix(self, infix=None):
        """Builds the output_prefix parameter for an image ShardedDataset object
        so that the index can then be correctly saved/loaded"""
        infix = self.__default_infix(infix)

        dataset_file = self.layout.get_dataset_file(infix + self.layout.img_corpname)
        return os.path.join(self.root, dataset_file)

    def output_prefix(self, infix=None):
        """Builds the output_prefix parameter for a ShardedDataset object
        so that the index can then be correctly saved/loaded"""
        infix = self.__default_infix(infix)

        dataset_file = self.layout.get_dataset_file(infix)
        return os.path.join(self.root, dataset_file)

    def load(self, infix=None, text_infix=None, img_infix=None,
             text_serializer=MmCorpus, img_serializer=MmCorpus):
        """Creates a ShardedMultimodalDataset. If corpora are not generated,
        raises a ValueError.

        :type infix: str
        :param infix: The infix (corpus label) of the desired text
            and image corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags or scaling down image
            dimensionality.)

            If ``infix`` is supplied, ``text_infix`` and ``img_infix`` are
            overriden by ``infix``.

        :type text_infix: str
        :param text_infix: The infix (corpus label) of the desired text
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type img_infix: str
        :param img_infix: The infix (corpus label) of the desired image
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type text_serializer: class froom gensim.interfaces.CorpusABC
        :param text_serializer: The corpus class that should be used to
            de-serialize the given text corpus data. Defaults to
            ``text_serializer`` passed in ``__init__``.

        :type img_serializer: class froom gensim.interfaces.CorpusABC
        :param img_serializer: The corpus class that should be used to
            de-serialize the given image corpus data. Defaults to
            ``img_serializer`` passed in ``__init__``.

        :rtype: safire.data.multimodal_dataset.ShardedMultimodalDataset
        :returns: A :class:`ShardedMultimodalDataset` representation of the
            dataset.

        :raises: ValueError
        """
        if not text_serializer:
            text_serializer = self.text_serializer
        if not img_serializer:
            img_serializer = self.img_serializer

        if infix:
            text_infix = infix
            img_infix = infix

        if not self.has_text_corpora(text_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))

        if not self.has_image_corpora(img_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))


        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        text_corpora_fnames = self.layout.required_text_corpus_names(text_infix)
        text_data = os.path.join(corpus_dir, text_corpora_fnames[0])
        text_obj = os.path.join(corpus_dir, text_corpora_fnames[2])
            # Skipping the '.mmcorp.index' file

        img_corpora_fnames = self.layout.required_img_corpus_names(img_infix)
        img_data = os.path.join(corpus_dir, img_corpora_fnames[0])
        img_obj = os.path.join(corpus_dir, img_corpora_fnames[2])

        textdoc2imdoc = os.path.join(self.root, self.layout.textdoc2imdoc)

        text_output_prefix = self.text_output_prefix(text_infix)
        img_output_prefix = self.img_output_prefix(img_infix)

        dataset = ShardedMultimodalDataset(text_output_prefix, text_obj,
                                           img_output_prefix, img_obj,
                                           textdoc2imdoc=textdoc2imdoc,
                                           text_serializer=text_serializer,
                                           img_serializer=img_serializer,
                                           text_mm_filename=text_data,
                                           img_mm_filename=img_data)

        return dataset

    def build_text(self, corpus, text_infix=None, dataset_init_args={}):
        """From the given corpus, builds a text ShardedDataset with the given
        infix."""
        output_prefix = self.text_output_prefix(text_infix)
        dataset = ShardedDataset(output_prefix, corpus, **dataset_init_args)
        dataset.save()

    def load_text(self, text_infix=None, dataset_init_args={}):
        """Loads a text-only dataset using the class
        :class:`UnsupervisedVTextCorpusDataset`.

        :type text_infix: str
        :param text_infix: The infix (corpus label) of the desired text
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type dataset_init_args: dict
        :param dataset_init_args: Further init args that will be passed to the
            dataset constructor.

        :rtype: safire.data.corpus_dataset.UnsueprvisedVTextCorpusDataset
        :returns: The :class:`UnsupervisedVTextCorpusDataset` built from the
            given text corpora.

        """
        if not self.has_text_corpora(text_infix):
            raise ValueError('Text corpora unavailable in SAFIRE dataset '+
                             'at %s with infix %s' % (os.path.join(self.root,
                             self.layout.corpus_dir), text_infix))


        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        text_corpora_fnames = self.layout.required_text_corpus_names(text_infix)
        text_mm_fname = os.path.join(corpus_dir, text_corpora_fnames[0])
        text_ic_fname = os.path.join(corpus_dir, text_corpora_fnames[2])

        output_prefix = self.text_output_prefix(text_infix)
        if not os.path.isfile(output_prefix):
            output_prefix = self.output_prefix(text_infix)

        logging.debug('Loading UnsupervisedShardedVTextCorpusDataset with'
                      ' data %s, obj %s' % (output_prefix, text_ic_fname))

        dataset_init_args['mm_corpus_filename'] = text_mm_fname

        dataset = UnsupervisedShardedVTextCorpusDataset(output_prefix,
                                                        text_ic_fname,
                                                        **dataset_init_args)

        return dataset

    def build_img(self, corpus, img_infix=None, dataset_init_args={}):
        """From the given corpus, builds a text ShardedDataset with the given
        infix."""
        output_prefix = self.img_output_prefix(img_infix)
        dataset = ShardedDataset(output_prefix, corpus, **dataset_init_args)
        dataset.save()

    def load_img(self, img_infix=None, dataset_init_args={}):
        """Loads an image-only dataset using the class
        :class:`UnsupervisedShardedImagenetCorpusDataset`.

        :type img_infix: str
        :param img_infix: The infix (corpus label) of the desired image
            corpora. (Infixes serve to differentiate corpora
            obtained with different methods, such as filtering tokens
            according to Part of Speech tags.)

        :type dataset_init_args: dict
        :param dataset_init_args: Further init args that will be passed to the
            dataset constructor.

        :rtype: safire.data.corpus_dataset.UnsueprvisedShardedImagenetCorpusDataset
        :returns: The :class:`UnsupervisedImagenetCorpusDataset` built from the
            given text corpora.

        """
        if not self.has_image_corpora(img_infix):
            raise ValueError('Image corpora unavailable in dataset '+
                             'at %s with infix %s (available: %s)' % (
                             os.path.join(self.root, self.layout.corpus_dir),
                             img_infix, '\n'.join(
                                 map(str, os.listdir(os.path.join(self.root,
                                                     self.layout.corpus_dir)))
                             )))


        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        img_corpora_fnames = self.layout.required_img_corpus_names(img_infix)
        img_mm_fname = os.path.join(corpus_dir, img_corpora_fnames[0])
        img_ic_fname = os.path.join(corpus_dir, img_corpora_fnames[2])

        output_prefix = self.img_output_prefix(img_infix)
        if not os.path.isfile(output_prefix):
            output_prefix = self.output_prefix(img_infix)

        logging.debug('Loading UnsupervisedShardedImagenetCorpusDataset with'
                      ' data %s, obj %s' % (output_prefix, img_ic_fname))

        dataset_init_args['mm_corpus_filename'] = img_mm_fname

        dataset = UnsupervisedShardedImagenetCorpusDataset(output_prefix,
                                                           img_ic_fname,
                                                           **dataset_init_args)

        return dataset

    def __default_infix(self, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix


class LearnerLoader(object):
    """This class is responsible for loading learners from a data directory.
    It mirrors the function of :class:`IndexLoader`, but deals with
    learner structures saved/loaded w.r.t. the data dir.

    >>> lLoader = LearnerLoader('path/to/dataset/root', 'name')
    >>> learner = lLoader.load_learner('.SDA-1000.img')

    This will load a learner used for a Denoising Autoencoder with 1000
    output neurons experiment. The learner also includes some experimental
    environment, such as intermediate models saved at various stages of
    learner operation.
    """

    def __init__(self, root, name):
        """Initializes the learner loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are described in the :class:`DataDirLayout` class.

            By default, the output learners will be serialized/saved into an
            ``learners/`` subdirectory of ``root``.

        :type name: str
        :param name: The dataset name. This will usually be the same as the
            name of the root directory. The name defines what filenames the
            loader will expect and use in files governing which text and image
            data will be loaded.

        """
        if not os.path.isdir(root):
            raise ValueError('Could not find dataset root: %s' . root)

        self.root = root
        self.layout = DataDirLayout(name)

        self._verify_layout()

    def _verify_layout(self):
        """Tests whether the data dir has the required layout for model/model
        handle loading."""

        files = os.listdir(self.root)

        if self.layout.learner_dir not in files:
            raise ValueError('Missing learner directory %s in dataset root (%s)' %
                             (self.layout.learner_dir, self.root))

    def has_learner(self, infix=None):
        """Checks whether a saved learner object with the given infix exists in
        the dataset.

        :type infix: str
        :param infix: The infix (learner label) of the desired learner.
        """
        infix = self.__default_infix(infix)

        learner_full_path = self.learner_name(infix)

        return os.path.isfile(learner_full_path)

    def save_learner(self, learner, infix=''):
        """Saves a model instance to file."

        :type learner: safire.learning.learners.BaseSGDLearner
        :param learner: A transformer to save in the data dir.

        :type infix: str
        :param infix: The infix (handle label) of the learner to save. Pick
            a sensible one. (Learners serve to differentiate saved learners
            with different settings.)

        """
        infix = self.__default_infix(infix)

        if self.has_learner(infix):
            logging.warn('Overwriting existing learner: %s' %
                         self.layout.get_learner_name(infix))

        learner_full_path = self.learner_name(infix)

        learner.save(learner_full_path)

    def load_learner(self, infix=None):
        """Loads a Learner object for the dataset with the specified infix.

        :type infix: str
        :param infix: The infix (learner label) of the desired learner.
            (Infixes serve to differentiate saved learners.)

        :raises: ValueError
        """
        infix = self.__default_infix(infix)

        if not self.has_learner(infix):
            raise ValueError('Cannot find learner %s with infix %s in data dir %s.' % (
                self.layout.get_learner_name(infix), infix, self.root))

        learner_full_path = self.learner_name(infix)

        learner = BaseSGDLearner.load(learner_full_path)

        return learner

    def learner_name(self, infix=None):
        """Builds the output_prefix parameter for a BaseSGDLearner object
        so that the learner can then be correctly saved."""
        infix = self.__default_infix(infix)

        learner_file = self.layout.get_learner_file(infix)
        return os.path.join(self.root, learner_file)

    def __default_infix(self, infix):
        """Handles converting an infix value of None to an empty string,
        because the Layout object does not accept None when looking for
        files/generating file names."""
        if infix is None:
            return ''
        return infix


