#!/usr/bin/env python

import logging
import os

from gensim.corpora.mmcorpus import MmCorpus

from .vtextcorpus import VTextCorpus
from .imagenetcorpus import ImagenetCorpus
from .safire.datasets.multimodal_dataset import MultimodalDataset
from .layouts import DataDirLayout


logger = logging.getLogger(__name__)

class DataDirLayout(object):
    """A static class that holds constants that define the layout of a
    dataset root dir. For specific layouts, override the ``name`` attribute."""

    def __init__(self, name):

        self.name = name

        self.img_dir = 'img'
        self.text_dir = 'text'
        self.corpus_dir = 'corpora'

        self.vtlist = self.name + '.vtlist'
        self.image_vectors = 'im.ftrs.csv'
        self.textdoc2imdoc = 'vtext-image-map.csv'

        self.img_corpname = '.img'
        self.text_corpname = '.vt'

        self.mm_corp_suffix = '.mmcorp'
        self.mm_index_suffix = '.mmcorp.index'
        self.text_corp_suffix = '.vtcorp'
        self.img_corp_suffix = '.icorp'

        self.tcorp_data = self.name + self.text_corpname + self.mm_corp_suffix
        self.tcorp_obj = self.name + self.text_corpname + self.text_corp_suffix

        self.icorp_data = self.name + self.img_corpname + self.mm_corp_suffix
        self.icorp_obj = self.name + self.img_corpname + self.img_corp_suffix

        self.required_corpus_names = [
                self.name + self.img_corpname + self.mm_corp_suffix,
                self.name + self.img_corpname + self.mm_index_suffix,
                self.name + self.img_corpname + self.img_corp_suffix,
                self.name + self.text_corpname + self.mm_corp_suffix,
                self.name + self.text_corpname + self.mm_index_suffix,
                self.name + self.text_corpname + self.text_corp_suffix ]


class MultimodalDatasetLoader(object):
    """A class that works as an interface for loading a SAFIRE dataset.

    The loading itself may take a shortcut through a ``load`` class method::

        >>> safire_multimodal_dataset = SAFIRE.load('/path/to/safire/root')
        >>> mlp = MultilayerPerceptron.setup(data=safire_multimodal_dataset, ...)

    Or the SAFIRE class may be instantiated:

        >>> safire = SAFIRE('/path/to/safire/root')
        >>> safire_multimodal_dataset = safire.load()

    Relies on gensim-style corpora already being generated. If they are not
    present, will first need to generate them. Because that is a time-consuming
    procedure, the SAFIRE class will not generate them automatically. Instead,
    the following workflow is available:

        >>> safire = SAFIRE('/path/to/safire/root')
        >>> safire.has_corpora()
        False
        >>> dataset = safire.load()
        ValueError: Corpora unavailable in SAFIRE dataset at /path/to/safire/root
        >>> safire.generate_corpora(safire_root)
        [...takes a long time..]
        >>> safire.has_corpora()
        True
        >>> dataset = safire.load()

    """

    ### TODO: Test, refactor to a more abstract level - constants given at
    #         __init__() time

    def __init__(self, root, name, sentences=False, text_loader=VTextCorpus,
                 img_loader=ImagenetCorpus):
        """Initializes the SAFIRE dataset loader.

        :type root: str
        :param root: The root directory where the safire data lies. The expected
            contents are the text/ and img/ directories, a ``*.vtlist`` file,
            an Imagenet output file in the format described in
            :class:`ImagenetCorpus` and a text-image mapping as described in
            :class:`MultimodalDataset`.

            By default, the output corpora will be serialized/saved into a
            corpora/ subdirectory of ``root``.

        :type sentences: bool
        :param sentences: Defines whether sentences should be used as documents
            in corpusifying the text data

        :type text_loader: gensim.interfaces.CorpusABC
        :param text_loader: Corpus class used to load the text data.

        :type img_loader: gensim.interfaces.CorpusABC
        :param img_loader: Corpus class used to load the image data.
        """
        if not os.path.isdir(root):
            raise ValueError('Could not find SAFIRE root: %s' . root)

        self.root = root

        self.sentences = sentences
        self.layout = DataDirLayout(name)

        self.root = root

        self._verify_layout()

        self.serializer = MmCorpus

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
        files = os.listdir(self.root)

        if self.layout.img_dir not in files:
            raise ValueError("Image dir %s missing in SAFIRE root (%s), available: %s" % (
                                                 self.layout.img_dir,
                                                 self.root,
                                                 str(files)))
        if self.layout.text_dir not in files:
            raise ValueError('Text dir %s missing in SAFIRE root (%s), available:\n%s' % (
                                                 self.layout.text_dir,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.vtlist not in files:
            raise ValueError('Vtlist %s missing in SAFIRE root (%s), available:\n%s' % (
                                                 self.layout.vtlist,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.image_vectors not in files:
            raise ValueError('Image vectors %s missing in SAFIRE root (%s), available:\n%s' % (
                                                 self.layout.image_vectors,
                                                 self.root,
                                                 '\n'.join(files)))

        if self.layout.textdoc2imdoc not in files:
            raise ValueError('Vtlist %s missing in SAFIRE root (%s), available:\n%s' % (
                                                 self.layout.textdoc2imdoc,
                                                 self.root,
                                                 '\n'.join(files)))


    def has_corpora(self):
        """Checks whether corpora for loading the SAFIRE multimodal dataset
        have been generated and are in the right place."""
        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)
        files = os.listdir(corpus_dir)
        for corpus in self.layout.required_corpus_names:
            if corpus not in files:
                logger.info('Corpus %s not found in corpus directory %s (safire root %s). Available:\n%s' % (
                                corpus,
                                corpus_dir,
                                self.root,
                                '\n'.join(files)))
                return False

        return True


    def build_corpora(self):
        """Creates all corpora necessary for the creation of the
        MultimodalDataset over the SAFIRE data."""

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        # Text corpus building/serialization
        with open(os.path.join(self.root, self.layout.vtlist)) as vtlist_handle:
            text_corpus = VTextCorpus(vtlist_handle, input_root=self.root)

            self.serializer.serialize(os.path.join(corpus_dir,
                                                   self.layout.tcorp_data),
                                      text_corpus)

            text_corpus.save(os.path.join(corpus_dir, self.layout.tcorp_obj))

        # Image corpus building/serialization
        with open(os.path.join(self.root, self.layout.image_vectors)) as img_handle:

            img_corpus = ImagenetCorpus(img_handle, delimiter=';')

            self.serializer.serialize(os.path.join(corpus_dir,
                                                   self.layout.icorp_data),
                                      img_corpus)

            img_corpus.save(os.path.join(corpus_dir, self.layout.icorp_obj))

    def load(self):
        """Creates the SAFIRE MultimodalDataset. If corpora are not generated,
        raises a ValueError.

        :raises: ValueError
        """
        if not self.has_corpora():
            raise ValueError('Corpora unavailable in SAFIRE dataset '+
                             'at %s' % os.path.join(self.root,
                                                    self.layout.corpus_dir))

        corpus_dir = os.path.join(self.root, self.layout.corpus_dir)

        text_data = os.path.join(corpus_dir, self.layout.tcorp_data)
        text_obj = os.path.join(corpus_dir, self.layout.tcorp_obj)

        img_data = os.path.join(corpus_dir, self.layout.icorp_data)
        img_obj = os.path.join(corpus_dir, self.layout.icorp_obj)

        textdoc2imdoc = os.path.join(self.root, self.layout.textdoc2imdoc)

        dataset = MultimodalDataset(text_data, text_obj, img_data, img_obj,
                                    aligned=False,
                                    textdoc2imdoc=textdoc2imdoc)

        return dataset
