#!/usr/bin/env python
"""Layouts are used by Loaders to discover data files.

The layout class describes where each file is in some dataset directory. **The
items that have to be in the Layout are given inside the Loader,** Layouts
cannot be swapped for each other without breaking the Loader. The Layout is
essentially just a description of the data directory that a particular Loader
expects.

"""

import logging
import os
import shutil
import sys


logger = logging.getLogger(__name__)


class DataDirLayout(object):
    """A static class that holds constants that define the layout of a
    multimodal dataset root dir. (Layouts are used by Loaders to discover data
    files.)

    The expected layout is::

        root/
        +----- text/
        |    +------ text1.vt.txt.gz
        |    +------ text2.vt.txt.gz
        |    +------ text3.vt.txt.gz
        |    +------ text4.vt.txt.gz
        |            etc.
        +----- img/
        |    +------ image1.jpg
        |    +------ image2.jpg
        |    +------ image3.jpg
        |    +------ image4.jpg
        +----- corpora/
        +----- datasets/
        +----- models/
        +----- indexes/
        +----- name.vtlist
        +----- im.ftrs.csv
        +----- vtext-image-map.csv

    The expected ``name.vtlist`` entries are a list of vtext filenames relative
    to the root::

        text/text1.vt.txt.gz
        text/text2.vt.txt.gz
        text/text3.vt.txt.gz
        text/text4.vt.txt.gz

    This file is expected to be given to :class:`VTextCorpus` as an input file.

    The expected ``im.ftrs.csv`` file format is::

        image1.jpg  0;0;0.2232;0.345;0;0;1.93833;0.84;0 .... ;0.012
        image2.jpg  0;1.607;0.637;0;0;1.2;0.2001;0.14;0 .... ;0.214
        image3.jpg  0.018;0;0;0.1345;0;0;0.39;1.084;0.2 .... ;1.09
        image4.jpg  0;1.607;0.637;0;0;1.2;0.2001;0.14;0 .... ;0

    with tab-separated image ID and feature value columns and 4096
    semicolon-separated entries in the second column. This file is expected to
    be given to :class:`ImagenetCorpus` as an input file.

    And the expected format of ``vtext-image-map.csv`` is::

        text/text1.vt.txt.gz    image1
        text/text1.vt.txt.gz    image3
        text/text2.vt.txt.gz    image2
        text/text3.vt.txt.gz    image3
        text/text3.vt.txt.gz    image4
        text/text3.vt.txt.gz    image1

    This file is given to the output :class:`MultimodalDataset` as the
    text-image mapping (parameter ``textdoc2imdoc2``. Note that the relationship
    between texts and images is M:N - a text can have any number of associated
    images, and an image can be used by any number of texts.

    The ``corpora/`` directory contains the saved corpora of various shapes and
    sizes. The corpus name format is::

        name.infix.type.suffix

    where:

    * ``name`` is the same as ``name.vtlist``,

    * ``infix`` is some string that describes the corpus,

    * ``type`` reflects whether the corpus has text or image data,

    * ``suffix`` is a string describing the corpus file type. The current
      suffixes are ``mmcorp`` for MmCorpus-serialized data, ``mmcorp.index`` for
      the MmCorpus index that went along with the ``name.infix.type``
      combination, ``vtcorp`` for the :class:`VTextCorpus` object used to read
      the given text corpus and ``icorp`` for the :class:`ImagenetCorpus` used
      to read the given image corpus.

    The ``models/`` directory contains saved models, with the suffix ``.sfm``,
    and saved model handles, with the suffix ``.mhandle``.

    The ``indexes/`` directory contains saved similarity indexes, built from
    various image corpora.


    """

    def __init__(self, name):
        """Initializes a layout object.

        :type name: str
        :param name: The dataset name. Currently expected to appear in two
            roles in the data directory:


            * The ``vtlist`` file passed to VTextCorpus to operate on has the name
              ``name.vtlist``

            * The corpus file names start with ``name``


        """

        self.name = name

        self.img_dir = 'img'
        self.text_dir = 'text'

        self.corpus_dir = 'corpora'
        self.dataset_dir = 'datasets'
        self.model_dir = 'models'
        self.index_dir = 'indexes'
        self.learner_dir = 'learners'
        self.temp_dir = 'tmp'
        self.introspection_dir = 'introspection'

        # Master files.
        self.vtlist = self.name + '.vtlist'
        self.image_vectors = self.name + '.' 'im.ftrs.csv'
        # Alternate way of reading image vectors: instead of keeping different
        # subsets of the vectors themselves, only keep one file with *all* the
        # image vectors for various datasets over the same source data and
        # use lists of image docnames to generate individual datasets.
        # This is enabled by the include/exclude_docnames feature of
        # ImagenetCorpus.
        self.source_image_vectors = 'im.ftrs.csv'
        self.image_docnames = self.name + '.img-docnames'

        self.textdoc2imdoc = self.name + '.' + 'vtext-image-map.csv'
        self.imgid2imgfile = self.name + '.' + 'img.ids2files.csv'

        self.annotation_inputs = 'annotation_inputs.csv'
        self.annotation_outputs = 'annotation_outputs.csv'

        self.img_corpname = '.img'
        self.text_corpname = '.vt'

         # Pipeline names should be agnostic to text vs. images (user handles
         # the naming).
        self.pipeline_suffix = '.pln'
        self.pipeline_serialization_suffix = '.pls'

        self.mm_corp_suffix = '.mmcorp'
        self.mm_index_suffix = '.mmcorp.index'
        self.text_corp_suffix = '.vtcorp'
        self.img_corp_suffix = '.icorp'

        self.dataset_suffix = '.shdat'

        self.model_suffix = '.sfm'
        self.handle_suffix = '.mhandle'
        self.transformer_suffix = '.sftrans'
        self.index_suffix = '.iidx'
        self.learner_suffix = '.lrn'

        self.tcorp_data = self.name + self.text_corpname + self.mm_corp_suffix
        self.tcorp_obj = self.name + self.text_corpname + self.text_corp_suffix

        self.icorp_data = self.name + self.img_corpname + self.mm_corp_suffix
        self.icorp_obj = self.name + self.img_corpname + self.img_corp_suffix

        self.required_suffixes = [
            self.text_corpname + self.mm_corp_suffix,
            self.text_corpname + self.mm_index_suffix,
            self.text_corpname + self.text_corp_suffix,
            self.img_corpname + self.mm_corp_suffix,
            self.img_corpname + self.mm_index_suffix,
            self.img_corpname + self.img_corp_suffix
        ]

        self.required_text_suffixes = [
            self.text_corpname + self.mm_corp_suffix,
            self.text_corpname + self.mm_index_suffix,
            self.text_corpname + self.text_corp_suffix,
        ]

        self.required_image_suffixes = [
            self.img_corpname + self.mm_corp_suffix,
            self.img_corpname + self.mm_index_suffix,
            self.img_corpname + self.img_corp_suffix
        ]

    def required_corpus_names(self, infix=None, update_separator=False):
        """Generates the required corpora for a given corpus infix. These are::

            [ '.'.join(self.name, infix, suffix)
               for suffix in self.required_suffixes ]

        Coming out as ``safire.infix.mmcorp``, ``safire.infix.mmcorp.index``,
        etc.

        :param infix: The corpus infix. This is either given manually, or
            generated from corpus/transformation parameters automatically in
            the corresponding class.

        :rtype: list(str)
        :returns: A list of the required corpus names for the given infix.
        """

        # If infix is given without separator
        infix_separator = ''
        if infix and update_separator and infix[0] != '.':
            infix_separator = '.'

        elif infix is None:
            infix = ''

        updated_infix = infix_separator + infix

        return [ self.name + updated_infix + suffix
                 for suffix in self.required_suffixes ]

    def required_text_corpus_names(self, infix=None, update_separator=False):
        """Generates the required text corpus names for a given corpus infix.
        These are::

            [ '.'.join(self.name, infix, suffix)
               for suffix in self.required_text_suffixes ]

        Coming out as ``safire.infix.vt.mmcorp``,
        ``safire.infix.vt.mmcorp.index``, etc.

        :param infix: The corpus infix. This is either given manually, or
            generated from corpus/transformation parameters automatically in
            the corresponding class.

        :rtype: list(str)
        :returns: A list of the required corpus names for the given infix.
        """

        # If infix is given without separator
        infix_separator = ''
        if infix and update_separator and infix[0] != '.':
            infix_separator = '.'

        elif infix is None:
            infix = ''

        updated_infix = infix_separator + infix

        return [ self.name + updated_infix + suffix
                 for suffix in self.required_text_suffixes ]

    def required_img_corpus_names(self, infix=None, update_separator=False):
        """Generates the required image corpora for a given corpus infix.
        These are::

            [ '.'.join(self.name, infix, suffix)
               for suffix in self.required_img_suffixes ]

        Coming out as ``safire.infix.img.mmcorp``,
        ``safire.infix.img.mmcorp.index``, etc.

        :param infix: The corpus infix. This is either given manually, or
            generated from corpus/transformation parameters automatically in
            the corresponding class.

        :rtype: list(str)
        :returns: A list of the required corpus names for the given infix.
        """

        # If infix is given without separator
        infix_separator = ''
        if infix and update_separator and infix[0] != '.':
            infix_separator = '.'

        elif infix is None:
            infix = ''

        updated_infix = infix_separator + infix

        return [ self.name + updated_infix + suffix
                 for suffix in self.required_image_suffixes ]

    def get_text_corpus_name(self, infix):
        """Returns the text corpus name for the given infix."""
        return self.name + infix + self.text_corpname + self.text_corp_suffix

    def get_text_corpus_file(self, infix):
        """Returns the text corpus file relative to data dir root."""
        corpname = self.get_text_corpus_name(infix)
        return os.path.join(self.corpus_dir, corpname)

    def get_image_corpus_name(self, infix):
        """Returns the text corpus name for the given infix."""
        return self.name + infix + self.img_corpname + self.img_corp_suffix

    def get_image_corpus_file(self, infix):
        """Returns the text corpus file relative to data dir root."""
        corpname = self.get_image_corpus_name(infix)
        return os.path.join(self.corpus_dir, corpname)

    def get_model_name(self, infix):
        """Returns the model name for the given infix."""
        return self.name + infix + self.model_suffix

    def get_model_file(self, infix):
        """Returns the path to model file with given infix from layout
         root."""
        return os.path.join(self.model_dir, self.get_model_name(infix))

    def get_handle_name(self, infix):
        """Returns the model handle name for the given infix."""
        return self.name + infix + self.handle_suffix

    def get_handle_file(self, infix):
        """Returns the path to model handle file with given infix from layout
         root."""
        return os.path.join(self.model_dir, self.get_handle_name(infix))

    def get_transformer_name(self, infix):
        """Returns the transformer name for the given infix."""
        return self.name + infix + self.transformer_suffix

    def get_transformer_file(self, infix):
        """Returns the path to transformer file with given infix from layout
         root."""
        return os.path.join(self.model_dir, self.get_transformer_name(infix))

    def get_index_name(self, infix):
        """Returns the index name for the given infix."""
        return self.name + infix + self.index_suffix

    def get_index_file(self, infix):
        """Returns the path to index file with given infix from layout root."""
        return os.path.join(self.index_dir, self.get_index_name(infix))

    def get_text_index_name(self, infix):
        """Returns the index name for the given infix."""
        return self.name + infix + self.text_corpname + self.index_suffix

    def get_text_index_file(self, infix):
        """Returns the path to index file with given infix from layout root."""
        return os.path.join(self.index_dir, self.get_text_index_name(infix))

    def get_dataset_name(self, infix):
        """Returns the index name for the given infix."""
        return self.name + infix + self.dataset_suffix

    def get_dataset_file(self, infix):
        """Returns the path to index file with given infix from layout root."""
        return os.path.join(self.dataset_dir, self.get_dataset_name(infix))

    def get_learner_name(self, infix):
        """Returns the index name for the given infix."""
        return self.name + infix + self.learner_suffix

    def get_learner_file(self, infix):
        """Returns the path to index file with given infix from layout root."""
        return os.path.join(self.learner_dir, self.get_learner_name(infix))


def clean_dir(path, force=False):
    """Removes everything from the directory given by path."""
    if not force:
        print 'Are you sure you want to delete everything in %s? [y/n]' % path
        confirmation = sys.stdin.readline().strip()
        if confirmation not in ['y', 'Y', 'yes', 'Yes', 'YES']:
            print 'Aborting...'
            return
        else:
            print 'Proceeding...'

    shutil.rmtree(path)
    os.makedirs(path)


def clean_data_root(root, force=True):
    """Cleans the given data root - removes corpora, datasets, models, etc."""
    layout = DataDirLayout(root)

    corpus_dir = os.path.join(layout.name, layout.corpus_dir)
    dataset_dir = os.path.join(layout.name, layout.dataset_dir)
    model_dir = os.path.join(layout.name, layout.model_dir)
    learner_dir = os.path.join(layout.name, layout.learner_dir)
    index_dir = os.path.join(layout.name, layout.index_dir)
    temp_dir = os.path.join(layout.name, layout.temp_dir)
    introspection_dir = os.path.join(layout.name, layout.introspection_dir)

    clean_dir(corpus_dir, force=force)
    clean_dir(dataset_dir, force=force)
    clean_dir(model_dir, force=force)
    clean_dir(learner_dir, force=force)
    clean_dir(index_dir, force=force)
    clean_dir(temp_dir, force=force)
    #clean_dir(introspection_dir, force=force)


def init_data_root(path, overwrite=False):
    """Initializes the given directory as a data root: creates all necessary
    subdirectories.

    If ``overwrite`` is set, will clean out all existing corpora/datasets/etc.
    in the given directory if it already is a data root (or some partial
    remainder - leftover ``corpora`` directory, etc.)."""
    layout = DataDirLayout(path)

    is_already_root = True

    corpus_dir = os.path.join(layout.name, layout.corpus_dir)
    if not os.path.exists(corpus_dir):
        is_already_root = False
        os.makedirs(corpus_dir)

    dataset_dir = os.path.join(layout.name, layout.dataset_dir)
    if not os.path.exists(dataset_dir):
        is_already_root = False
        os.makedirs(dataset_dir)

    model_dir = os.path.join(layout.name, layout.model_dir)
    if not os.path.exists(model_dir):
        is_already_root = False
        os.makedirs(model_dir)

    learner_dir = os.path.join(layout.name, layout.learner_dir)
    if not os.path.exists(learner_dir):
        is_already_root = False
        os.makedirs(learner_dir)

    index_dir = os.path.join(layout.name, layout.index_dir)
    if not os.path.exists(index_dir):
        is_already_root = False
        os.makedirs(index_dir)

    temp_dir = os.path.join(layout.name, layout.temp_dir)
    if not os.path.exists(temp_dir):
        is_already_root = False
        os.makedirs(temp_dir)

    introspection_dir = os.path.join(layout.name, layout.introspection_dir)
    if not os.path.exists(introspection_dir):
        is_already_root = False
        os.makedirs(introspection_dir)

    if is_already_root:
        logging.warn('Directory already is a safire data root.')

    if overwrite:
        logging.warn('Overwriting to clean root...')
        clean_data_root(path, force=True)

