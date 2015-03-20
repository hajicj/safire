"""
This module contains classes that ...
"""
import logging
import os
import codecs
from gensim.utils import is_corpus
import numpy
import operator
from safire.datasets.dataset import CompositeDataset
from safire.datasets.transformations import FlattenedDatasetCorpus
from safire.introspection import html_utils
from safire.utils.transcorp import get_id2doc_obj, get_id2word_obj, \
    find_type_in_pipeline

__author__ = "Jan Hajic jr."


class WriterABC(object):
    """Base class for introspection writers. The primary method is ``run()``,
    which writes the introspection output to a file and returns the full path to
    its filename. Other important methods include ``write()``, which writes to
    a given file, and ``generate_filename()``, which returns the filename for
    writing based on the parameters.

    Each writer has its own implementation of ``write()`` and can but doesn't
    have to override ``generate_filename()`` and ``run()`` implemented in this
    base class. However, to generate the filename, you will need to override
    the method ``iid_to_filename()``, which needs a file format (html, txt...)
    to turn an item ID into a filename.
    """
    def __init__(self, root):
        """Initialize the writer.

        :param root: The root directory to which the introspection files should
            be written.
        """
        self.root = root

    def run(self, iid, value, corpus):
        """Writes to a file and returns the filename.

        :param iid: The item ID, based on which the source document of the item
            can be retrieved.

        :param value: The item itself.

        :param corpus: The corpus on which the introspection transformer was
            applied. Used by the ``write()`` method to extract additional
            information such as document-id or vocabulary mappings (in various
            writers).
        """
        filename = self.generate_filename(iid)
        with codecs.open(filename, 'w', 'utf-8') as write_handle:
            self.write(write_handle, iid, value, corpus)
        return filename

    def write(self, output_handle, iid, value, corpus):
        """Generates the introspection string and writes it to the given file.
        """
        raise NotImplementedError()

    def generate_filename(self, iid):
        """Generates the filename to which to write the introspection output,
        based on the item ID. The generated name is a full path, depending on
        the init arg ``root``.

        :param iid: The item ID for which to write the output.

        :return: Filename string, path including root (should be absolute).
        """
        filename = self.iid_to_filename(iid)
        full_filename = os.path.join(self.root, filename)
        return full_filename

    @staticmethod
    def iid_to_filename(iid):
        """Converts an item ID into a filename, minus the absolute path (which
        is handled by the ``root`` init arg).

        :param iid: The item ID for which to generate the filename.

        :return: Filename string.
        """
        raise NotImplementedError
        # We cannot determine the suffix of the file at this point, need
        # implementation of write() (Is it html? Is txt? ..?)
        # filename = 'introspect.{0}.html'.format(iid)
        # return filename


class HtmlSimpleWriter(WriterABC):
    """This class writes a simple HTML output into two divs. It also creates
    Next and Prev links to (expected) introspection outputs for the next and
    previous item, to simplify browsing.

    A key method is ``get_html_content()``, which is responsible for generating
    the html code that corresponds to the content of the introspected item.
    """

    def __init__(self, root, **kwargs):
        super(HtmlSimpleWriter, self).__init__(root, **kwargs)

    @staticmethod
    def iid_to_filename(iid):
        """Implements the naming scheme."""
        return 'introspection.{0}.html'.format(iid)

    def generate_filename(self, iid):
        """Generates the absolute filename under which to save the introspection
        result for the given item ID.

        Note that this function generates a file name, NOT an URL."""
        abs_filename = os.path.abspath(os.path.join(self.root,
                                                    self.iid_to_filename(iid)))
        return abs_filename

    def write(self, output_handle, iid, value, corpus):
        """Generates the introspection result and writes it into the given
        file handle."""
        header = self.get_html_header()
        body = self.get_html_body(iid, value, corpus)
        footer = self.get_html_footer()

        print u'Document:\n'
        print header
        print body
        print footer

        output = header + u'\n\n' + body + u'\n\n' + footer
        output_handle.write(output + u'\n')

    @staticmethod
    def get_html_header():
        """Generates the constant part of the HTML output that comes *before*
        the content."""
        return unicode(html_utils.head)

    @staticmethod
    def get_html_footer():
        """Generates the constant part of the HTML output that comes *after*
        the content."""
        return u'\n'.join(
            [html_utils.as_comment(u'Generated using HtmlSimpleWriter.'),
             unicode(html_utils.foot)])

    def generate_next_and_prev_links(self, iid):
        # Add next/prev links.
        prev_file = html_utils.as_local_url(self.generate_filename(iid - 1))
        prev_link = html_utils.as_link(prev_file, 'Previous item')
        next_file = html_utils.as_local_url(self.generate_filename(iid + 1))
        next_link = html_utils.as_link(next_file, 'Next item')
        links_table = html_utils.as_table([[prev_link, next_link]])
        return links_table

    def generate_value(self, iid, value, corpus):
        value_content = html_utils.with_tag(value, 'div')
        return value_content

    def get_html_content(self, iid, value, corpus):
        """Generates the "interesting" part of the introspection file:
        formats the contents of the item that is being introspected."""
        elements = []

        # Create heading.
        docname = str(get_id2doc_obj(corpus)[iid])

        print 'Docname: {0}'.format(docname)
        print 'Value: {0}'.format(value)
        absolute_docname = os.path.join(self.root, docname)
        heading = u'Safire introspection: iid {0}, docname {1}' \
                  u''.format(iid, absolute_docname)
        elements.append(html_utils.with_tag(heading, 'h1'))

        # Paste content.
        value_content = self.generate_value(value)
        elements.append(value_content)

        links_table = self.generate_next_and_prev_links(iid)
        elements.append(html_utils.with_tag(links_table, 'div'))

        # Combine elements into a single string and wrap in a <body> tag.
        text = u'\n'.join(elements)
        return text

    def get_html_body(self, iid, value, corpus):
        """Formats the content as the ``<body>`` element of an html file."""
        content = self.get_html_content(iid, value, corpus)
        return html_utils.text_with_tag(content, 'body', newline=True)


class HtmlStructuredFlattenedWriter(HtmlSimpleWriter):
    """This writer combines outputs of several html writers. It formats them
    in a single-row table. The difference is that ``get_html_content()`` expects
    a tuple as values, with one value per component writer.

    The writer is (clumsily) designed to be used on top of an index-based
    FlattenedDatasetCorpus with the ``structured`` flag set. (This may be
    refactored in the near future if flattening is split into two blocks:
    reordering and combining.)
    """
    def __init__(self, root, writers, **kwargs):

        # Check that all writers are HTML writers
        for writer in writers:
            if not isinstance(writer, HtmlSimpleWriter):
                raise TypeError('Supplied a non-HtmlSimpleWriter writer to'
                                ' HtmlCompositeWriter: {0}'
                                ''.format(type(writer)))
        self.writers = writers
        super(HtmlStructuredFlattenedWriter, self).__init__(root, **kwargs)

    def get_html_content(self, iid, values, corpus):
        """Combines the value contents generated from each of the component
        writers. Assumes there is one value per writer. Relies on the
        ``generate_value()`` method of HtmlSimpleWriter (and its possible
        descendants)."""
        if len(values) != len(self.writers):
            raise ValueError('Must supply same number of values as there are '
                             'writers in the HtmlStructuredFlattenedWriter. (Values: {0},'
                             ' writers: {1})'.format(len(values),
                                                     len(self.writers)))
        if not isinstance(corpus, FlattenedDatasetCorpus):
            raise TypeError('Flattened dataset writer only works when the'
                            ' given corpus is the result of applying'
                            ' a FlattenComposite transformer.')

        elements = []

        individual_iids = corpus.obj.indexes[iid]
        heading_text = u'Safire introspection: composite IID {0}' \
                       u''.format(individual_iids)
        heading = html_utils.with_tag(heading_text, 'h1')
        elements.append(heading)

        composite = find_type_in_pipeline(corpus, CompositeDataset)
        sources = composite.data
        values = [self.writers[i].generate_value(individual_iids[i],
                                                 values[i],
                                                 sources[i])
                  for i in xrange(len(values))]
        combined_values = html_utils.as_table([values])
        elements.append(combined_values)

        links_table = self.generate_next_and_prev_links(iid)
        elements.append(links_table)

        text = u'\n'.join(elements)
        return text


class HtmlImageWriter(HtmlSimpleWriter):
    """This writer interprets the value as an image and instead of writing
    the value itself, will use the ``iid`` and id2doc mapping available through
    the transformer to link to the source image.

    .. note::

        For usage within the safire multimodal experiments, note that the image
        IDs do *not* have the ``img/`` prefix like the vtlist entries, so you
        need to supply the image directory in the writer ``root`` init arg.
    """
    def generate_value(self, iid, value, corpus):
        docname = str(get_id2doc_obj(corpus)[iid])
        image_abspath = os.path.join(self.root, docname)
        image_url = html_utils.as_local_url(image_abspath)
        text = html_utils.as_image(image_url)
        return text


class HtmlVocabularyWriter(HtmlSimpleWriter):
    """This writer interprets the value as references to a vocabulary and the
    accompanying frequencies. It will print out these word-frequency pairs."""
    def __init__(self, root, top_k=None, min_freq=0.001, **kwargs):
        super(HtmlVocabularyWriter, self).__init__(root, **kwargs)
        self.top_k = top_k
        self.min_freq = min_freq

    def generate_value(self, iid, value, corpus):
        vocabulary = get_id2word_obj(corpus)
        # Different types = different treatment
        freqs = {}
        if isinstance(value, numpy.ndarray):
            if len(value.shape) != 2:
                raise ValueError('Can currently handle only 2-dimensional '
                                 'ndarrays (with one row) as document matrices,'
                                 ' supplied ndarray shape: {0}'
                                 ''.format(value.shape))
            if len(value) > 1:
                raise ValueError('Cannot currently write vocabulary for more'
                                 ' than one document at once; supplied document'
                                 ' ndarray shape: {0}'.format(value.shape))
            # Load freqs from numpy ndarray
            for wid, freq in enumerate(value[0]):
                if freq >= 0.001:  # Guard agains various 32-vs-64 troubles
                    word = vocabulary[wid]
                    freqs[word] = freq

        # Differentiate between single item and multiple gensim items?
        elif isinstance(value, list) and len(value) > 0:

            if len([i for i in xrange(len(value)) if len(value[i]) != 2]) == 0:
                fw_pairs = value
            else:
                iscorp, _ = is_corpus(value)
                if iscorp:
                    if len(value) > 1:
                        raise ValueError('Cannot currently write vocabulary for'
                                         ' morethan one document at once; '
                                         'supplied value has more than one '
                                         'gensim-style vector of word-frequency'
                                         ' pairs (our best guess: {0}, but may'
                                         ' have more levels of nesting...)'
                                         ''.format(len(value)))
                fw_pairs = value[0]

            for item in fw_pairs:
                wid = item[0]
                freq = item[1]
                word = vocabulary[wid]
                freqs[word] = freq

        if len(freqs) == 0:
            return html_utils.with_tag(u'Empty document, iid: {0}.'.format(iid),
                                       u'div')

        # Sort
        wf_pairs = sorted(freqs.items(),
                          key=operator.itemgetter(1),
                          reverse=True)

        # Filter
        wf_pairs = wf_pairs[:self.top_k]
        wf_pairs = [(w, f) for w, f in wf_pairs if f >= self.min_freq]

        # Organize into a table
        wf_pairs_texts = [[u'{0}: {1}'.format(w, f)] for w, f in wf_pairs]
        wf_pairs_table = html_utils.as_table(wf_pairs_texts)
        return wf_pairs_table


class HtmlSimilarImagesWriter