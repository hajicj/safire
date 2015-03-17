"""
This module contains classes that ...
"""
import logging
import os
from safire.introspection import html_utils
from safire.utils.transcorp import get_id2doc_obj

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

    def run(self, iid, value, obj):
        """Writes to a file and returns the filename.

        :param iid: The item ID, based on which the source document of the item
            can be retrieved.

        :param value: The item itself.

        :param obj: The introspection object. Used by the ``write()`` method to
            extract additional information.
        """
        filename = self.generate_filename(iid)
        with open(filename, 'w') as write_handle:
            self.write(write_handle, iid, value, obj)
        return filename

    def write(self, output_handle, iid, value, obj):
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

    def __init__(self, root, *args, **kwargs):
        super(HtmlSimpleWriter, self).__init__(root, *args, **kwargs)

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

    def write(self, output_handle, iid, value, introspection_transformer):
        """Generates the introspection result and writes it into the given
        file handle."""
        header = self.get_html_header()
        body = self.get_html_body(iid, value, introspection_transformer)
        footer = self.get_html_footer()

        print 'Document:\n'
        print header
        print body
        print footer

        output_handle.write('\n\n'.join([header, body, footer]))

    @staticmethod
    def get_html_header():
        """Generates the constant part of the HTML output that comes *before*
        the content."""
        return html_utils.head

    @staticmethod
    def get_html_footer():
        """Generates the constant part of the HTML output that comes *after*
        the content."""
        return u'\n'.join(
            [html_utils.as_comment('Generated using HtmlSimpleWriter.'),
             html_utils.foot])

    def get_html_content(self, iid, value, introspection_transformer):
        """Generates the "interesting" part of the introspection file:
        formats the contents of the item that is being introspected."""
        elements = []

        # Create heading.
        docname = get_id2doc_obj(introspection_transformer.corpus)[iid]
        print 'Docname: {0}'.format(docname)
        print 'Value: {0}'.format(value)
        absolute_docname = os.path.join(self.root, docname)
        heading = u'Safire introspection: iid {0}, docname {1}' \
                  u''.format(iid, absolute_docname)
        elements.append(html_utils.with_tag(heading, 'h1'))

        # Paste content.
        elements.append(html_utils.with_tag(value, 'div'))

        # Add next/prev links.
        prev_file = html_utils.as_local_url(self.generate_filename(iid - 1))
        prev_link = html_utils.as_link(prev_file, 'Previous item')

        next_file = html_utils.as_local_url(self.generate_filename(iid + 1))
        next_link = html_utils.as_link(next_file, 'Next item')

        links_table = html_utils.as_table([[prev_link, next_link]])
        elements.append(html_utils.with_tag(links_table, 'div'))

        # Combine elements into a single string and wrap in a <body> tag.
        text = u'\n'.join(elements)
        return text

    def get_html_body(self, iid, value, introspection_transformer):
        """Formats the content as the ``<body>`` element of an html file."""
        content = self.get_html_content(iid, value, introspection_transformer)
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
    def __init__(self, root, writers, *args, **kwargs):
        self.writers = writers
        super(HtmlStructuredFlattenedWriter, self).__init__(root, *args, **kwargs)

    def get_html_content(self, iid, values, introspection_transformer):
        """Combines the contents generated from each of the component writers.
        Assumes there is one value per writer."""
        if len(values) != len(self.writers):
            raise ValueError('Must supply same number of values as there are '
                             'writers in the HtmlStructuredFlattenedWriter. (Values: {0},'
                             ' writers: {1})'.format(len(values),
                                                     len(self.writers)))
        iids = introspection_transformer.corpus.obj.indexes[iid]
        contents = [self.writers[i].get_html_content(iids[i],
                                                     values[i],
                                                     introspection_transformer)
                    for i in xrange(len(values))]
        combined_content = html_utils.as_table([contents])
        return combined_content