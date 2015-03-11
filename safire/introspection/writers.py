"""
This module contains classes that ...
"""
import logging
import os
from safire.introspection import html_utils

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
    """This class writes a simple HTML output into a table.
    """

    def __init__(self, *args, **kwargs):
        super(HtmlSimpleWriter, self).__init__(*args, **kwargs)

    @staticmethod
    def iid_to_filename(iid):
        return 'introspection.{0}.html'

    def generate_filename(self, iid):
        abs_filename = os.path.abspath(os.path.join(self.root,
                                                    self.iid_to_filename(iid)))
        return self.url_separator.join([self.url_local_prefix,
                                        abs_filename])

    def write(self, output_handle, iid, value, introspection_transformer):

        header = self.get_html_header()
        output_handle.write(header + '\n\n')

        body = self.get_html_body(iid, value, introspection_transformer)
        output_handle.write(body + '\n\n')

        footer = self.get_html_footer()
        output_handle.write(footer + '\n\n')

    def get_html_header(self):
        return html_utils.head

    def get_html_footer(self):
        return html_utils.as_comment('Generated using HtmlSimpleWriter.')

    def get_html_body(self, iid, value, introspection_transformer):

        elements = []

        # Create heading.
        docname = introspection_transformer.id2doc[iid]
        heading = u'Safire introspection: iid {0}, docname {1}' \
                  u''.format(iid, docname)
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
        return html_utils.text_with_tag(text, 'body', newline=True)
