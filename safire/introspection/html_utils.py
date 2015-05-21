"""
This module contains functions for generating html code. Note that all output strings
are unicode.
"""
import logging
import os

__author__ = "Jan Hajic jr."

# ---------------------------------------------------------------------------- #

# Constants

#: This is the URL prefix for browsing local files.
url_local_prefix = 'file://localhost'

#: Cannot use os.path.join(), have to use url_separator.join()
url_separator = '/'

#: Universal safire HTML introspection output header. Note that it should not be
#: used unless html_utils.foot is used at the end of the html document.
head = '''
<!DOCTYPE html>
<html lang="en" class="">
<head>
<meta charset="utf-8"/>
<title>Safire introspection</title>
</head>
'''

#: Must close all unclosed tags from head. Never use html_utils.head without
#: adding html_utils.foot at the end of the document!
foot = '''
</html>
'''

comment_start = '<!--'
comment_end = '-->'

# ---------------------------------------------------------------------------- #

# Utility functions: elementary building blocks


def as_comment(text, newline=False):
    """Formats the given string as an HTML comment. If ``newline`` is set, will
    write the opening and closing tag on separate lines. No newline is inserted
    after the closing tag.

    >>> print as_comment("comment")
    <!-- comment -->
    >>> print as_comment("comment with newlines", newline=True)
    <!--
    comment with newlines
    -->

    """
    if newline is True:
        commented_text = '\n'.join([comment_start, text, comment_end])
    else:
        commented_text = ' '.join([comment_start, text, comment_end])
    return commented_text


def as_attributes(**attributes):
    """Formats the given key-value pairs as a html tag attributes string.

    >>> as_attributes(width=159, height=392)
    u'width="159" height="392"'

    """
    if len(attributes) == 0:
        return u''
    formatted_atrributes = []
    for k, v in attributes.items():
        formatted_atrributes.append(u'{0}="{1}"'.format(k, v))
    attribute_string = ' '.join(formatted_atrributes)
    return attribute_string


def as_tag(tagname, pair=True, **attributes):
    """Formats the given string as a html tag. Returns <tagname>, </tagname>
    pairs. If the ``attributes`` kwarg dict is supplied, adds its key-value
    pairs as tag attributes.

    >>> as_tag('body')
    (u'<body>', u'</body>')
    >>> as_tag('img', pair=False, src="file://localhost/C:/Users/Guest/logo.png")
    u'<img src="file://localhost/C:/Users/Guest/logo.png"/>'
    """
    formatted_attrs = as_attributes(**attributes)
    if pair is True:
        if formatted_attrs != '':   # So that we don't get <body >
            formatted_attrs = u''.join([u' ', formatted_attrs])
        open_tag = u''.join([u'<', tagname, formatted_attrs, u'>'])
        closing_tag = u'</' + tagname + u'>'
        return open_tag, closing_tag
    else:
        tag = u'<' + tagname + u' ' + formatted_attrs + u'/>'
        return tag


def text_with_tag(text, tagname, newline=False, **attributes):
    """Envelopes the given text in a tag with the given name. If ``newline`` is
    set, will write the opening and closing tag on separate lines. No newline is
    inserted after the closing tag. Note that the function only accepts strings;
    for arbitrary data, use :function:`with_tag()`.

    Attributes for the tag can be given, but watch out for reserved python
    names if you use this in code. It's on the safe side to first gather the
    attributes into a dictionary, then pass the dictionary as kwargs.

    >>> text_with_tag('A paragraph.', 'p', **{'class': 'basic'})
    u'<p class="basic">A paragraph.</p>'
    >>> print text_with_tag('A paragraph.', 'p', newline=True)
    <p>
    A paragraph.
    </p>

    """
    tag_start, tag_end = as_tag(tagname, pair=True, **attributes)
    if newline is True:
        output = u'\n'.join([tag_start, text, tag_end])
    else:
        output = u''.join([tag_start, text, tag_end])
    return output


def with_tag(data, tagname, newline=False, **attributes):
    """Envelopes the given data in a tag with the given name. If ``newline`` is
    set, will write the opening and closing tag on separate lines. No newline is
    inserted after the closing tag. To convert data to string, the function uses
    ``u'{0}'.format(data)``.

    Attributes for the tag can be given, but watch out for reserved python
    names if you use this in code. It's on the safe side to first gather the
    attributes into a dictionary, then pass the dictionary as kwargs.

    >>> print with_tag('A paragraph.', 'p', **{'class': 'basic'})
    <p class="basic">A paragraph.</p>
    >>> print with_tag(42, 'td')
    <td>42</td>

    """
    return text_with_tag(u'{0}'.format(data), tagname, newline=newline,
                         **attributes)


# ---------------------------------------------------------------------------- #

# Specific html entities (tables, links, images...)

def as_table(data):
    """Formats the given data (assumes a list of lists, or something that can
    be iterated over using a nested for-loop) as a table.

    Some tabspacing is done (``\t`` for cell tags).

    >>> print as_table([[1, 2], [3, 4]])
    <table>
    <tr>
        <td>1</td>
        <td>2</td>
    </tr>
    <tr>
        <td>3</td>
        <td>4</td>
    </tr>
    </table>
    """
    table_rows = []
    for row in data:
        cells = [with_tag(c, 'td') for c in row]
        table_rows.append('    ' + '\n    '.join(cells))

    rows_as_tags = [with_tag(r, 'tr', newline=True) for r in table_rows]
    rows_combined = '\n'.join(rows_as_tags)
    table = with_tag(rows_combined, 'table', newline=True)
    return table


def as_link(url, text=None):
    """Formats the given url as a link. If the text is given, uses it as the link
    text, otherwise uses the url as the link text.

    >>> as_link('file://localhost/C:/blah.html')
    u'<a href="file://localhost/C:/blah.html">file://localhost/C:/blah.html</a>'
    """
    if text is None:
        text = url
    return with_tag(text, 'a', newline=False, href=url)


def as_image(url, **attributes):
    """Formats the given URL as an image.

    >>> as_image('file://localhost/C:/foo.png')
    u'<img src="file://localhost/C:/foo.png"/>'
    """
    return as_tag('img', pair=False, src=url, **attributes)

# ---------------------------------------------------------------------------- #

# URL tools


def as_local_url(path):
    """Formats the given path as a local url (typically prefixes the path with
    ``file://localhost/``). Checks that the given path exists and logs a warning
    if it doesn't.

    >>> as_local_url('blah')
    'file://localhost/blah'
    """
    if not os.path.exists(path):
        logging.debug('Creating link to nonexistent file: {0}'.format(path))

    return url_separator.join([url_local_prefix, path])


# ---------------------------------------------------------------------------- #

# Safire-specific tools