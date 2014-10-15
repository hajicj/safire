#!/usr/bin/env python
"""
Maps image IDs from an ImagenetCorpus to image files for exploring
Safire model performance.
"""
import codecs
import copy
import logging
import os
import random
import string
import webbrowser
import StringIO
import operator
import gensim
import math
import safire.utils
from safire.data import VTextCorpus
from safire.utils.transcorp import id2word, bottom_corpus, run_transformations, \
    get_transformers


__author__ = 'Jan Hajic jr.'


class TextBrowser(object):
    """
    Provides functionality for looking directly at the texts used to build
    queries.
    """

    def __init__(self, root, corpus, gzipped=True, text2im=None, im2text=None,
                 first_n_sentences=None):
        """
        :type root: str
        :param root: Path to the directory relative to which files in the
            ids2files_map are given. Used in retrieving images. Best practice:
            use an absolute path.

        :type ids2files: dict
        :param ids2files: A dictionary that maps a document ID in corpus to a file
            relative to root.

        :type corpus: safire.data.vtextcorpus.VTextCorpus
        :param corpus: A VTextCorpus, from which document parsing settings
            will be taken. A deep copy is made, because we will be manipulating
            the corpus settings during TextBrowser operation.

        :type gzipped: Boolean
        :param gzipped: If set, will consider the text files with corpus docs
            gzipped.

        :type text2im: dict
        :param text2im: Optionally, a text-image mapping can be supplied.
            This enables looking up image IDs for texts.

        :type im2text: dict
        :param im2text: Optionally, an image-text mapping can be supplied.
            This enables looking up text IDs for images.

        :type first_n_sentences: int
        :param first_n_sentences: Only show this many first sentences from each
            displayed text file.
        """
        self.root = root

        # Loading corpus and applying browser settings
        self.corpus = copy.deepcopy(corpus) # This may take long(-ish).
        self.transformers = get_transformers(self.corpus)

        self.vtcorp = bottom_corpus(self.corpus)
        if not isinstance(self.vtcorp, VTextCorpus):
            raise ValueError('Wrong corpus type supplied: expects VTextCorpus, not %s' % type(self.vtcorp))
        self.vtcorp.allow_dict_updates = False # Freeze dictionary

        self.ids2files = self.vtcorp.id2doc # Ref...
        self.files2ids = self.vtcorp.doc2id

        self.gzipped = gzipped

        # "Bonus" members: if given, a text-image mapping will be available.
        self.text2im = text2im
        self.im2text = im2text

        self.first_n_sentences = first_n_sentences

        # Internals
        self.prev_temp = None
        self.to_delete = [] # Temporary files to clean up later.
        self.prev_docid = None


    def __getitem__(self, id_or_filename):
        """Gets the text file name, incl. path to root, if an ID is given.
        If a filename is given (relative to the root), returns the corresponding
        ID. Can also parse a filename given together with the root.

        Can parse an iterable of IDs/filenames; returns the corrseponding list
        (may be a mix of IDs and filenames, but that is not recommended).

        :raises: KeyError
        """
        if isinstance(id_or_filename, list) or isinstance(id_or_filename, tuple):
            return [self[item] for item in id_or_filename]
        else:
            if isinstance(id_or_filename, int):
                return os.path.join(self.root, self.ids2files[id_or_filename])
            elif id_or_filename in self.files2ids:
                return self.files2ids[id_or_filename]
            elif id_or_filename.startswith(self.root):
                short_filename = id_or_filename[len(self.root):]
                if short_filename in self.files2ids:
                    return self.files2ids[short_filename]

            raise KeyError('Image with ID or filename %s not found.' % id_or_filename)

    def parse(self, filename, first_n_sentences=None):
        """From a vtext filename, retrieves the raw text of the document, grouped
        by sentence."""
        temp_retcol = self.vtcorp.retcol
        temp_retidx = self.vtcorp.retidx
        temp_filter = self.vtcorp.token_filter

        retcol = 'form'

        try:

            self.vtcorp.retcol = retcol
            self.vtcorp.retidx = self.vtcorp.colnames.index(retcol)
            self.vtcorp.token_filter = None

            with self.vtcorp._get_doc_handle(filename) as input_handle:

                sentence_buffer = self.vtcorp.parse_sentences(input_handle)
                text = []
                for sent in sentence_buffer:
                    ### DEBUG
                    #for w in sent:
                    #    safire.utils.check_malformed_unicode(w)

                    text.extend(sent)
                #print 'Text: %s' % text
                doc = self.vtcorp.dictionary.doc2bow(text, allow_update=False)
                #print 'Doc: %s' % str(doc)


        finally:
            self.vtcorp.retcol = temp_retcol
            self.vtcorp.retidx = temp_retidx
            self.vtcorp.token_filter = temp_filter

        if self.first_n_sentences and not first_n_sentences:
            first_n_sentences = self.first_n_sentences
        if first_n_sentences:
            if len(sentence_buffer) > first_n_sentences:
                sentence_buffer = sentence_buffer[:first_n_sentences]

        return doc, sentence_buffer

    def create_header(self, filename):
        """Creates a header for the shown document."""
        line1 = 'Document file: %s' % filename
        line2 = '=' * len(line1)
        return '\n'.join([line1, line2])

    def format_text(self, bow, sentence_buffer, header):
        """Formats the text in the sentence buffer to something printable."""
        output_lines = [header]

        for sentence in sentence_buffer:
            output_line = ' '.join(sentence)
            output_lines.append(output_line)

        #output_text = '\n'.join((l for l in output_lines))
        s = StringIO.StringIO()
        s.write('\n'.join(output_lines))

        output = s.getvalue()

        return output

    def text_to_window(self, text):
        """Opens a new window and prints the formatted text there."""
        if self.prev_temp and os.path.isfile(self.prev_temp):
            try:
                os.remove(self.prev_temp)
            except Exception:
                if os.path.isfile(self.prev_temp): # Deletion unsuccessful
                    self.to_delete.append(self.prev_temp)

        tmp_file = self._generate_tmp_filename()

        tmp_handle = codecs.open(tmp_file, 'w', encoding='utf-8')
        tmp_handle.write(text + '\n')
        tmp_handle.close()

        webbrowser.open(tmp_file)

        self.prev_temp = tmp_file

    def get_text(self, docid, first_n_sentences=None):
        """For the given document ID, builds the text to show.

        If ``docid`` is an integer, will attempt to convert it to filename."""
        if isinstance(docid, int):
            filename = self[docid]
        else:
            filename = docid # Assumes document IDs are also filenames. This
                             # holds for corpora created from vtlists.
            filename = os.path.join(self.root, filename)

        logging.debug('Reading filename: %s' % filename)

        doc, sentence_buffer = self.parse(filename, first_n_sentences)
        header = self.create_header(filename)
        text = self.format_text(doc, sentence_buffer, header)
        return text

    def get_multiple_texts(self, docids, first_n_sentences):
        """Retrieves texts for the given document IDs."""
        texts = [self.get_text(docid, first_n_sentences) for docid in docids]
        final_text = '\n\n'.join(texts)
        return final_text

    def show(self, docid, first_n_sentences=None):
        """Load the given text in a separate window."""
        text = self.get_text(docid, first_n_sentences)
        self.text_to_window(text)

    def show_multiple(self, docids, first_n_sentences=None):
        """Load the given set of texts in a separate window."""
        final_text = self.get_multiple_texts(docids, first_n_sentences)
        self.text_to_window(final_text)

    def get_representation(self, docid):
        """Returns how the document is represented by the corpus: the word IDs
        in the document and their output values."""
        if isinstance(docid, int):
            filename = self[docid]
        else:
            filename = docid

        with self.vtcorp._get_doc_handle(filename) as doc_handle:
            representation = run_transformations(doc_handle, *self.transformers)
            #document, sentences = self.vtcorp.parse_document_and_sentences(doc_handle)

        return representation

    def get_word_representation(self, docid, highest_scoring=None):
        """Returns the document representation with word IDs converted to the
        original tokens.

        If highest_scoring is given, sorts from highest to lowest freq."""
        doc = self.get_representation(docid)

        # Need to backtrack through feature ID mappings
        word_doc = [ (unicode(id2word(self.corpus, w[0])), w[1])
                     for w in doc ]

        sorted_wdoc = word_doc
        if highest_scoring:
            sorted_wdoc = sorted(word_doc, key=operator.itemgetter(1), reverse=True)

            if highest_scoring < len(sorted_wdoc):
                sorted_wdoc = sorted_wdoc[:highest_scoring]

        return sorted_wdoc

    def format_representation(self, word_doc, n_cols=3):
        """Returns a string for printing out the document representation.

        :param word_doc: A document representation.

        :param n_cols: How many columns should the output have? [NOT IMPL.]
        """
        swdoc_pairs = [ '\t'.join([w[0], str(w[1])]) for w in word_doc ]
        columns = gensim.utils.grouper(swdoc_pairs,
                                       math.ceil(len(swdoc_pairs) / float(n_cols)))
        swdoc_row_tuples = map(list, zip(*columns))

        swdoc_rows = [ '\t\t'.join(row) for row in swdoc_row_tuples]
        output = '\n'.join(swdoc_rows)

        return output


    def _generate_tmp_filename(self):
        code = ''.join(random.choice(string.ascii_uppercase + string.digits)
                       for _ in xrange(16))
        return 'text.' + code + '.tmp.txt'

    def __del__(self):
        if self.prev_temp and os.path.isfile(self.prev_temp):
            os.remove(self.prev_temp)
        for filename in self.to_delete:
            if os.path.isfile(filename):
                os.remove(filename)