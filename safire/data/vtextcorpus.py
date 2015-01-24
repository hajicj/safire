# -*- coding: utf-8 -*-
#
#
import codecs
import time
import logging
import re
import gzip
import os
#import cPickle

#from gensim import interfaces, utils
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary
import itertools

import filters.positional_filters as pfilters

logger = logging.getLogger(__name__)


def _strip_UFAL(token):
    """
    >>> _strip_ufal('blah_:W_some_suffix')
    'blah'
    """
    if '_' in token:
        idx = token.index('_')
        token = token[:idx]
    return token

# Various functions for filtering out sentences (positional filter).
# The first argument of each is the ``sentences`` list of lists.


class VTextCorpus(TextCorpus):
    """A text corpus that works on files in Vertical Text format.
    The vtext format has one token per line, with various columns.
    The column names are defined on initialization.

    Which columns should be returned is also specified on init.
    """
    def __init__(self, input=None, colnames=['form', 'lemma', 'tag'],
                 retcol='lemma', input_root=None,
                 delimiter='\t', gzipped=True,
                 sentences=False,
                 dictionary=None, allow_dict_updates=True,
                 doc2id=None, id2doc=None, token_filter=None,
                 token_transformer='strip_UFAL',
                 pfilter=None, pfilter_full_freqs=False,
                 filter_capital=False,
                 label=None, precompute_vtlist=True):
        """Initializes the text corpus.

        :param input: The input for a VTextCorpus is a handle
            with one document file name per line or a file that can be
            opened as such.

        :param colnames: A list of VText column names. The column names should
            be unique.

        :param retcol: The name of the column to return. (Has to be in
            colnames.)

        :param input_root: The path to prepend to each entry in input.

        :param delimiter: The character that delimits columns in the VText
            input (should usually be '\t', which is default).

        :param gzipped: Set to True if the vtext files are gzipped.

        :param sentences: Set to True if docs should not be entire files
            but rather sentences.

            .. warning::

                This option is scheduled for deprecation.

            .. warning::

                If you set ``sentences``, you MUST unset ``precompute_vtlist``,
                as the ``doc2id`` and ``id2doc`` mappings do NOT get precomputed
                on a per-sentence basis.

        :param dictionary: If specified, the corpus will share a dictionary.
            Useful for multiple VTextCorpus instances running on a single
            dataset.

        :param allow_dict_updates: If True, the supplied dictionary will not be
            updated with new items.

        :param id2doc: If specified, the corpus will use the given map
            from document IDs to their names, as given in input vtlist.

        :param doc2id: If specified, the corpus will use the given map
            from document names to their IDs.

            .. warn:

              There is no check that `id2doc` and `doc2id` contain inverse
              mappings!

        :param token_filter: A callable that returns bool on the vtext fields
            (the argument to token_filter will be a split line of the vtext,
            with members corresponding to the columns as defined by the
            ``colnames`` parameter.

        :param token_transformer: A callable that transforms a token before
            recording it into the corpus. For example, it may strip some
            common suffix, etc. If ``None``, leaves the token as-is.

            A shortcut is ``strip_UFAL``, which strips everything to
            the right of the first underscore found in the token (incl. the
            underscore). This strips unnecessary information from the lemmas
            obtained from MorphoDiTa, from the MorphLex lemma dictionary
            at UFAL. **This is currently the default behavior.**

        :param pfilter: A number, either an integer or a float. If
            an integer (K) is given, will only take the first K sentences from
            the beginning of a document. If a float is given, will use the given
            fraction of sentences from the beginning of a document

        :param pfilter_full_freqs: Used in conjunction with
            ``positional_filter``. If set, will use word frequencies from the
            entire document, but but only for the words that appear at positions
            that pass the positional filter.

        :param filter_capital: If set, will remove all words that start with
            a capital letter in their retfield. This serves to remove named
            entities.

        :param label: A "name" the corpus carries with it for identification.

        :param precompute_vtlist: If set, will load the list of vtext files in
            the ``input`` file on initialization. This will then enable running
            ``__getitem__`` with an integer lookup key (will retrieve the key-th
            document) and caching. Set to True by default. The only reason to
            unset this is when the list of *.vt files does not fit into memory;
            this may be an issue at around several hundred million *.vt files,
            so for example all Wikipedias in all languages are fine.
        """
        self.label = label

        if colnames is None:
            raise ValueError('Cannot initialize VTextCorpus without colnames.')
        if retcol is None:
            raise ValueError('Cannot initialize VTextCorpus without recol.')
        if retcol not in colnames:
            raise ValueError('Retcol name (%s) is not in colnames: %s' % (
                retcol, str(colnames)))

        # Input properties
        self.__do_cleanup = False

        self.input = input
        self.gzipped = gzipped
        self.input_root = input_root

        # Initialize dictionary
        if dictionary is None:
            dictionary = Dictionary()

        self.dictionary = dictionary
        self.allow_dict_updates = allow_dict_updates

        self.locked = False

        # Initialize document ID tracking
        if doc2id is None:
            doc2id = {}
        if id2doc is None:
            id2doc = []

        self.doc2id = doc2id
        self.id2doc = id2doc

        # Initialize filters
        self.token_filter = token_filter

        if token_transformer == 'strip_UFAL':
            token_transformer = _strip_UFAL

        self.token_transformer = token_transformer

        # Initialize column parsing
        self.delimiter = delimiter
        self.colnames = colnames
        self.retcol = retcol
        self.retidx = self.colnames.index(self.retcol)  # Already checked that
                                                        # retcol is in colnames
        self.filter_capital = filter_capital
        self.filter_capital_regex_str = u'[A-Z' + u'ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ' + u']'

        # Output behavior properties
        self.sentences = sentences

        # Initializes the positional filter
        self.positional_filter = None
        self.positional_filter_kwargs = {}
        self._init_positional_filter(pfilter, pfilter_full_freqs)

        # Processing stats
        self.n_processed = 0
        self.n_words_processed = 0

        # Precompute list of *.vt files?
        self.precompute_vtlist = precompute_vtlist
        self.vtlist = []
        if self.precompute_vtlist:
            self.vtlist = self._precompute_vtlist(self.input)

    def __iter__(self):
        """
        The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each
        document.
        """
        for text in self.get_texts():
            #logging.debug('__iter__ Yielding text: %s' % str(text))
            yield self.doc2bow(text)

    def doc2bow(self, text, allow_update=None):
        """Mini-method for generating BOW representation of text."""
        if allow_update is None:
            allow_update=self.allow_dict_updates
        bow = self.dictionary.doc2bow(text,
                                      allow_update=allow_update)

        return bow

    def doc_full_path(self, doc):
        """Given a vtlist entry, returns the full path to the vtlist file.
        (Simply combines ``self.input_root`` and ``doc``"""
        if self.input_root is not None:
            doc = os.path.join(self.input_root, doc.strip())
            #logger.debug(
            #    'Doc name after prepending with input root: %s' % doc)
        return doc

    def get_texts(self):
        """
        One iteration of get_texts() should yield one document, which means
        one file. Note that get_texts() can work even with vtlists that do not
        fit into memory, which would be a very rare occasion indeed.
        """
        batch_no = 1
        timed_batch_size = 1000
        start_time = time.clock()  # Logging time
        batch_start_time = start_time

        total_yielded = 0

        # Should re-set counters on each new iter() call..?
        self.n_processed = 0
        self.n_words_processed = 0

        input_handle = self.input
        if isinstance(self.input, str):
            input_handle = open(self.input)

        for docno, doc in enumerate(input_handle):
            # logger.debug('Processing doc no. %d: %s' % (docno, doc.strip()))

            doc = doc.strip()

            # NOTE: doc_short_name is the document name *before*
            # self.input_root is applied - it's the path to the file
            # from the corpus root. This is the preferred ID of documents.
            doc_short_name = doc

            doc = self.doc_full_path(doc)
            if not self.precompute_vtlist:
                self.vtlist.append(doc)

            # This is where the actual parsing happens.
            with self._get_doc_handle(doc) as doc_handle:

                document, sentences = self.parse_document_and_sentences(
                    doc_handle)

            # This will probably get deprecated (no sentences)
            if not self.sentences:

                if not self.precompute_vtlist:
                    docid = doc_short_name
                    self.doc2id[docid] = total_yielded
                    self.id2doc.append(docid)

                total_yielded += 1

                yield document

            else:

                for sentno, sentence in enumerate(sentences):
                    docid = doc_short_name + '.' + str(sentno)
                    self.doc2id[docid] = total_yielded
                    self.id2doc.append(docid)

                    #logger.debug('get_texts: Yielding sentence: %s' % str(
                    #    [tok for tok in sentence]))
                    total_yielded += 1

                    yield sentence

            self.n_processed += 1
            if self.n_processed % timed_batch_size == 0:
                batch_end_time = time.clock()
                batch_time = batch_end_time - batch_start_time
                batch_total_time = batch_end_time - start_time
                logger.info(
                    'Done batch no. %d, total docs %d. Batch time %f s (%f s / doc), total time %f s (%f s / doc)' % (
                        batch_no, self.n_processed, batch_time,
                        batch_time / timed_batch_size, batch_total_time,
                        batch_total_time / self.n_processed))
                batch_no += 1
                batch_start_time = time.clock()

        if isinstance(self.input, str):
            input_handle.close()

        end_time = time.clock()
        total_time = end_time - start_time
        logger.info('Processing corpus took %f s (%f s per document).' % (
            total_time, total_time / self.n_processed))

    def parse_document_and_sentences(self, doc_handle):
        """Parses the whole document. Returns both the sentences and the
        whole document, filtered by pfilter."""
        # Pre-parse entire document
        sentences = self.parse_sentences(doc_handle)

        # Process sentences and prepare document to return
        flt_sentences = sentences
        if self.positional_filter:
            flt_sentences = self._apply_positional_filter(sentences)

        document = list(itertools.chain(*flt_sentences))

        return document, sentences

    def parse_sentences(self, doc_handle):
        """
        Parses one vtext document into sentences, based on self.colnames and
        self.retcol. Interprets blank lines as sentence breaks.

        If a sentence filter is specified, applies the sentence filter.
        """
        logging.debug('Parsing sentences from handle %s' % str(doc_handle))

        filter_capital_regex = None
        if self.filter_capital:
            filter_capital_regex = re.compile(self.filter_capital_regex_str)

        sentences = []
        current_buffer = []
        for line_no, line in enumerate(doc_handle):
            # Quick fix?
            #if self.gzipped:
            #    line = line.decode('utf-8')

            #logging.debug('Line: %s' % line.strip())
            #safire.utils.check_malformed_unicode(line.strip())

            if line.strip() == '':
                #logger.debug('parse_sentences: Adding sentence at %d: %s' % (
                #    line_no, str(current_buffer)))
                sentences.append(current_buffer)
                current_buffer = []
                continue

            fields = line.strip().split(self.delimiter)
            if self.token_filter:

                if self.token_filter(fields):
                    ret_field = fields[self.retidx]
                    if self.token_transformer:
                        ret_field = self.token_transformer(ret_field)
                    if self.filter_capital and filter_capital_regex.match(ret_field):
                        continue
                    current_buffer.append(ret_field)
            else:
                ret_field = fields[self.retidx]
                if self.token_transformer:
                    ret_field = self.token_transformer(ret_field)
                if self.filter_capital and filter_capital_regex.match(ret_field):
                    continue
                current_buffer.append(ret_field)

        return sentences

    def dry_run(self):
        """Loops through the entire corpus without outputting the documents
        themselves, to generate the corpus infrastructure: document ID mapping,
        dictionary, etc."""
        for _ in self:
            pass

    def lock(self):
        """In order to use the corpus as a transformer, it has to be locked: it
        only reads texts and doesn't update anything."""
        self.allow_dict_updates = False
        self.locked = True

    def locked(self):
        """Checks whether the corpus is locked."""
        return self.locked and (self.allow_dict_updates == False)

    def unlock(self):
        self.allow_dict_updates = True
        self.locked = False

    def reset_input(self, vtlist_filename, input_root=None, lock=True):
        """Used to re-direct the VTextCorpus to a different *.vtlist input
        file. By default:

        * The input root is the same,
        * The corpus will be *locked*: the dictionary will not be updated.
          This is to reflect the usage: to load a VTextCorpus that was used
          for training and use it to correctly feed data to the higher-level
          corpora.

        :type vtlist_filename: str
        :param vtlist_filename: The new vtlist filename. To be consistent, this
            is only the raw filename without the path (more precisely, with only
            the path relative to ``self.input_root``, if the ``input_root``
            parameter is ``None``, or relative to the ``input_root`` path
            parameter).

        :type input_root: str
        :param input_root: If the corpus input root relative to which the
            *.vt.txt files in the new vtlist ``vtlist_filename`` are stored
            differs from the current input root, re-set it using this parameter.

        :type lock: bool
        :param lock: If the corpus should be locked, i.e. in read-only mode
            (no modifications to the dictionary/other behavior), set this.
            By default, lock is set.
        """
        if input_root:
            self.input_root = input_root
        self.input = vtlist_filename
        self.vtlist = []
        self.doc2id = {}
        self.id2doc = []
        if self.precompute_vtlist:
            self.vtlist = self._precompute_vtlist(self.input)
        if lock:
            self.lock()

    def __len__(self):
        """Computes the number of known documents in the corpus.

        If the list of
        *.vt files to process was precomputed, returns the total expected number
        of documents. If it was not precomputed, returns the number of documents
        processed from the current input. Note that resetting the input will
        also reset corpus length; for a total of documents retrieved from the
        corpus, see the ``n_processed`` attribute."""
        return len(self.vtlist)

    def __del__(self):
        if self.__do_cleanup:
            self.input.close()

    def __getitem__(self, item, allow_update=True):
        """If item is a *.vt file handle, returns the document representation.
        If item is an integer, returns representation of the item-th document
        from the current vtlist.

        .. note:: Retrieval by vthandle will update the dictionary unless
            ``allow_update`` is explicitly set to False!

        .. note::

            Retrieval by integer key will probably be the preferred method;
            retrieval by *.vt handle will probably be moved to a
            VTextCorpusTransformer class, as the usage pattern more closely
            mirrors a transformation of vertical text documents.

        Does NOT support retrieving sentences."""
        if self.sentences:
            raise ValueError('__getitem__ calls not supported when retrieving'
                             'sentences as documents.')
        if isinstance(item, int):
            if not self.precompute_vtlist:
                raise TypeError('Doesn\'t support indexing without precomputing'
                                'the vtlist.')
            with self._get_doc_handle(self.vtlist[item]) as vthandle:
                document, _ = self.parse_document_and_sentences(vthandle)

            return self.doc2bow(document, allow_update=allow_update)

        document, _ = self.parse_document_and_sentences(item)
        out = self.doc2bow(document, allow_update=allow_update)
        return out

    def _init_positional_filter(self, positional_filter, positional_full_freqs):
        """Initializes position-based filtering."""
        if isinstance(positional_filter, int):
            if positional_full_freqs:
                self.positional_filter = pfilters.words_from_first_k
            else:
                self.positional_filter = pfilters.first_k
            self.positional_filter_kwargs['k'] = positional_filter
        elif isinstance(positional_filter, float):
            if positional_full_freqs:
                self.positional_filter = pfilters.words_from_first_p
            else:
                self.positional_filter = pfilters.first_p
            self.positional_filter_kwargs['p'] = positional_filter

    def _apply_positional_filter(self, sentences):
        """
        Based on the settings of ``positional_filter`` and ``positional_freqs``,
        filters the ordered sentences.

        If the positional filter is an integer N, will use the first N
        sentences. If the positional filter is a floating-point number, will
        use

        :param sentences: The words that will be passed to the dicitonary's
            doc2bow method.

        :return: The filtered document.
        """
        return self.positional_filter(sentences, **self.positional_filter_kwargs)

    def _get_doc_handle(self, doc):
        if self.gzipped:
            gz_fh = gzip.open(doc)
            doc_reader = codecs.getreader('utf-8')
            return doc_reader(gz_fh)
        else:
            return codecs.open(doc, 'r', 'utf-8')

    def _precompute_vtlist(self, input):
        # Should also compute the doc2id and id2doc mapping.
        # Does NOT support sentences.
        if not isinstance(input, str):
            raise TypeError('Cannot precompute vtlist from a handle,'
                            ' must supply parameter input as filename.')
        vtlist = []
        with open(input) as vtl_handle:
            for i, vtname in enumerate(vtl_handle):
                doc_short_name = vtname.strip()
                self.id2doc.append(doc_short_name)
                self.doc2id[doc_short_name] = i

                doc_full_name = self.doc_full_path(doc_short_name)
                vtlist.append(doc_full_name)
        return vtlist
