#!/usr/bin/env python
"""
``multimodal_pipeline.py`` is a tutorial that trains a multimodal pipeline
from some test data. It shows how to set up a complex training
scenario with multiple data sources.
"""
import argparse
import logging
import os
from gensim.models import TfidfModel
import numpy

import safire
from safire.data.layouts import DataDirLayout
from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.serializer import Serializer
from safire.learning.models import DenoisingAutoencoder
from safire.learning.learners import BaseSGDLearner
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
from safire.learning.interfaces import SafireTransformer
from safire.datasets.transformations import docnames2indexes, FlattenComposite
from safire.utils import parse_textdoc2imdoc_map
from safire.utils.transformers import GeneralFunctionTransform
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter

__author__ = 'Jan Hajic jr.'

# These variables are settings for the pipeline. You can ignore them for now
# and come back to them when you understand how a pipeline is built up.
# The individual steps are heavily commented.

# Where are our data?
data_root = safire.get_test_data_root()
layout = DataDirLayout('test-data')
# The DataDirLayout defines how a directory for safire experiments is structured
# (but you don't really have to use that layout, it just makes life a bit easier
# if you do more experiments on the same data).

# Parameters used in text pipeline:

# This is a dictionary of keyword arguments for the VTextCorpus constructed in
# build_text_pipeline(). Most of the entries are the defaults for files output
# by UFAL's MorphoDiTa tagger.
vtcorp_settings = {'colnames': ['form', 'lemma', 'tag'],
                   # This specifies that the vertical text will have 3 columns.
                   # For the purposes of the corpus, their names are 'form',
                   # 'lemma' and 'tag' (in this order). The vertical text format
                   # doesn't really define which columns there are and what they
                   # should be called, it only defines that there's one token
                   # per line and that sentences are separated by a blank line.
                   'retcol': 'lemma',
                   # This specifies that the vocabulary of the corpus will be
                   # built from entries in the 'lemma' column. (We could also
                   # have a vocabulary of forms, if necessary, or of tags, for
                   # some morpho- or syntactic features.) We could also use
                   # 'first', 'second' and 'third' as the colnames parameter
                   # and set retcol to 'second', or any other naming scheme, as
                   # long as retcol is found in colnames.
                   # Even if your vertical text files only have one column, you
                   # have to specify the retcol.
                   'delimiter': '\t',
                   # This specifies that columns of the vtext file are separated
                   # by tabspace. Nothing more, nothing less.
                   'gzipped': True,
                   # This specifies that the individual vertical text files
                   # have been compressed using gzip.
                   'token_transformer': 'strip_UFAL',
                   # This parameter ordinarily accepts a function that is
                   # applied to each token before accepting it into the
                   # vocabulary. The 'strip_UFAL' is a shortcut for stripping
                   # MorphoDiTa output lemmas to their short form, e.g.
                   # na-1_:W to na-1. (Disambiguation markers like -1, -2 etc.
                   # are retained.) We lied a bit in the 'retcol' explanation:
                   # the vocabulary of the corpus will be built from entries
                   # in the 'retcol' column, processed by the token_transformer.
                   # (Token_transformer can also be None and the exact value
                   # from retcol is retained.)
                   'token_filter': PositionalTagTokenFilter(['N', 'A', 'V'], 0),
                   # This sets up filtering by part of speech. Specifically,
                   # we want to retain all tokens that have the values 'N', 'A'
                   # or 'V' in the 0-th position of their tag. This filter is
                   # specifically applied only to the 'tag' column, but it can
                   # be extended using its 'tag_colname' attribute to filter
                   # other columns (such as the 'form' column for something
                   # that starts with a capital letter).
                   'pfilter': 0.2,
                   # This parameter sets up filtering by position in document,
                   # retaining tokens from only the first 0.2 of a document.for
                   # For our newspaper data, we found that this is a reasonable
                   # way of extracting only the keywords relevant to the scope
                   # of the article as a whole and not partial topics.
                   'pfilter_full_freqs': True,
                   # In order to better capture the relative importance of the
                   # keywords found in the first 'pfilter' proportion of the
                   # text, we compute these keywords' frequencies from the
                   # whole document, not just this first 'pfilter' proportion.
                   'filter_capital': True,
                   # This setting throws away all tokens that start with a
                   # capital letter. Note that this applies to the 'retcol'
                   # column, so in our case, we throw away *lemmas* that are
                   # capitalized, not forms.
                   'precompute_vtlist': True,
                   # This setting loads the list of vertical text files into
                   # memory at initialization. This way, our VTextCorpus will
                   # be able to respond to __getitem__ requests: for the i-th
                   # document, it will process the vertical text file that is
                   # i-th in the vtlist. Also, we will know the length of
                   # the corpus -- the number of documents -- right at
                   # construction time.
                   # Technically, this option can be switched off, which can be
                   # useful in some online setting where the input file is
                   # a socket and iterating over it will gradually yield more
                   # and more documents. However, in that case, the corpus will
                   # only support iterating over it one by one, not
                   # random-access retrieval, and dimensionality data (which are
                   # derived from the size of the VTextCorpus vocabulary) will
                   # never be quite reliable (until we forbid updating the
                   # vocabulary, which can also be done -- but more on that
                   # elsewhere).
                   }
vtlist = os.path.join(data_root, layout.vtlist)
serialization_vtname = 'serialized.vt.shcorp'
serialization_vtfile = os.path.join(data_root,
                                    layout.corpus_dir,
                                    serialization_vtname)

# Used in image pipeline
icorp_settings = {'delimiter': ';',
                  'dim': 4096,
                  'label': ''}
image_vectors = os.path.join(data_root,
                             layout.image_vectors)
serialization_iname = 'serialized.i.shcorp'
serialization_ifile = os.path.join(data_root,
                                   layout.corpus_dir,
                                   serialization_iname)

serialization_mmname = 'serialized.mmdata.shcorp'
serialization_mmfile = os.path.join(data_root,
                                    layout.corpus_dir,
                                    serialization_mmname)

###############################################################################


# Pipeline building is split into three parts: the text pipeline, the image
# pipeline and combining them. Each of these parts is contained in a funciton.
def build_text_pipeline():
    """This function creates an example text pipeline. There are several stages.
    The stages will be commented upon throughout the function."""

    vtcorp = VTextCorpus(input=vtlist,
                         input_root=data_root,
                         **vtcorp_settings)
    # TL;DR: VTextCorpus is the guy that feeds "raw" text data as per-document
    # vectors.
    #
    # VTextCorpus is the class that allows us to read text data from vertical
    # text files. These files are produced for example by the MorphoDiTa tagger,
    # which we used to lemmatize our texts. The VTextCorpus needs a "vtlist"
    # file: a master file that tells it where to look for the individual files
    # with the tagged texts. This means that your lemmatized files can be all
    # over the place, as long as they are recorded in the vtlist file.
    #
    # VTextCorpus is used to iterate over *documents*: one "read" of VTextCorpus
    # corresponds to reading one entire vertical text file.
    #
    # The vtlist file contains one path to a vertical text document per line.
    # This path can be either absolute, or relative. If it is a relative path,
    # the ``input_root`` parameter will tell VTextCorpus relative to which
    # directory the paths in the vtlist are.
    #
    # Other VTextCorpus settings control various processing options that should
    # happen at the word-level. Because VTextCorpus iterates over entire
    # documents, all processing that applies to individual words or sentences
    # must be done inside VTextCorpus. Some of the supported operations are
    # filtering by part-of-speech tag or taking only words that appear in the
    # beginning of the document.

    vtcorp.dry_run()
    # TL;DR: Because VTextCorpus is lazy and we need some info about the data
    # which we don't get until we've seen it all, we need to force it to read
    # the data.
    #
    # Initializing VTextCorpus does *not* read the data, it only "prepares the
    # groundwork" like initializing all token-based processing steps or slurping
    # up the list of files to process to compute how many documents the corpus
    # will contain. However, one piece of information cannot be obtained without
    # iterating through the data: finding the size of the vocabulary. While
    # this doesn't matter when we are working with sparse vectors, as soon as
    # we want to train a neural network or something, we need to know the
    # dimension of the space in which the documents live.
    #
    # (Don't worry about efficiency too much. VTextCorpus caches the results of
    # document retrieval calls, so you will not be doing all the file opening
    # and closing twice on subsequent processing. Think about dry_run() as
    # forcing a non-lazy read of the data, instead of lazily deferring it to
    # some subsequent step in the pipeline.)

    pipeline = vtcorp
    # The pipeline is nothing else than a chaining of corpus and/or dataset
    # objects. On the bottom of each pipeline is a corpus that actually reads
    # the data from some kind of storage; the other stages of the pipeline then
    # transform this data.

    tfidf = TfidfModel(pipeline)
    pipeline = tfidf[pipeline]
    # TL;DR: This stacks a transformation block on top of the pipeline. First,
    # the transformation needs to be initialized (and this specific transform
    # needs to see the data we're using, so we give it the pipeline up to now),
    # then we stack the transformer block on top of the pipeline.
    #
    # We apply our first transformation of the day: the TF-IDF model.
    # Transformations are applied in two steps:
    #
    # * initialize the transformer object (in this case, we need to compute
    #   the term frequencies and inverse document frequencies),
    # * apply the transformer to our pipeline.
    #
    # These two steps are kept separate because you might want to apply
    # a transformer trained initialized on one corpus to a different corpus,
    # for example when evaluating your experiment on some unseen test data.
    # Note that the second step calls __getitem__ of the initialized
    # transformer, not __call__!

    freqfilter = FrequencyBasedTransformer(pipeline,
                                           k=110,
                                           discard_top=10)
    pipeline = freqfilter[pipeline]
    # TL;DR: Another transform block, this time to keep the top 11th to 110th
    # most frequent word.
    #
    # We want to only retain some more frequent words in the vocabulary. The
    # FrequencyBasedTransformer can do this for us. It is another transformer
    # object like the TfidfModel, this time a part of safire rather than gensim.
    # In the settings, we told it to keep 110 most frequent words but discard
    # also the top 10 most frequent, so we will be left with a vocabulary of
    # 100.

    tanh = GeneralFunctionTransform(numpy.tanh,
                                    multiplicative_coef=0.1)
    pipeline = tanh[pipeline]
    # TL;DR: Yet another transformation, this time through a sigmoid to squish
    # the inputs to (0, 1).
    #
    # Because we will be later feeding the results into a neural network that
    # will expect input values between 0 and 1, we have to normalize our data.
    # The hyperbolic tangent is a sigmoid function that we can use. However,
    # it's not the only one and you may also wish to apply various other
    # functions, so safire provides a GeneralFunctionTransform that will be
    # applied to each element of each document vector individually.
    #
    # In addition to choosing a function, we can also choose a multiplicative
    # coefficient by which each element will be multiplied before applying the
    # function. (There's also an additive coefficient option which we don't
    # use here.) In our case, we want to "slope out" the tanh sigmoid for longer
    # so that there is better discrimination between frequencies like 2 and 3
    # versus 20 or 30; if we used the raw tanh function, these would all be
    # very close to 1, which would make us lose quite some information.
    # (Check this out: try plotting tanh(x) versus tanh(0.1x); a simple
    # online plotter I used to estimate these constants is fooplot.com.)

    serializer = Serializer(pipeline, ShardedCorpus,
                            fname=serialization_vtfile)
    pipeline = serializer[pipeline]
    # TL;DR: So far we've only built the pipeline, now we run the data through
    # the transformation blocks and save them. This saves time for further
    # processing.
    #
    # Now that our pipeline is built, we want to save the results. This is
    # an important step for efficiency -- later on, when training a neural
    # network model or anything else, we will need repeated random access to
    # the data, and drawing them all the way through the pipeline again and
    # again would slow our progress.
    #
    # Saving the data -- or, as it is called in safire (and gensim),
    #  *serializing*  a pipeline -- is done using the Serializer class. We need
    #  to supply three things to a Serializer:
    #
    # * the pipeline to serialize,
    # * the definition of the serialization format,
    # * the file to which to serialize.
    #
    # The definition of the serialization format is handled by any class that
    # implements the IndexedCorpus interface from gensim. Safire provides the
    # ShardedCorpus class, but you can write your own or use gensim's other
    # serializable classes like the MmCorpus or BleiCorpus. You can implement
    # your own as well, but note that it must support *three* ways of calling
    # __getitem__: with an int, with a slice and with an arbitrary list.
    #
    # The "file to which to serialize" was actually a small lie. Each serializer
    # class implements its own mechanism of saving the data, but all of them
    # need some filename from which they derive other filenames they use.
    #
    # TL;DR: Use the Serializer to save the data at some stage during
    # processing, it will speed up access to them later on.

    return pipeline
    # At this point, our text data is safely transformed and ready for (fast)
    # processing. We're done here!


def build_image_pipeline():
    """Just like build_text_pipeline(), this function builds an image pipeline.
    The individual steps are different, but they follow the exact same
    principles."""
    icorp = ImagenetCorpus(image_vectors,
                           **icorp_settings)
    pipeline = icorp
    # Starting out with an ImagenetCorpus, which is written to read from
    # a specific csv file.

    tanh = GeneralFunctionTransform(numpy.tanh,
                                    multiplicative_coef=0.3)
    pipeline = tanh[pipeline]
    # We use a different multiplicative coefficient here, because the image
    # data falls into a different range.

    serializer = Serializer(pipeline, ShardedCorpus,
                            fname=serialization_ifile)
    pipeline = serializer[pipeline]
    # Mustn't forget to serialize.

    return pipeline


def build_multimodal_pipeline(text_pipeline, image_pipeline):
    """This function combines the text and image pipelines into a multimodal
    dataset above which a joint model can be trained."""
    text_dataset = Dataset(text_pipeline)
    image_dataset = Dataset(image_pipeline)
    # The Dataset class is a wrapper around a pipeline that adds some
    # functionality. Datasets provide two things over pipelines:
    #
    # * they guarantee a dimension,
    # * they provide a batch-retrieval interface using batch index and batch
    #   size.
    #
    # The first functionality, guaranteeing a dimension, is the more important
    # one.
    #
    # (They also support a rudimentary train-dev-test split, by
    # setting aside a certain proportion of the data for testing and development
    # sets, but we do not use that here; there are better ways of doing that.)

    multimodal_dataset = CompositeDataset((text_dataset, image_dataset),
                                          names=('text', 'img'),
                                          aligned=False)
    # CompositeDataset is a class that allows combining multiple datasets into
    # one. Here, we combine our text and image data into a multimodal dataset.
    # However, our data is not in text-image pairs yet, so we cannot assume
    # that the individual datasets comprising the CompositeDataset are aligned.
    # (If we forgot to set ``aligned=False``, the initialization would fail as
    # the text and image data lengths are different; however, this cannot be
    # relied on!)
    # We add the names mainly for clarity (although CompositeDataset allows
    # retrieving data from individual sub-sets by name as well).

    # Now for a slighlty complicated part: we need to tell the composite dataset
    # which text fits to which image. There is no one-line shortcut for this
    # action yet.
    t2i_file = os.path.join(data_root,
                            layout.textdoc2imdoc)
    # The t2i_file variable contains the path to a file that records the text
    # document-image document pairs. However, we need to translate this mapping
    # into a set of index pairs, where the first index will lead to the text
    # data and the second index will lead to the image data. The
    # CompositeDataset can then be flattened using these indices to say which
    # text goes with which image.

    t2i_map = parse_textdoc2imdoc_map(t2i_file)
    # This function collects for each text file the set of images associated
    # with it.

    t2i_list = [[text, image]
                for text in t2i_map
                for image in t2i_map[text]]
    # This operation simply converts the per-text dictionary into a list of
    # (text document, image document) names.

    t2i_indexes = docnames2indexes(multimodal_dataset, t2i_list)
    # However, we still need to turn document names into document indices.
    # Luckily, both VTextCorpus and ImagenetCorpus -- the bottom corpora of the
    # text and image pipelines -- have recorded the document names associated
    # with each data point in the respective datasets. The ``docunames2indexes``
    # function utilizes this feature. (Of course, the names in
    # the t2i_file must correspond to the names in the vtlist and image_vectors
    # files used to read the original data.)

    flatten = FlattenComposite(multimodal_dataset,
                               t2i_indexes)
    flat_multimodal_dataset = flatten[multimodal_dataset]
    # FlattenComposite is just another transformation block, although
    # specifically designed to deal with composite datasets (it will refuse to
    # work on anything else). Its role is to stitch items from individual
    # subsets of the composite dataset together. (It does this on the fly.)

    serializer = Serializer(flat_multimodal_dataset,
                            ShardedCorpus,
                            serialization_mmname)
    pipeline = serializer[flat_multimodal_dataset]
    # Finally, we serialize the flattened results.

    return pipeline

###############################################################################

if __name__ == '__main__':

    text_pipeline = build_text_pipeline()
    image_pipeline = build_image_pipeline()

    multimodal_pipeline = build_multimodal_pipeline(text_pipeline,
                                                    image_pipeline)

    print 'Now we can train something! ' \
          'But that will be explained in another tutorial.'

    mmdata = Dataset(multimodal_pipeline, devel_p=0.1, test_p=0.1)
    model_handle = DenoisingAutoencoder.setup(mmdata,
                                              n_out=100,
                                              reconstruction='cross-entropy')
    # The setup() method provides a model handle, with a .train(), .validate(),
    # .test() and .run() method.

    learner = BaseSGDLearner(n_epochs=3, batch_size=1, validation_frequency=4)
    # The learner will run the training iterations. Not yet, though.

    sftrans = SafireTransformer(model_handle,
                                mmdata,
                                learner)
    output = sftrans[mmdata]
    # And -- the SafireTransformer is again just one more pipeline block!
    # There are now the 100-dimensional representations of the joint text-image
    # model in output. Woot!
    # The training process was all run during SafireTransformer initialization.

