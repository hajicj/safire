"""
This module contains classes that allow storing and loading Safire pipelines
from configuration files.

Configuring experiments
=======================

While the basic mechanism for persistence is simply saving and loading the
pipelines themselves, it should be possible to export (and import) experimental
settings without having to write the Python code. Think of the configuration
as pseudocode, which an interpreter will "translate" into Python code.

The difficult part of thus interpreting an experiment configuration is the fact
that parts of the pipeline depend on each other. Not only are pipeline blocks
connected to their previous block, some transformers are initialized using the
output of the pipeline on which they are applied. In other words, there are
dependencies to resolve when building a pipeline from a configuration.

We can actually think about the configuration as a Makefile. Individual pipeline
components are targets that may depend on each other. Circular dependencies are
forbidden. This doesn't mean that we have to use a Makefile specifically for
storing experiment configurations, only that the principle is similar. The
Pylearn2 library uses YAML to store configurations, including objects that
point to one another -- we can use that and implement some mechanism for
dependency resolution (it's just a topological sort of a DAG, anyway).

We have some options of resolving these dependencies:

* Requiring that the symbols are declared in the order in which they are needed
  (Who builds circular pipelines, anyway?)
* Creating the dependency graph and only then initializing the pipeline blocks,
  in the correct order.


How to write a configuration
============================

Safire pipelines can be configured in plain old *.ini format, with a few
safire-specific extras. We'll illustrate on a toy example::

    [info]
    name=toy_example

    [builder]
    save=False

    [foo]
    _class=safire.Foo
    _label=foo_1234
    some_property=True
    link=`bar`

    [bar]
    _class=safire.Bar
    property_through_eval=`'+'.join(info.name, 'bar_4321')`

Building this configuration would execute the following Python code::

    >>> name = 'toy_example'
    >>> builder = safire.utils.ConfigBuilder(save=True)
    >>> bar = safire.Bar(property_through_eval='+'.join(info.name, 'bar_4321'))
    >>> foo = safire.Foo(label='foo_1234', some_property=True, link=bar)

However, this only builds some of the

Note how the statements in backticks are interpreted as literal code. You can
reference other objects in the config file, but you have to put the reference
in backticks, otherwise it's treated as a regular string. Notice also how
``bar`` was initialized before ``foo`` -- the builder can see the dependency.

A full example::

    [info]
    name=full_example

    [builder]
    save=True
    save_every=False
    log_blocks=True
    log_saving=True

    [loader]
    root=/var/data/safire
    name=safire-notabloid

    [vtcorp]
    _class=safire.data.vtextcorpus.VTextCorpus
    _label=NAV.pf0.3.pff.fc
    input_root=`loader.root`
    vtlist=`os.path.join(loader.root, loader.layout.vtlist)`
    token_filter=`tokenfilter`
    pfilter=0.2
    pfilter_full_freqs=True
    filter_capital=True
    precompute_vtlist=True

    [tokenfilter]
    _class=safire.data.filters.positionaltagfilter.PositionalTagTokenFilter
    values=`['N', 'A', 'V']`
    t_position=0

    [tfidf]
    _class=gensim.models.TfidfModel
    _label=tfidf
    corpus=`vtcorp`
    normalize=True

    [serializer]
    _class=safire.data.serializer.Serializer
    corpus=`tfidf`
    serializer_class=`safire.data.sharded_corpus.ShardedCorpus`
    fname=`loader.generate_serialization_target(''.join(vtcorp.label, tfidf.label))`
    overwrite=False




The configuration builder
=========================

The workhorse of converting a config file into a pipeline is the
:class:`ConfigBuilder`. The builder receives the output of configuration file
parsing and outputs the pipeline object. This may take long: for example, the
entire training of a neural network can be a part of pipeline initialization!

The builder itself is bootstrapped from the ``builder`` section of the config
file. If this section is not given, the default builder settings will be used.

Building a pipeline comes in stages:

* Create a list of requested objects and their dependencies,
* Initialize the individual objects in the correct order.

Persistence
===========

A major feature of Safire should be that experiment results are persistent: you
can save both the state of pipeline and the pipeline's output data. In Safire,
due to the strict separation between data and pipeline components, these are
two different actions:

* Pipelines are **saved**,
* Data is **serialized**.

Saving pipelines is handled through the ``.save()`` method available to each
pipeline block. The pipeline is saved recursively. (Internally, the ``corpus``
attribute is saved like any other, so the save action propagates down the
pipeline, and the transformers associated with each pipeline stage are saved as
well through the ``obj`` attribute.) The only thing tricky about saving
pipelines is that you have to be careful about your own pipeline blocks: they
all have to be ``cPickle``-able.

Serializing pipelines is a little trickier. Instead of calling a function over
the pipeline, serialization - and efficient reading of serialized data - is
handled using a special :class:`Serializer` transformer. Applying the Serializer
to a pipeline creates a special :class:`SwapoutCorpus` block that reads data
from the Serializer instead of the previous pipeline block.

Saving intermediate pipelines
-----------------------------

The above implies that the correct order for saving and re-using intermediate
pipeline results is:

#. Serialize
#. Save

The pipeline will be saved with the SwapoutCorpus/Serializer block. Also, you
won't have to worry about loading the serialized data -- it will come
automatically with the pipeline, as everything that is needed to access it was
saved with the SwapoutCorpus/Serializer block.

Actually, it is strongly encouraged to serialize whenever you wish to save.
There are various pipeline blocks that initialize their state through the
__iter__ method and will be incomplete unless the iteration is run in its
entirety. At the same time, serializing anything worth saving will save time
later on, when you need to re-use the data -- be it for further processing,
analysis, visualization -- and because the Serializer iterates through all the
data in the pipeline, the pipeline is guaranteed to have all its internal state
initialized completely.

Using saved pipelines in a configuration
----------------------------------------

So far, we have talked about saving and serializing pipelines by themselves.
Now we will describe how these mechanisms are leveraged in configuring
experiments.

When a pipeline is built according to a configuration, the builder runs
backwards through the blocks in the tentative build graph and checks each block
for a `save_label` configuration value. If that value is set, it will try to
load the given block. If successful, it will assume that all blocks accessible
from the loaded block have been loaded. If unsuccessful, it will handle the
failure: either it will go on initializing (``load_fail_soft``), or raise
an exception (``load_fail_hard``, default setting).


"""
import logging

__author__ = "Jan Hajic jr."

