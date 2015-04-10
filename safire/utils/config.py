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

    [_info]
    name=toy_example

    [_builder]
    save=False
    savename=_info.name

    [foo]
    _class=safire.Foo
    _label=foo_1234
    some_property=True
    link=bar

    [bar]
    _class=safire.Bar
    property_through_eval='+'.join(_info.name, 'bar_4321')

(Do not worry about ``[_info]`` and ``[_builder]`` yet.) Building this
configuration would execute the following Python code::

    >>> name = 'toy_example'
    >>> builder = safire.utils.ConfigBuilder(save=True)
    >>> bar = safire.Bar(property_through_eval='+'.join(info.name, 'bar_4321'))
    >>> foo = safire.Foo(label='foo_1234', some_property=True, link=bar)

Each section of a configuration file corresponds to an object. In fact, you can
think of the configuration file as pseudocode for initialization. The ``_class``
special value defines the class of the objject, the section name is the name
of the initialized instance of that class and all values that do not start
with an underscore are used as initialization arguments.

Generally, **everything that doesn't start with an underscore translates
literally to Python code**.

This also
implies that you need to input objects **in the order in which they are needed**
as the configuration parser and builder don't have the capability to determine
the dependencies between the objects automatically.

There is a set of defined special sections that the ConfigBuilder interprets
differently. These all have names that start with an underscore.

Assembling pipelines
--------------------

However, while the ``bar`` object is used in initializing ``foo``, this is not
a pipeline, these are only the building blocks. To assemble a pipeline, we need
to apply the blocks. To this end, there's the special ``[_assembly]`` section::

    [_assembly]
    _#1=`bar`
    _#2=`foo[#1]`
    _#3=`foo[#2]`

which will result in the following action::

    >>> pipeline = bar
    >>> pipeline = foo[pipeline]
    >>> pipeline = foo[pipeline]

If we wanted to save the resultant pipeline, we'd need to set
``_builder.save=True``. Then, the following code would execute::

    >>> pipeline.save(builder.savename)

For integrating with Safire's default experiment layouts, you should define
a ``loader`` section::

    [_loader]
    _class=safire.data.loaders.MultimodalShardedDatasetLoader
    root=PATH/TO/YOUR/ROOT
    name=YOUR_NAME

Then, it's possible to define a systematic savename for example as::

    [_builder]
    savename=`loader.get_pipeline_name(info.name)`


Referring to pipeline components in config
------------------------------------------

A tricky situation is when the *initialization* of a transformer requires
the *pipeline* up-to-date. When you need to do that, use the identifiers of
pipeline stages defined in the ``[_assembly]`` section. (See the ``[tfidf]``
section of the full example in the next section.) The builder collects
references to the assembled pipeline from the transformer sections and
interleaves transformer initialization with pipeline application.

This is *always* a concern for Serializers and very often for other transformers
 -- more or less anything that is trained.


A full example
--------------

::

    [_info]
    name=full_example

    [_loader]
    _class=safire.data.loaders.MultimodalShardedDatasetLoader
    root=/var/data/safire
    name=safire-notabloid

    [_builder]
    save=True
    save_every=False
    savename=`loader.get_pipeline_name(info.name)`
    log_blocks=True
    log_saving=True

    [_assembly]
    _#1=`vtcorp`
    _#2=`tfidf[#1]`
    _#3=`serializer[#2]`

    [_persistence]
    _loader=`loader`
    _#1=vtcorp
    _#2=tfidf
    _#3=tfidf.serialized

    [tokenfilter]
    _class=safire.data.filters.positionaltagfilter.PositionalTagTokenFilter
    values=`['N', 'A', 'V']`
    t_position=0

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

    [tfidf]
    _class=gensim.models.TfidfModel
    _label=tfidf
    corpus=`#1`
    normalize=True

    [serializer]
    _class=safire.data.serializer.Serializer
    corpus=`#2`
    serializer_class=`safire.data.sharded_corpus.ShardedCorpus`
    fname=`loader.generate_serialization_target(''.join(vtcorp.label, tfidf.label))`
    overwrite=False


This config would produce the following code::

    loader = MultimodalShardedDatasetLoader(root='/var/data/safire', name='safire-notabloid')
    tokenfilter = PositionalTagTokenFilter(values=['N', 'A', 'V'], t_position=0)
    vtcorp = VTextCorpus(...)
    pipeline = vtcorp
    tfidf = TfidfModel(...)
    pipeline = tfidf[pipeline]
    serializer = Serializer(...)
    pipeline = serializer[pipeline]
    savename = loader.get_pipeline_name(info.name)
    pipeline.save(savename)


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

Saving pipelines from a configuration
-------------------------------------

So far, we have talked about saving and serializing pipelines in code.
Now we will describe how these mechanisms are leveraged in configuring
experiments.

Saving the finished pipeline is handled through ``[_builder]`` arguments
``save`` and ``savename``: unless ``save`` is set to ``False``, the finihsed
pipeline will be saved under the ``savename`` name. (When using Safire data
layouts, it's recommended to define ``savename`` using a Loader's
``generate_pipeline_name()`` method.)

For more fine-grained control over which stages of the pipeline should be saved,
you can define a special ``[_persistence]`` section that defines the savename
for each stage at which you wish to save the pipeline::

    [_persistence]
    _#1=path/to/data/full_example.vtcorp.pln
    _#2=path/to/data/full_example.tfidf.pln
    _#3=path/to/data/full_example.serialized_tfidf.pln

If a ``_loader`` value is defined in ``_persistence``, the values are
interpreted as labels for the loader's ``get_pipeline_name()`` method::

    [_persistence]
    _loader=`loader`
    _#1=vtcorp
    _#2=tfidf
    _#3=serialized_tfidf

Note that saving ``#3`` is redundant when the ``builder.save`` flag is set to
True; if the supplied names do not match, the pipeline is simply saved twice.

Using saved pipelines in a configuration
----------------------------------------

When a configuration is built, the builder first checks whether a pipeline can
be loaded under the ``savename`` attribute. If yes, it simply loads the
pipeline. This behavior can be disabled using the ``no_default_load=True``
builder setting.

If the ``[_persistence]`` section is defined (and loading the entire pipeline
failed or was disabled), the builder then attempts to load individual stages
based on the saving names, going backwards from the later blocks to the earlier
ones. Again, this can be disabled by setting ``no_persistence_load=True`` in
the builder section. Fine-grained control over which pipeline stages to *not*
load, for example when experimenting with different settings for a block,
can be achieved using for example ``first_to_load=#2``, which will not attempt
to load blocks #3 and onward.

When a block is loaded, the config builder will assume that all blocks
accessible from the loaded block have been loaded. This implies that if you
wish to re-run an experiment with different settings, you'll have to disable


"""
import copy
import logging
import collections
import itertools

__author__ = "Jan Hajic jr."

class ConfigParser(object):
    """The ConfigParser reads the configuration file and provides an interface
    to the config values that the ConfigBuilder will use. It does *not*
    interpret the configuration file (that's the Builder's job).
    """
    pass


class ConfigBuilder(object):
    """The ConfigBuilder takes a parsed config file and generates the actual
    Python code requested in the config.
    """
    def __init__(self):
        raise NotImplementedError()

    def build_block_dependency_graph(self, assembly):
        """Builds a dependency graph of the pipeline blocks (the topology of the
        pipeline)."""
        # The graph is in a list-of-neighbors representation. If block X depends
        # on block Y, then Y will be a member of graph[X].
        graph = collections.defaultdict(set)

        # Initialize vortices
        for block in assembly:
            graph[block] = {}

        # Initialize edges. "Parent" is the tail end of the arrow, "child"
        # is the head end. Parent depends on child.
        # Ex: _#2=foo[_#1]   ... #2 is parent, #1 is child (#2 depends on #1).
        for parent, child in itertools.product(assembly.keys(),
                                               assembly.keys()):
            if child in assembly[parent]:
                graph[parent].add(child)

        return graph

    def add_transformer_dependencies(self, block_graph, transformers, assembly):
        """Adds to the block graph edges leading to transformers that depend on
        pipeline blocks being already built.

        :param block_graph: The output of the :meth:`build_block_dependency_graph`
            method: a graph of dependencies in list-of-neighbors representation.
            Expects a dict of iterables.

        :param transformers: A dictionary of dictionaries of string values.
            Members of the dictionary are individual transformers used in
            constructing the pipeline. Used here to determine dependencies of
            transformers on blocks.

        :param assembly: A dictionary of string values describing the pipeline
            topology. Used here to determine dependencies of blocks on
            transformers.

        :return: The enriched graph.
        """
        enriched_graph = copy.deepcopy(block_graph)
        for t in transformers:
            enriched_graph[t] = {}
            for key, value in t.items():
                for b in block_graph:
                    if b in value:
                        enriched_graph[t].add(b)

        for t in transformers:
            for a in assembly:  # All these should already have been added.
                if t in assembly[a]:
                    # Initializing block A depends on having transformer T
                    enriched_graph[a].add(t)
        return enriched_graph

    @staticmethod
    def sort_dependency_graph(graph):
        """Reorders the vertices (blocks) in the block dependency graph in their
        topological order.  Raises a ValueError if the graph is not acyclic.

        >>> g = { '_#1': [], '_#2': [], '_#3': ['_#1', '_#2'], '_#4': ['_#3']}
        >>> ConfigBuilder.sort_dependency_graph(g)
        [{'_#1': []}, {'_#2': []}, {'_#3': ['_#1', '_#2']}, {'_#4': ['_#3']}]

        """
        sorted_graph = []

        unsorted_graph = copy.deepcopy(graph)
        # While vertices are left in the unsorted graph:
        #  - iterate throuh vertices
        #  - find one that doesn't depend on anything in the unsorted graph
        #  - add vertex to sorted graph
        #  - remove vertex from unsorted graph
        while unsorted_graph:
            for vertex in unsorted_graph:
                can_remove = True
                for child in unsorted_graph[vertex]:
                    if child in unsorted_graph:
                        can_remove = False
                        break
                if can_remove:
                    sorted_graph.append({vertex: graph[vertex]})
                    del unsorted_graph[vertex]
                    break

        return sorted_graph

    def build(self, configuration):

        # Determine block dependencies from the _assembly section. These
        # dependencies represent the topology of the pipeline.
        block_graph = self.build_block_dependency_graph(configuration._assembly)

        # Add transformer dependencies, to set block/transformer interleaving
        combined_graph = self.add_transformer_dependencies(block_graph,
                                                        configuration._assembly,
                                                        configuration.objects)

        # Sort dependency graph
        sorted_graph = self.sort_dependency_graph(combined_graph)

        # At this point, we have the topology of initialization. The only thing
        # NOT covered by the dependency graph are transformer-transformer
        # dependencies. These cannot be determined automatically, as there is
        # no convention for transformer names similar to block names in the
        # _assembly section by which to differentiate references to transformers
        # in other transformers' init code.
        # Therefore, the order of initialization of individual transformers has
        # to be defined by the order in which they are given in the
        # configuration file. This is why an OrderedDict must be used when
        # parsing them.
        # To fully determine the order of initialization, we need to make sure
        # that this ordering is respected in the dependency graph. To this end,
        # we will check the topological sort output for inconsistencies with
        # the transformer ordering from the config file.

        # Initialize loader.
        loader = self.execute_init(configuration._loader)
        if not hasattr(loader, 'generate_pipeline_name'):
            raise TypeError('Loader of type {0} does not support generating'
                            ' pipeline names.'.format(type(loader)))

        # Decide which objects have to be initialized and which should be
        # loaded.
        # The loading order should be determined from the assembly graph!
        # Should be a BFS through the dependency DAG.

        # Generate persistence codes: loading
        #   - decision: generate if-statements for persistence, or look ahead
        #     into the configuration, test availability and only generate the
        #     code that will actually get executed?
        #   --> second option looks better - determining what to load and what
        #       to initialize sounds like something the builder should figure
        #       out. Also, the builder should *not* be a templating machine.
        #   --> However, this means pre-loading the Loader: make
        #       it a special section. (Paths are a part of what the builder
        #       needs to work with.)
        #   --> Problem: the loader needs to be available for initialization,
        #       too. Initialize the loader separately, using the _class
        #       mechanism, and then pull it in?
        # Should the loader be compulsory??
        #

        # Assembly names should be generated by the parser.
        # Generate object init codes (uses assembly names already)
        obj_init_codes = [self.obj_init_code(obj)
                          for obj in configuration.objects]

        # Resolve dependencies

        # Generate block application codes



class ConfigRunner(object):
    """The ConfigRunner gets the code generated by the ConfigBuilder and
    executes it."""
    pass