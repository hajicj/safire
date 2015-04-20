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

Safire pipelines can be configured in plain old *.ini format, with some
safire-specific extras. We'll illustrate on a toy example::

    [_info]
    name='toy_example'

    [_builder]
    save=False
    savename=_info.name

    [foo]
    _class=safire.Foo
    _label='foo_1234'
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

Note that the values are used *literally* as Python code. This means that all
strings must be quoted.

Generally, **everything that doesn't start with an underscore translates
literally to Python code**.

Dependencies
------------

The configuration parser and builder don't have the capability to determine
the dependencies between the objects automatically. An alternative is specifying
the objects on which a transformer depends in a ``_dependencies`` value::

    [baz]
    _class=safire.Baz
    _dependencies=foo,bar
    inputs_1=foo
    inputs_2=bar

In the future, we will add a Python parser to resolve the dependencies from the
object init args themselves.

There is also a set of defined special sections that the ConfigBuilder
interprets differently. These all have names that start with an underscore.
Objects corresponding to these sections (most often the ``_loader`` object)
are always initialized before the non-special sections, so you do *not* have
to explicitly mark dependencies on them.

Assembling pipelines
--------------------

However, while the ``bar`` object is used in initializing ``foo``, this is not
a pipeline, these are only the building blocks. To assemble a pipeline, we need
to apply the blocks. To this end, there's the special ``[_assembly]`` section::

    [_assembly]
    _1_=bar
    _2_=foo[_1_]
    _3_=foo[_2_]

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
    root='PATH/TO/YOUR/ROOT'
    name='YOUR_NAME'

Then, it's possible to define a systematic savename for example as::

    [_builder]
    savename=loader.get_pipeline_name(info.name)


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
    root='/var/data/safire'
    name='safire-notabloid'

    [_builder]
    save=True
    save_every=False
    savename=_loader.get_pipeline_name(info.name)
    log_blocks=True
    log_saving=True

    [_assembly]
    _1_=vtcorp
    _2_=tfidf[_1_]
    _3_=serializer[_2_]

    [_persistence]
    _loader=loader
    _1_='vtcorp'
    _2_='tfidf'
    _3_='tfidf.serialized'

    [tokenfilter]
    _class=safire.data.filters.positionaltagfilter.PositionalTagTokenFilter
    values=['N', 'A', 'V']
    t_position=0

    [vtcorp]
    _class=safire.data.vtextcorpus.VTextCorpus
    _label='NAV.pf0.3.pff.fc'
    _dependencies=_loader,tokenfilter
    input_root=_loader.root
    vtlist=os.path.join(_loader.root, _loader.layout.vtlist)
    token_filter=tokenfilter
    pfilter=0.2
    pfilter_full_freqs=True
    filter_capital=True
    precompute_vtlist=True

    [tfidf]
    _class=gensim.models.TfidfModel
    _label='tfidf'
    corpus=_1_
    normalize=True

    [serializer]
    _class=safire.data.serializer.Serializer
    _dependencies=_loader,vtcorp,tfidf
    corpus=_2_
    serializer_class=safire.data.sharded_corpus.ShardedCorpus
    fname=_loader.generate_serialization_target(''.join(vtcorp.label, tfidf.label))
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
    _1_=path/to/data/full_example.vtcorp.pln
    _2_=path/to/data/full_example.tfidf.pln
    _3_=path/to/data/full_example.serialized_tfidf.pln

If a ``_loader`` value is defined in ``_persistence``, the values are
interpreted as labels for the loader's ``get_pipeline_name()`` method::

    [_persistence]
    loader=_loader
    _1_=vtcorp
    _2_=tfidf
    _3_=serialized_tfidf

Note that saving ``_3_`` is redundant when the ``builder.save`` flag is set to
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


"""
import copy
import inspect
import logging
import collections
import itertools
import imp
import gensim
import os
import sys
import pprint

__author__ = "Jan Hajic jr."

# Some utility functions.


def names_in_code(code_string, eval=True):
    """Returns a list of all names used in the given code.

    If a list of code strings is given, will output all names in each of the
    strings.

    >>> names_in_code('a+b+c')
    set(['a', 'c', 'b'])
    >>> names_in_code('a=5;b=4', eval=False)
    set(['a', 'b'])
    >>> names_in_code(['safire.utils.transcorp', 'safire.data.vtextcorpus.VTextCorpus'])
    set(['transcorp', 'VTextCorpus', 'utils', 'vtextcorpus', 'safire', 'data'])

    :param code_string: A string of Python code. Has to be eval()-able
        or exec()-able, depending on the ``eval`` flag.

    :param eval: If True, will compile the ``code_string`` using ``eval`` mode.
        If False, will compile using ``exec`` mode.

    :return: The set of names in the code string.
    """
    if isinstance(code_string, list):
        return set(itertools.chain(*[names_in_code(s, eval=eval) for s in code_string]))
    if isinstance(code_string, dict):
        return set(itertools.chain(*[names_in_code(s, eval=eval)
                                     for s in code_string.values()]))

    mode = 'eval'
    if eval is False:
         mode = 'exec'
    code = compile(code_string, '<string>', mode)
    members = dict(inspect.getmembers(code))
    names = members['co_names']
    return set(names)


# Configuration classes.

class Configuration(object):
    """Plain Old Data class; the data structure that represents a configuration.
    """
    def __init__(self):
        # Each object or special object is a dict with string keys and string
        # values. The values will be interpreted as Python code.
        self.objects = collections.OrderedDict()
        self.special_objects = collections.OrderedDict()

        self.expected_special_objects = ['_info',
                                         '_builder',
                                         '_loader',
                                         '_assembly',
                                         '_persistence']
        self.required_special_objects = ['_info', '_builder', '_assembly']

        for so in self.expected_special_objects:
            setattr(self, so, None)

    def add_object(self, name, obj):
        """Adds the given object to itself under the given name. If the name
        starts with an underscore, considers the object special.
        """
        if name.startswith('_'):
            self.special_objects[name] = obj
            if name in self.expected_special_objects:
                setattr(self, name, obj)
        else:
            self.objects[name] = obj

    def is_valid(self):
        """Checks integrity of the given configuration."""
        valid = True
        missing_required_sections = []
        for so in self.required_special_objects:
            if getattr(self, so) is None:
                valid = False
                missing_required_sections.append(so)
        return valid, missing_required_sections

    def get_available_names(self):
        available = set(itertools.chain(self.objects,
                                        self._assembly.keys()))
        if 'loader' in self._assembly and 'loader' not in self.objects:
            available.remove('loader')
        return available


class ConfigParser(object):
    """The ConfigParser reads the configuration file and provides an interface
    to the config values that the ConfigBuilder will use. It does *not*
    interpret the configuration file (that's the Builder's job).
    """

    def parse(self, file):
        """Given a config file (handle), will build the Configuration object."""
        conf = Configuration()

        current_section = {}
        current_section_name = ''
        in_section = False
        for line_no, line in enumerate(file):
            line = line.strip()
            # Comment
            if line.startswith('#'):
                continue

            # Empty line
            if line == '':
                # ends section.
                if in_section:
                    conf.add_object(current_section_name, current_section)
                    # Clear current section
                    current_section = {}
                    current_section_name = ''
                    in_section = False
                else:
                    pass

            # Section start
            elif line.startswith('['):
                if not line.endswith(']'):
                    raise ValueError('On line {0} of configuration: missing'
                                     ' closing bracket for section name.\n  '
                                     'Line: {1}'.format(line_no, line))
                if in_section:
                    raise ValueError('On line {0} of configuration: starting'
                                     ' a section within another section.\n  '
                                     'Line: {1},\n  in section {2}'
                                     ''.format(line_no, line, current_section_name))
                in_section = True

                section_name = line[1:-1]
                if section_name in conf.objects:
                    raise ValueError('On line {0} of configuration: section '
                                     '{1} has already been defined.'
                                     ''.format(line_no, section_name))
                if section_name in conf.special_objects:
                    raise ValueError('On line {0} of configuration: special '
                                     'section {1} has already been defined.'
                                     ''.format(line_no, section_name))
                current_section_name = section_name

            # Not a comment, not an empty line, not a section heading
            else:
                # key-value pair for current section
                if in_section:
                    if '=' not in line:
                        raise ValueError('On line {0} of configuration: in '
                                         'section, but not in key=value format'
                                         ' - missing \'=\' character.\n  '
                                         'Line: {1}'.format(line_no, line))
                    key, value = line.split('=', 1)

                    # Special keys handling
                    if key in {'_dependencies', '_import'}:
                        value = value.split(',')
                    if key == '_access_deps':
                        values = {}
                        for dep in value.split('|'):
                            name, code = dep.split(':', 1)
                        values[name] = code
                        value = values

                    current_section[key] = value
                # junk
                else:
                    raise ValueError('On line {0} of configuration: not a '
                                     'comment, empty line or heading, but not '
                                     'within a section.\n  Line: {1}'
                                     ''.format(line_no, line))

        # Deal with last section
        if in_section:
            conf.add_object(current_section_name, current_section)

        is_valid, missing_required_sections = conf.is_valid()
        if not is_valid:
            raise ValueError('Configuration loaded but is not valid. Missing'
                             ' required sections: {0}'
                             ''.format(', '.join(missing_required_sections)))

        return conf


class ConfigBuilder(object):
    """The ConfigBuilder takes a parsed config file and generates the actual
    Python code requested in the config.
    """
    def __init__(self, conf):
        self.configuration = conf

        self._info = self.dict2module(conf._info, '_info')
        self._loader = self._execute_init(conf._loader, _info=self._info)

        # Bootstrap its own settings
        bootstrap_module = self.eval_dict(conf._builder, no_eval='special',
                                          _info=self._info,
                                          _loader=self._loader)
        if '_import' not in bootstrap_module:
            bootstrap_module['_import'] = []
        for k, v in bootstrap_module.items():
            setattr(self, k, v)

        self.imports = dict([self._execute_import(import_name)
                             for import_name in self._import])

        self._assembly = self.dict2module(conf._assembly,
                                          '_assembly',
                                          no_eval='all',
                                          _info=self._info,
                                          _loader=self._loader,
                                          _builder=self)
        self._persistence = self.dict2module(conf._persistence,
                                             '_persistence',
                                             no_eval='special',
                                             _info=self._info,
                                             _loader=self._loader,
                                             _builder=self,
                                             _assembly=self._assembly)

        # IMPORTANT: a dictionary of all available objects. When a dependency
        # is requested, it is taken from here.
        self.objects = {'_info': self._info,
                        '_loader': self._loader,
                        '_builder': self,
                        '_assembly': self._assembly,
                        '_persistence': self._persistence}
        self.exclude_from_build_output = {'_info', '_loader', '_builder',
                                          '_assembly', '_persistence'}

        # IMPORTANT: a graph of dependencies between configuration object.
        # Works with object names, not the objects themselves.
        self.deps_graph = self.build_dependency_graph(conf)
        self.sorted_deps = self.sort_dependency_graph(self.deps_graph)

    def build_block_dependency_graph(self, assembly):
        """Builds a dependency graph of the pipeline blocks (the topology of the
        pipeline)."""
        # The graph is in a list-of-neighbors representation. If block X depends
        # on block Y, then Y will be a member of graph[X].
        graph = collections.defaultdict(set)

        # Initialize vortices
        for block in assembly:
            graph[block] = set()

        # Initialize edges. "Parent" is the tail end of the arrow, "child"
        # is the head end. Parent depends on child.
        # Ex: _2_=foo[_1_]   ... #2 is parent, #1 is child (#2 depends on #1).
        for parent, child in itertools.product(assembly.keys(),
                                               assembly.keys()):
            if child in names_in_code(assembly[parent]):
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
            enriched_graph[t] = set()

        # Add dependencies of transformers on blocks
        for t in transformers:
            for key, value in transformers[t].items():
                #print 'Transformer {0}: key {1}, value {2}'.format(t, key, value)
                names = names_in_code(value)
                for b in block_graph:
                    if b in names:
                        enriched_graph[t].add(b)

        # Add dependencies of blocks on transformers
        for t in transformers:
            for a in assembly:  # All these should already have been added.
                if t in names_in_code(assembly[a]):
                    # Initializing block A depends on having transformer T
                    enriched_graph[a].add(t)

        # Add transformer-transformer dependencies
        for t in transformers:
            if '_dependencies' in transformers[t]:
                dependencies = transformers[t]['_dependencies']
                for d in dependencies:
                    enriched_graph[t].add(d)

        return enriched_graph

    def build_dependency_graph(self, configuration):
        block_graph = self.build_block_dependency_graph(configuration._assembly)
        block_and_obj_graph = \
            self.add_transformer_dependencies(block_graph,
                                              configuration.objects,
                                              configuration._assembly)
        return block_and_obj_graph

    def autodetect_dependencies(self, conf_object):
        """Attempts to detect all object dependencies that the given conf_object
        will require. Returns a list of the detected object names.
        Does not scan special attributes."""
        available = self.configuration.get_available_names()

        deps = set()
        for k, v in conf_object.items():
            if k.startswith('_'):
                continue
            code = compile(v, '<string>', 'eval')
            members = dict(inspect.getmembers(code))
            names = members['co_names']
            for name in names:
                if name in available:
                    deps.add(name)
        return deps

    @staticmethod
    def sort_dependency_graph(graph):
        """Reorders the vertices (blocks) in the block dependency graph in their
        topological order. Returns an OrderedDict.

        Raises a ValueError if the graph is not acyclic.

        >>> g = { 'C': ['A', 'B'], 'D': ['C'], 'A': [], 'B': []}
        >>> sorted_g = ConfigBuilder.sort_dependency_graph(g)
        >>> [name for name in sorted_g]
        ['A', 'B', 'C', 'D']
        """
        sorted_graph_list = []
        # print 'Sorting dependency graph: {0}'.format(pprint.pformat(dict(graph)))
        unsorted_graph = copy.deepcopy(graph)
        # While vertices are left in the unsorted graph:
        #  - iterate throuh vertices
        #  - find one that doesn't depend on anything in the unsorted graph
        #  - add vertex to sorted graph
        #  - remove vertex from unsorted graph
        while unsorted_graph:
            found_vertex_to_remove = False
            for vertex in unsorted_graph:
                can_remove = True
                for child in unsorted_graph[vertex]:
                    if child in unsorted_graph:
                        can_remove = False
                        break
                if can_remove:
                    found_vertex_to_remove = True
                    sorted_graph_list.append({vertex: graph[vertex]})
                    del unsorted_graph[vertex]
                    break
            if not found_vertex_to_remove:
                raise ValueError('Dependency graph is probably cyclic: {0}\n'
                                 'Sorted graph so far: {1}\n'
                                 'Unsorted: {2}'
                                 ''.format(pprint.pformat(dict(graph)),
                                           pprint.pformat(sorted_graph_list),
                                           pprint.pformat(dict(unsorted_graph))))

        sorted_graph = collections.OrderedDict()
        for item in sorted_graph_list:
            for key, value in item.items():
                sorted_graph[key] = value
        return sorted_graph

    def resolve_persistence(self):
        """Categorizes the objects according to whether they will be loaded
        directly, will be loaded as a dependency of something else, or will have
        to be initialized."""
        # Handle persistence:
        #  - determine which objects should be loaded and which should be
        #    initialized
        #     --> needs loader to check for availability
        able_to_load = []
        able_to_init = []
        for item in self.sorted_deps:
            if self.can_load(item):
                # Mark all objects that will be loaded.
                able_to_load.append(item)
            else:
                # Mark all that will be initialized.
                able_to_init.append(item)

        # Prune load/init sets according to dependency graph.
        will_load = set()
        will_be_loaded = set()
        to_resolve = set(self.sorted_deps.keys())
        while to_resolve:
            # No more candidates for loading
            if len(able_to_load) == 0:
                break
            current_obj = able_to_load.pop()
            # print 'Resolving load/will be loaded for object {0}'.format(current_obj)
            if current_obj not in to_resolve:
                # print '   ...has already been resolved, will not load.'
                continue
            else:
                # print '   ...will load.'
                will_load.add(current_obj)
                to_resolve.remove(current_obj)
            # Traverse all dependencies of current object and mark them as
            # "to be loaded"
            deps = copy.deepcopy(self.deps_graph[current_obj])
            while deps:
                current_dep = deps.pop()
                if current_dep not in to_resolve:
                    continue
                to_resolve.remove(current_dep)
                will_be_loaded.add(current_dep)
                # Add to dependency BFS traversal queue
                for d in self.deps_graph[current_dep]:
                    deps.add(d)
        # Whatever's left will be initialized, as it is unreachable from
        # the load-able objects.
        will_init = to_resolve

        logging.debug('Working with configuration: {0}'.format(self._info.name))
        logging.debug('  Will load: {0}'.format(will_load))
        logging.debug('  Will be loaded transitively: {0}'.format(will_be_loaded))
        logging.debug('  Will initialize: {0}'.format(will_init))

        return will_load, will_be_loaded, will_init

    def is_block(self, name):
        """Checks whether the given ``name`` belongs to a block or not."""
        if hasattr(self._assembly, name):
            return True
        else:
            return False

    def can_load(self, obj_name):
        """Determines whether the given item can be loaded.

        If the ``loader`` key is defined for ``self._persistence``, will
        interpret the values in _persistence as inputs for the loader's
        ``get_pipeline_name`` method.
        """
        if not hasattr(self._persistence, obj_name):
            return False
        else:
            if hasattr(self._persistence, 'loader'):
                fname = self._loader.pipeline_name(
                                getattr(self._persistence, obj_name))
            else:
                fname = getattr(self._persistence, obj_name)
            if os.path.isfile(fname):
                return True
            else:
                return False

    def get_loading_filename(self, persistent_label):
        """For the given persistence label, generates a filename under which the
        given persistent block should be available. The generated filename
        depends on the presence of the ``loader`` attribute in the _persistence
        section; if the loader is there, it will use the loader's
        ``pipeline_name`` method."""
        if hasattr(self._persistence, 'loader'):
            fname = self._persistence.loader.pipeline_name(persistent_label)
        else:
            fname = persistent_label
        return fname

    def get_dep_symbols(self, name):
        """Just looks into the dependency graph. All symbols used in the
        pipeline that is being built should be there."""
        return self.deps_graph[name]

    def is_dep_obj_accessible(self, name, symbol):
        """Checks that the actual object named ``symbol`` can be accessed as a
        dependency from the object called ``name``."""
        if self.is_block(name):
            symbol_conf = getattr(self._assembly, name)
            if symbol in symbol_conf:
                return True
        else:
            conf = self.configuration.objects[name]
            if '_access_deps' in conf:
                if symbol in conf['_access_deps']:
                    return True
            if self._uses_block(name, symbol):
                return True
        return False

    def _uses_block(self, name, blockname):
        """Checks whether the object of the given ``name`` depends on the block
        given with the given ``blockname``."""
        if not hasattr(self._assembly, blockname):
            # blockname does not refer to a block!
            return False
        if blockname in self.deps_graph[name]:
            return True
        else:
            return False

    def retrieve_dep_obj(self, name, obj, symbol):
        """Retrieve the actual dependency object signified by ``symbol`` from
        the ``obj`` with the ``name``."""
        if not self.is_dep_obj_accessible(name, symbol):
            return None
        obj_conf = self.configuration.objects[name]
        access_expr = obj_conf['_access_deps'][symbol]
        dep_obj = eval(access_expr, globals(), locals={name: obj})
        return dep_obj

    def build(self):
        """This method does the heavy lifting: determining dependencies, dealing
        with persistence, etc. It outputs a dictionary of all objects (blocks
        and non-blocks) that do not depend on anything.
        """
        will_load, will_be_loaded, will_init = self.resolve_persistence()

        logging.debug('Will load: {0}'.format(will_load))
        logging.debug('Will be loaded transitively: {0}'.format(will_be_loaded))
        logging.debug('Will initialize: {0}'.format(will_init))

        # Loading:
        #  - sort items to load according to the order in the sorted graph
        load_queue = []
        for item in self.sorted_deps:
            if item in will_load:
                load_queue.append(item)

        for item in load_queue:
            #  - load item (using loader, etc.)
            print 'Loading item: {0}'.format(item)
            persistent_label = getattr(self._persistence, item)
            print '   Persistent label: {0}'.format(persistent_label)
            fname = self.get_loading_filename(persistent_label)
            obj = self._execute_load(fname)
            #  - add item to object dict
            self.objects[item] = obj

            #  - traverse dependencies according to _access_deps. This time, it
            #    means iterating over the actual objects, as they have been
            #    loaded. Add each of these to the object dict.
            obj_queue = [(item, obj)]
            while obj_queue:
                current_symbol, current_obj = obj_queue.pop()

                # Add to objects dict.
                self.objects[current_symbol] = current_obj

                # Traverse dependency graph further
                current_dep_symbols = self.get_dep_symbols(current_symbol)
                current_dep_objs = {dep_symbol: self.retrieve_dep_obj(current_symbol,
                                                                  current_obj,
                                                                  dep_symbol)
                                    for dep_symbol in current_dep_symbols
                                    if self.is_dep_obj_accessible(current_symbol,
                                                                  dep_symbol)}
                obj_queue.extend(current_dep_objs.items())

        # Verify that everything that should have been loaded has been loaded
        for item in will_be_loaded:
            if item not in self.objects:
                logging.warn('Object {0} not found after it should have been'
                             ' loaded. (Available objects: {1})'
                             ''.format(item, self.objects.keys()))

        # Initialize objects that weren't loaded.
        # Note that the ordering needs to be re-computed.
        to_init = [item for item in self.sorted_deps if item in will_init]
        for item in to_init:
            if self.is_block(item):
                # print 'Initializing block: {0}'.format(item)
                self.init_block(item, **self.objects)
            else:
                # print 'Initializing object: {0}'.format(item)
                self.init_object(item,
                                 self.configuration.objects[item],
                                 **self.objects)
            if item in self.configuration._persistence:
                label = self.configuration._persistence[item]
                fname = self.get_loading_filename(label)
                print 'Saving object {0} to fname {1}'.format(item, fname)
                self.objects[item].save(fname)

        # At this point, all objects - blocks, transformers and others - should
        # be ready. We return a dictionary of objects on which nothing depends.
        reverse_deps = collections.defaultdict(set)
        for obj_name in self.objects:
            if obj_name not in reverse_deps:
                reverse_deps[obj_name] = set()
            deps = self.deps_graph[obj_name]
            for d in deps:
                reverse_deps[d].add(obj_name)
        output_objects = {obj_name: obj
                          for obj_name, obj in self.objects.items()
                          if len(reverse_deps[obj_name]) == 0
                            and obj_name not in self.exclude_from_build_output}
        return output_objects

    def run_saving(self):
        """Saves the built objects according to the persistence instruction."""
        for obj_name, obj_label in self.configuration._persistence.items():
            if obj_name == 'loader' and 'loader' not in self.objects:
                continue
            fname = self.get_loading_filename(obj_label)
            # print 'Saving object {0} to file {1}'.format(obj_name, fname)
            self.objects[obj_name].save(fname)

    @staticmethod
    def eval_dict(dictionary, no_eval=None, **kwargs):
        """Evaluates the dictionary values as Python code (using the ``eval()``
        built-in) with the kwargs as the local scope. Returns a new dictionary
        with the ``eval()``-uated values instead of the originals.

        :param no_eval: If set to None, will eval all dict members. If set to
            'special', will not eval special dict members (those that start
            with ``_``). If set to 'nonspecial', will only eval special dict
            members.
        """
        if no_eval is None:
            return {k: eval(v, globals(), kwargs) for k, v in dictionary.items()}
        elif no_eval == 'special':
            output = {}
            for k, v in dictionary.items():
                if not k.startswith('_'):
                    output[k] = eval(v, globals(), kwargs)
                else:
                    output[k] = v
            return output
        elif no_eval == 'nonspecial':
            output = {}
            for k, v in dictionary.items():
                if k.startswith('_'):
                    output[k] = eval(v, globals(), kwargs)
                else:
                    output[k] = v
            return output

    def init_object_or_block(self, name, conf_obj, save=True, **kwargs):
        """Top-level initialization method: gets a name and the configuration
        of an object and initializes the object. If ``save`` is set, will
        also check ``_persistence`` whether the object should be saved and if
        yes, will save it using the appropriate persistent label."""
        if name in self.configuration.objects:
            self.init_object(name, conf_obj, **kwargs)
        elif name in self.configuration._assembly:
            self.init_block(name, **kwargs)
        if save and name in self.configuration._persistence:
            label = self.configuration._persistence[name]
            fname = self.get_loading_filename(label)
            print 'Saving object {0} as {1}'.format(name, fname)
            self.objects[name].save(fname)

    def init_object(self, name, conf_obj, **kwargs):
        print 'Initializing object with name: {0}'.format(name)
        locals_names = kwargs
        locals_names.update(self.imports)
        obj = self._execute_init(conf_obj, **locals_names)
        self.objects[name] = obj

    def init_block(self, name, **kwargs):
        locals_names = kwargs
        locals_names.update(self.imports)
        obj = eval(getattr(self._assembly, name), globals(), locals_names)
        self.objects[name] = obj

    def load_object(self, filename, name, conf_obj, **kwargs):
        obj = self._execute_load(filename)
        self.objects[name] = obj

    @staticmethod
    def _execute_import(imported):
        imported_name = imported
        # print 'Importing: {0}'.format(imported)
        try:
            imported_obj = __import__(imported)
            # The imported_obj is the root module (e.g. for
            # safire.utils.transcorp, imported_obj is the safire module and
            # utils.transcorp are respective attributes (that are also of type
            # module)). We need to store with the given module the actual module
            # name corresponding to that module, e.g. if imported was
            # safire.utils.transcorp, we put just 'safire' in the names.
            #  ... this has a problem with multiple 'safire...' imports, right?
            imported_levels = imported.split('.')
            imported_name = imported_levels[0]
            # if len(imported_levels) > 1:
            #     for import_level in imported_levels[1:]:
            #         imported_obj = getattr(imported_obj, import_level)

        # Importing something from a module
        except ImportError:
            modulename, classname = imported.rsplit('.', 1)
            imported_obj = getattr(__import__(modulename, fromlist=[classname]),
                                   classname)
            imported_name = classname
            # print '  Imported class: {0}'.format(imported_obj)
        # print '    object: {0}'.format(imported_obj)
        return imported_name, imported_obj

    @staticmethod
    def _execute_init(obj, **kwargs):
        """Initializes an object based on its configuration.

        Use **kwargs to pass dependencies.
        The kwargs are used as the ``locals`` scope when each non-special
        key-value pair in the ``obj`` configuration dict is evaluated through
        Python's ``eval()``.

        If the configuration object doesn't have a _class special key, raises
        a ValueError.

        >>> obj = {'_class': 'safire.data.vtextcorpus.VTextCorpus', 'precompute_vtlist': 'False', 'pfilter': 'baz(4)'}
        >>> def foo(x): return x + 1
        >>> foo(4)
        5
        >>> vtcorp = ConfigBuilder._execute_init(obj, baz=foo)
        >>> vtcorp.positional_filter_kwargs
        {'k': 5}

        """
        # print 'Initializing object {0}'.format(pprint.pformat(obj))
        current_locals = kwargs
        # Pull in classes used during initialization. Only use them in local
        # context, don't pollute sys.modules!
        ### DEBUG
        #print 'Executing init for object:\n{0}'.format(pprint.pformat(obj))
        if '_import' in obj:
            for imported in obj['_import']:
                # Importing a module:
                imported_name, imported_obj = ConfigBuilder._execute_import(
                    imported)
                current_locals[imported_name] = imported_obj

        if '_init' in obj:
            init_expr = eval(obj['_init'], globals(), current_locals)
        elif '_class' in obj:
            cls_string = obj['_class']
            modulename, classname = cls_string.rsplit('.', 1)
            init_expr = getattr(__import__(modulename, fromlist=[classname]),
                                classname)
        else:
            raise ValueError('Cannot initialize without _class or _init'
                             ' given! (Passed obj: {0})'.format(obj))

        init_args_as_kwargs = {k: eval(v, globals(), current_locals)
                               for k, v in obj.items() if not k.startswith('_')}
        ### DEBUG
        # pprint.pprint((init_expr, init_args_as_kwargs))
        initialized_object = init_expr(**init_args_as_kwargs)
        return initialized_object

    @staticmethod
    def _execute_load(filename):
        """Loads the object saved to the given file. Used when a pipeline block
        or transformer is available according to the _persistence section."""
        obj = gensim.utils.SaveLoad.load(filename)
        return obj

    @staticmethod
    def dict2module(obj, name, no_eval=None, **kwargs):
        """Converts a dictionary into a module with the given name.

        The created module is **not** added to ``sys.modules``.

        The module will have attributes corresponding to the keys in ``obj``.
        Their values will be obtained through calling ``eval()`` on the
        corresponding values in the ``obj``. The kwargs are used as the
        ``locals`` scope when each non-special key-value pair in the ``obj``
        configuration dict is evaluated through Python's ``eval()``.

        Special keys (those that start with an underscore) get no special
        treatment, they become attributes of the output module (including the
        leading underscore).

        >>> def foo(x): return x + 1
        >>> obj = { '_special': 'True', 'nonspecial': 'False', 'size': 'baz(3)'}
        >>> obj_name = 'example'
        >>> m = ConfigBuilder.dict2module(obj, obj_name, baz=foo)
        >>> m._special, m.nonspecial, m.size
        (True, False, 4)
        >>> n = ConfigBuilder.dict2module(obj, obj_name, no_eval='all',  baz=foo)
        >>> n._special, n.nonspecial, n.size
        ('True', 'False', 'baz(3)')
        >>> o = ConfigBuilder.dict2module(obj, obj_name, no_eval='special',  baz=foo)
        >>> o._special, o.nonspecial, o.size
        ('True', False, 4)
        >>> p = ConfigBuilder.dict2module(obj, obj_name, no_eval='nonspecial',  baz=foo)
        >>> p._special, p.nonspecial, p.size
        (True, 'False', 'baz(3)')
        """
        m = imp.new_module(name)
        if no_eval == 'all':
            attributes = obj
        else:
            attributes = ConfigBuilder.eval_dict(obj, no_eval, **kwargs)
        for k, v in attributes.items():
            setattr(m, k, v)
        return m


class ConfigRunner(object):
    """The ConfigRunner gets the code generated by the ConfigBuilder and
    executes it."""
    pass