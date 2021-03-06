"""
This module contains classes that allow creating and using Safire pipelines
from configuration files.

Configuring experiments
=======================

While the basic mechanism for persistence is simply saving and loading the
pipelines themselves, it should be possible to export (and import) experimental
settings without having to write the Python code. Think of the configuration
as pseudocode, which an interpreter will "translate" into Python code.

The complex part of thus interpreting an experiment configuration is the fact
that parts of the pipeline depend on each other. Not only are pipeline blocks
connected to their previous block, some transformers are initialized using the
output of the pipeline on which they are applied, or they use helper objects
which are neither blocks, nor transformers. In other words, there are many
dependencies to resolve when building a pipeline from a configuration.

We can actually think about the configuration as a directed acyclic graph of a
computation. The vertices of the graph are individual objects of the pipeline.
Settings of these objects are like attributes of the vertex. The edges in this
graph are dependencies between these pipeline components. The computation then
consists of initializing each of these objects in the right order, so that all
dependencies of a component that is being initialized have already been created.

How to write a configuration
============================

Safire pipelines can be configured in plain old ``*.ini`` format, with some
safire-specific extras. We'll illustrate on a toy example::

    [_info]
    name='toy_example'

    [_builder]
    save=False
    savename=_info.name

    [foo]
    _class=safire.Foo
    _dependencies=bar
    label='foo_1234'
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
literally to Python code**. Things that do start with an undrscore usually
do as well, but there are a few exceptions in the special sections (see below).

Initializing objects
^^^^^^^^^^^^^^^^^^^^

In the example, we've seen that objects can be initialized through the
``_class`` special attribute. However, that is not the only way to initialize
objects (although it is most intuitive: an object correponds to an instance
of a class, so we initialize it by calling a class's ``__init__``).

Sometimes, we need to call a different function than ``__init__``: look for
example at the test configuration for training
(``test/test-data/test_training_config.ini``), which needs to call
``DenoisingAutoencoder.setup()`` to initialize the ``model_handles`` object.
This is achieved using the ``_init`` special attribute::

    [model_handles]
    _import=safire.learning.models.denoising_autoencoder.DenoisingAutoencoder,theano
    _init=DenoisingAutoencoder.setup
    data=Dataset(_2_)
    n_out=200
    activation=theano.tensor.nnet.sigmoid
    backward_activation=theano.tensor.tanh
    reconstruction='mse'
    heavy_debug=False

Essentially, the ``_class`` mechanism is a special case of ``_init`` when the
function to call is ``__init__()``. The following are equivalent:

* _class=safire.learning.models.denoising_autoencoder.DenoisingAutoencoder
* _init=safire.learning.models.denoising_autoencoder.DenoisingAutoencoder.__init__

A third, yet more general way is available. The ``_exec`` mechanism just tells
the builder to evaluate the given expression and use the result as the
initialized object. From the same file::

    [run_handle]
    _dependencies=model_handles
    _exec=model_handles['run']

This will simply execute the following code::

    >>> run_handle = model_handles['run']

Note that if you initialize something from ``_exec``, it's pointless to use
any initialization attributes. You could theoreticaly do this::

   [foo]
   _exec=bar.func(arg1='xyz', arg2='abc')

instead of::

    [foo]
    _init=bar.func
    arg1='xyz'
    arg2='abc'

Initialization namespace
^^^^^^^^^^^^^^^^^^^^^^^^

You may need to use classes and functions that are not immediately available to
the builder using initialization. Also, you will often use previously
initialized blocks. In order to properly evaluate the initialization statement,
all these names have to be available to the ``eval()`` namespace.

The configuration builder will automatically pass all its special objects and
all the objects that have already been initialized to object initialization.
Also, initialization through ``_class`` automatically imports the required
class. However, in case you need something that is not in this automatically
constructed namespace, you will need to specify the import manually.

Imports on a global basis can be specified in the ``_import`` special attribute
of the ``[_builder]`` section. These names will be included in each local
namespace during object initialization. If you want some names to be available
only to a specific object, you can use the ``_import`` special attribute of that
object's section.

Imports will first attempt to resolve as ``from x.y.z import last`` and if that
fails, as ``import x.y.z.last``. In the first case, the available name will be
``last``. In the second case, you will need to use ``x.y.z.last``. The
``from ... import ...`` pattern works only if ``last`` is a class.

An example::

    [joint_serializer]
    _import=safire.data.sharded_corpus.ShardedCorpus
    _class=safire.data.serializer.Serializer
    corpus=_joint_data_
    serializer_class=ShardedCorpus
    fname=_loader.pipeline_serialization_target('.joint_data')
    overwrite=True

The Serializer needs to specify a class that will be used for writing/accessing
the data. However, this class needs to be imported first, so the section imports
it for itself. The class is then available using ``ShardedCorpus`` only, as
the import was resolved as
``from safire.data.sharded_corpus import ShardedCorpus``. (This is also the
underlying mechanism for ``_class``: first, the given class gets locally
imported and then initialized using the section's attributes as kwargs.)

Assembling pipelines
--------------------

However, while the ``bar`` object is used in initializing ``foo``, this is not
a pipeline -- these are only the building blocks. To assemble a pipeline, we need
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

Filenames
---------

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

Dependencies
------------

The configuration parser and builder don't have the capability to determine
the dependencies between the objects automatically. This has to be done by
specifying the objects on which a transformer depends in a ``_dependencies``
value::

    [baz]
    _class=safire.Baz
    _dependencies=foo,bar
    inputs_1=foo
    inputs_2=bar

(In the future, we might add a Python parser to resolve the dependencies from
the object init args themselves.)

There is also a set of defined special sections that the ConfigBuilder
interprets differently. These all have names that start with an underscore.
Objects corresponding to these sections (most often the ``_loader`` object)
are always initialized before the non-special sections, so you do *not* have
to explicitly mark dependencies on them.

Block dependencies do not need to be declared, as long as the convention of
keeping unique block names (ideally starting and ending with an underscore) is
observed.

Accessing dependencies
^^^^^^^^^^^^^^^^^^^^^^

When a builder is building a pipeline from components that already have been
saved, the dependencies of a persistent component will often be loaded as well.
This is especially true for pipeline blocks that load the entire pipelines
they head. While the objects themselves are loaded implicitly, as they are
attributes (of an attribute of an attribute...) of the persistent object,
the builder needs to have access to them specifically, because other objects
that have yet to be initialized may depend on them, and therefore will need the
builder to supply them in their ``locals()`` initialization namespace.

The rules for accessing dependencies for pipeline blocks are simple: if the name
is a block, then it will be accessed using the ``corpus`` attribute; otherwise,
it will be accessed through the ``obj`` attribute. (This is a conscious design
decision: TransformedCorpus and its dependents should never be created outside
applying a transformer, with ``SwapoutCorpus`` being the notable -- and
temporary -- exception.)

However, rules for accessing dependencies of transformers cannot be written:
transformers are very general and may require various helper objects that may
or may not remain accessible after the transformer is initialized. Because
extracting the access expressions automatically is thus practically impossible,
the access expressions for accessible dependencies must be supplied explicitly.
To this end, there is an ``_access_deps`` special key that complements the
``_dependencies`` special key.



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
all have to be ``cPickle``-able. (This can be helped by implementing your
own ``__getstate__`` and ``__setstate__`` methods.)

Serializing pipelines is a little trickier. Instead of calling a function over
the pipeline, serialization - and efficient reading of serialized data - is
handled using a special :class:`Serializer` transformer. Applying the Serializer
to a pipeline creates a special :class:`SwapoutCorpus` block that reads data
from the Serializer instead of the previous pipeline block. For writing and
reading the serialized data, Safire provides the :class:`ShardedCorpus` class.

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

For control over which stages of the pipeline should be saved,
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
    _1_=.vtcorp
    _2_=.tfidf
    _3_=.serialized_tfidf

Serialization is not handled separately by the configuration mechanism: all you
need to do is define a Serializer block in your config and the data will be
serialized accordingly.

Using saved pipelines in a configuration
----------------------------------------

When a configuration is built, the builder first checks whether a pipeline can
be loaded under the ``savename`` attribute. If yes, it simply loads the
pipeline. This behavior can be disabled using the ``no_default_load=True``
builder setting. [NOT IMPLEMENTED]

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
wish to re-run an experiment with different settings, you'll either have to use
different names, or clear all the saved objects before building the pipeline.
This can be achieved by setting ``clear=True`` in the ``[_builder]`` section.
(The ``run.py`` script that builds a config also has a ``--clear`` flag for
convenience.) This deletes *all* persisted blocks. (Therefore, it may not be
ideal to use when you are changing settings for some of the blocks that only
come later in the pipeline, as you'd have to build the unchanged blocks over
and over. The current suggestion is to split your configuration in two: the
unchanged part, which you do not touch, and the changing part, where you fiddle
with settings and use ``--clear``.)


Artifacts
=========

The things left behind when a Safire configuration is built are called
*artifacts*. There are functions provided to resolve which artifacts will be
created by building a configuration and which artifacts a configuration needs
in order to be built.

[WIP]

Miscellany and caveats
======================

The configuration mechanism in Safire is quite flexible, but by no means
perfect. There are several common use-cases that are not yet implemented
satisfactorily.

Single pipeline split into multiple configs
-------------------------------------------

In complex cases, you may wish to operate with several configurations that
adjoin one another. For instance, a configuration for preprocessing, then one
for training, then one for evaluation. This situation isn't currently handled
very well in Safire, although it is possible to link pipelines from different
configurations manually.

You can extend a previously ``_persist``ed pipeline in another configuration
using the ``_exec`` or ``_init`` mechanisms. Suppose the following is an excerpt
from ``first.ini``::

  [_assembly]
  _1_=SwapoutCorpus(data_source, data_source)
  _2_=transformer[_1_]
  _3_=serializer[_2_]

  [_persistence]
  _3_=.first_transformed

and the following form ``second.ini``:

  [_builder]
  _import=gensim.utils.SaveLoad

  [_assembly]
  _1_=output_of_first
  _2_=transformer[_1_]
  _3_=serializer[_3_]

  [_persistence]
  _3_=.second_retransformed

  [output_of_first]
  _init=SaveLoad.load
  fname=_loader.pipeline_name('.first_transformed')

Now, if we export a pipeline from ``second.ini``::

  >>> # Build the first configuration
  >>> conf1 = ConfigParser().parse('first.ini')
  >>> builder1 = ConfigBuilder(conf1)
  >>> builder1.build()
  >>> # Now build the second configuration and export a pipeline
  >>> conf2 = ConfigParser().parse('second.ini')
  >>> builder = ConfigBuilder(conf2)
  >>> builder.build()
  >>> pipeline = builder.export_pipeline('_3_')

we can, through the recursive ``.corpus`` mechanism of blocks, access the blocks
built and persisted by ``first.ini``::

  >>> isinstance(pipeline.output.corpus.corpus.corpus.corpus, SwapoutCorpus)
  True

although in the pipeline's ``objects`` dict, we can only access objects that are
named in ``second.ini`` -- the blocks built from ``first.ini`` are currently
"hidden" from the pipeline built from ``second.ini``.

More principled linking of pipelines is a feature that will be added.

Also, there is currently no mechanism to keep track of your
serialized data (which label did I use for the preprocessed text..? etc.)


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
import operator
import re
from safire.data.pipeline import Pipeline
from safire.data.serializer import Serializer
import safire.utils.transcorp

__author__ = "Jan Hajic jr."

##############################################################################

###########################
# Some utility functions. #
###########################


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
    #logging.debug('names_in_code: Evaluating code string: {0}'
    #              ''.format(code_string))
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


def depgraph2gvgraph(obj_dependency_graph, format='svg', **graph_kwargs):
    """Creates a dot-print graph of the given dependency graph. Although
    this function doesn't print the graph, it needs to know its output format
    (by default, it's *.svg).

    Depends on the graphviz package, which it tries to import locally;
    if the import fails, logs an error and returns ``None``."""
    try:
        import graphviz as gv
    except ImportError:
        logging.error('Missing package graphviz, cannot draw graph.')
        return None

    graph = gv.Digraph(format=format, **graph_kwargs)
    for obj in obj_dependency_graph:
        graph.node(obj, label=obj)  # Attributes?

    for obj, deps in obj_dependency_graph.items():
        for dep in deps:
            graph.edge(dep, obj)

    return graph


def build_pipeline(config_file, output_name):
    """Shortcut function for getting a pipeline from a config file.

    :param config_file: The configuration file from which you want to get the
        pipeline.

    :param output_name: The name of the desired pipeline's output block

    :returns: Pipeline with the given block as the output.
    """
    cparser = ConfigParser()
    conf = cparser.parse(config_file)
    cbuilder = ConfigBuilder(conf)
    cbuilder.build()
    pipeline = cbuilder.export_pipeline(output_name)
    return pipeline

# Dealing with artifacts (stuff that configurations leave behind).
# Important for keeping track of what is available where.


def _pln_artifacts(builder):
    """Inside function to extract all pipeline block artifacts from a
    ConfigBuilder. The public interface is to use :func:`pln_artifacts`
    and call it with a configuration (that function initializes the builder
    and calls this function)."""
    plns = {}
    for obj, label in builder.configuration._persistence.items():
        if obj != 'loader' and not obj.startswith('__'):
            fname = builder.get_loading_filename(label)
            plns[obj] = fname
    return plns


def pln_artifacts(conf):
    """Extract all ``*.pln`` artifact names created by the configuration.
    (Assumes that all artifacts are created through the ``_persistence``
    mechanism -- which they should be!)"""
    builder = ConfigBuilder(conf)
    return _pln_artifacts(builder)


def _pls_artifacts(builder):
    """Extract all ``*.pls`` artifact names from a ConfigBuilder. The public
    interface is to use :func:`pls_artifacts`  and call it with a configuration
    (that function initializes the builder and calls this function).

    Currently only recognizes serializer objs that are subclasses of
    :cls:`safire.data.serializer.Serializer` and have a ``fname`` kwarg that
    defines the ``*.pls`` filename.
    """
    plss = {}

    known_serializer_classes = [Serializer]

    serializer_conf_objs = {}

    # Get all Serializer-class objects.
    for conf_obj_name, conf_obj in builder.configuration.objects.items():
        if '_class' in conf_obj:
            _, cls = builder._execute_import(conf_obj['_class'])
            for serializer_cls in known_serializer_classes:
                if issubclass(cls, serializer_cls):
                    serializer_conf_objs[conf_obj_name] = conf_obj

    # Check each serializer conf obj for serialization target (fname attribute)
    for conf_obj_name, conf_obj in serializer_conf_objs.items():
        fname_attr = conf_obj['fname']  # Base class Serializer always has that
        fname = eval(fname_attr, globals(), builder.objects)
        # Possible problem with locals -- we may need to take _import etc.
        # into account.
        # TODO: The eval should fail gracefully.
        plss[conf_obj_name] = fname

    return plss


def pls_artifacts(conf):
    """Extract all ``*.pls`` artifact names created by the configuration.
    These names correspond to calls of ``_loader.pipeline_serialization_target``
    that are used in Serializer-class blocks during configuration building.

    Note that while Serializer objects can be initialized through various object
    creation mechanisms (``_class``, ``_init``, ``_exec``), only
    ``_class``-based serializers will be discovered.
    """
    builder = ConfigBuilder(conf)
    return _pls_artifacts(builder)


def artifacts(conf):
    """Finds out which *artifacts* the given configuration leaves behind when
    it is fully built.

    There are two types of artifacts: saved pipelines (``*.pln`` files) and
    serialized data (``*.pls`` files). The individual shard files are assumed
    to be under transparent control by the corresponding ``*.pls`` file, the
    one to which the serializer class instance itself is saved.

    The function returns a tuple of lists of artifact filenames.

    .. note:

        This function is the first step towards an indexing system.

    To check for artifacts, the function uses the configuration's loader,

    :type conf: safire.utils.config.Configuration
    :param conf: The configuration to get artifacts from.

    :rtype: tuple(dict(str => str), dict(str => str))
    :returns: A tuple ``(plns, plss)`` of dicts of saved pipeline names and
        saved serialization targets. The pipeline savenames are labeled
        according to the ``_persistence`` keys, the serialization targets are
        named according to their respective Serializer *transformer* conf
        sections.
    """
    # Pre-build the configuration to get its loader and dependency graph.
    builder = ConfigBuilder(conf)

    plns = _pln_artifacts(builder)  # Pipeline save objects
    plss = _pls_artifacts(builder)  # Pipeline serialization objects

    return plns, plss


def list_artifacts(confs):
    """Gather all artifacts of the given list of configurations. Returns
    plns/plss dicts like the function :func:`artifacts`.

    :param confs: A list of Configurations.
    """
    plns = dict()
    plss = dict()

    for conf in confs:
        c_plns, c_plss = artifacts(conf)
        plns.update(c_plns)
        plss.update(c_plss)

    return plns, plss


def required_artifacts(conf):
    """This function detects what artifacts a configuration will require to be
    successfully built. Due to the flexibility of Safire, this
    detection is far from perfect: it can only detect stuff that gets loaded
    from the ``_loader`` attribute using :meth:`pipeline_name` or
    :meth:`pipeline_serialization_target`. Also, the discovery cannot currently
    deal with parentheses inside the ``pipeline_name`` and
    ``pipeline_serialization_target`` calls and cannot deal with more than one
    call per object key/value.

    :returns: A tuple of ``(plns, plss)`` dicts. The dict keys are the artifact
        filenames, the values are the object names that need the artifact.
    """
    builder = ConfigBuilder(conf)

    plns_regex = re.compile('_loader\.pipeline_name\([^\)].*\)')
    plss_regex = re.compile('_loader\.pipeline_serialization_target\([^\)].*\)')

    req_plns = dict()
    req_plss = dict()

    for c_obj_name, c_obj in conf.objects:
        for key, value in c_obj.items():
            pln_match = plns_regex.search(value)
            if pln_match:
                artifact_expr = pln_match.group(0)
                fname = eval(artifact_expr, globals(), builder.objects)
                req_plns[fname] = c_obj_name

            pls_match = plss_regex.search(value)
            if pls_match:
                artifact_expr = pls_match.group(0)
                fname = eval(artifact_expr, globals(), builder.objects)
                req_plss[fname] = c_obj_name

    return req_plns, req_plss

###############################################################################

##########################
# Configuration classes. #
##########################


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
        """Given a config file (handle), will build the Configuration object.
        If a string is passed, will interpret it as the filename."""
        if isinstance(file, str):
            with open(file) as conf_handle:
                output = self.parse(conf_handle)
            return output


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
                        #print 'Access deps for name {0}: {1}'.format(current_section_name, value)

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

    Builder options
    ---------------

    The ConfigBuilder has some settings that control its behavior. These can be
    set through the ``[_builder]`` section of the configuration file. Currently,
    the following settings are recognized:

    * ``_import``: Gives a list of modules, classes or functions to import into
      the local environment that the builder uses. (This section may be
      superseded in the future by a separate ``[_import]`` section.)
    * ``no_loading``: If set, will ignore all load-able objects during building.
    * ``clear``: If set, will clear all load-able objects before building. This
      should guarantee that the entire pipeline will be re-built from scratch.

    Other options will be added as necessary.
    """
    def __init__(self, conf):
        # Defaults
        self.no_loading = False
        self.clear = False
        self._import = []

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
        # is requested, it is taken from here. This dict also gets always
        # passed as the locals() environment to eval() calls.
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

    def build(self):
        """This method does the heavy lifting: determining dependencies, dealing
        with persistence, etc. It outputs a dictionary of all objects (blocks
        and non-blocks) that do not depend on anything.
        """
        if hasattr(self, 'clear') and self.clear is True:
            logging.info('Clearing all saved objects...')
            self.clear_saved()

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
            logging.info('========== Loading item: {0}\t'
                         '=========='.format(item))
            persistent_label = getattr(self._persistence, item)
            logging.info('   Persistent label: {0}'.format(persistent_label))
            fname = self.get_loading_filename(persistent_label)
            obj = self._execute_load(fname)
            #  - add item to object dict
            self.objects[item] = obj

            #  - traverse dependencies according to _access_deps. This time, it
            #    means iterating over the actual objects, as they have been
            #    loaded. Add each of these to the object dict.
            obj_queue = [(item, obj)]
            while obj_queue:
                logging.debug('Current queue:\n{0}'
                              ''.format(pprint.pformat(map(operator.itemgetter(0), obj_queue))))
                current_symbol, current_obj = obj_queue.pop()
                logging.debug('Current symbol: {0}, obj: {1}'
                              ''.format(current_symbol, current_obj))

                # Add to objects dict.
                self.objects[current_symbol] = current_obj

                # Traverse dependency graph further
                current_dep_symbols = self.get_dep_symbols(current_symbol)
                current_dep_objs = {dep_symbol: self.retrieve_dep_obj(current_symbol,
                                                                  current_obj,
                                                                  dep_symbol)
                                    for dep_symbol in current_dep_symbols
                                    if self.is_dep_obj_accessible(current_symbol,
                                                                  dep_symbol)
                                    and dep_symbol is not None}
                logging.debug('Extending dep_objs queue: {0}'.format(current_dep_objs.keys()))
                obj_queue.extend(current_dep_objs.items())

        # Verify that everything that should have been loaded has been loaded
        for item in will_be_loaded:
            if item not in self.objects:
                logging.warn('Object {0} not found after it should have been'
                             ' loaded.' #(Available objects: {1})'
                             ''.format(item,
                             #          pprint.pformat(self.objects.keys())
                             )
                )

        # Initialize objects that weren't loaded.
        # Note that the ordering needs to be re-computed.
        to_init = [item for item in self.sorted_deps if item in will_init]
        for item in to_init:
            if self.is_block(item):
                logging.info('========== Initializing block: {0}\t'
                             '=========='.format(item))
                self.init_block(item, **self.objects)
            else:
                logging.info('========== Initializing object: {0}\t'
                             '=========='.format(item))
                self.init_object(item,
                                 self.configuration.objects[item],
                                 **self.objects)
            logging.debug('Initialized:\n{0}'
                          ''.format(safire.utils.transcorp.log_corpus_stack(self.objects[item])))
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

    def export_pipeline(self, name):
        """Creates a :class:`safire.data.pipeline.Pipeline` with the output
        block ``name`` and the pruned depgraph/sorted depgraph/objects
        dictionary."""
        if not self.is_block(name):
            raise ValueError('Specified pipeline output {0} is not a block.'
                             ''.format(name))

        output = self.objects[name]

        pruned_depgraph = {name: self.deps_graph[name]}
        depqueue = list(self.deps_graph[name])
        while depqueue:
            current_dep = depqueue.pop()
            deps_to_add = self.deps_graph[current_dep]
            pruned_depgraph[current_dep] = deps_to_add
            depqueue.extend(deps_to_add)

        pruned_sortedgraph = collections.OrderedDict()
        for depname in self.sorted_deps:
            if depname in pruned_depgraph:
                pruned_sortedgraph[depname] = self.sorted_deps[depname]

        pruned_objects = {name: self.objects[name]}
        for depname in pruned_depgraph:
            pruned_objects[depname] = self.objects[depname]

        pipeline = Pipeline(output,
                            pruned_objects,
                            pruned_depgraph,
                            pruned_sortedgraph)
        return pipeline

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

    def get_dependency_graph_drawing(self, format='svg', **graph_kwargs):
        """Creates the dependency graph of the configuration; doesn't render it.
        """
        # graph = depgraph2gvgraph(self.deps_graph, format=format, **graph_kwargs)
        try:
            import graphviz as gv
        except ImportError:
            logging.error('Could not import graphviz, exiting.')
            return None

        # Drawing style(s)
        block_node_attrs = {'style': 'filled',
                            'fillcolor': '#ffb46e',
                            'shape': 'rectangle'}
        noblock_node_attrs = {'style': 'filled',
                              'fillcolor': '#eeffee'}
        dep_edge_attrs = {}
        access_dep_edge_attrs = {'style': 'dashed',
                                 'color': '#bbbbbb',
                                 'fontcolor': '#bbbbbb'}

        will_load_updates = {'color': 'red',
                             'fillcolor': '#aaffaa'}
        block_will_be_loaded_updates = {'fillcolor': '#cccccc',
                                        'color': block_node_attrs['fillcolor']}
        noblock_will_be_loaded_updates = {'fillcolor': '#cccccc',
                                          'color': '#aaaaaa'}
        will_init_updates = {}

        will_load, will_be_loaded, will_init = self.resolve_persistence()

        graph = gv.Digraph(format=format, **graph_kwargs)
        for name in self.deps_graph:
            if self.is_block(name):
                node_attrs = copy.deepcopy(block_node_attrs)
                if name in will_load:
                    node_attrs.update(will_load_updates)
                elif name in will_be_loaded:
                    node_attrs.update(block_will_be_loaded_updates)
                elif name in will_init:
                    node_attrs.update(will_init_updates)
                graph.node(name, label=name, **node_attrs)
            else:
                node_attrs = copy.deepcopy(noblock_node_attrs)
                if name in will_load:
                    node_attrs.update(will_load_updates)
                elif name in will_be_loaded:
                    node_attrs.update(noblock_will_be_loaded_updates)
                elif name in will_init:
                    node_attrs.update(will_init_updates)
                graph.node(name, label=name, **node_attrs)

        for obj, deps in self.deps_graph.items():
            for dep in deps:
                graph.edge(dep, obj, **dep_edge_attrs)
            if obj in self.configuration.objects \
                    and '_access_deps' in self.configuration.objects[obj]:
                # print 'Access deps found on object {0} with deps {1}'.format(obj, deps)
                access_deps = self.configuration.objects[obj]['_access_deps']
                # print 'Access deps: {0}'.format(access_deps)
                for a_dep, access in access_deps.items():
                    # print 'Drawing access_deps edge {0} -> {1}'.format(obj, a_dep)
                    graph.edge(obj, a_dep, label=access,
                               **access_dep_edge_attrs)
        return graph

    def draw_dependency_graph(self, filename, format='svg', **graph_kwargs):
        """Draws the dependency graph of the configuration to the given file."""
        # Default graph style
        graph_style = {'label': '\nConfiguration dependency graph for: '
                                + self._info.name,
                       'fontsize': str(len(self.deps_graph) / 2),
                       'font': 'Helvetica',
                       }
        if 'graph_attr' in graph_kwargs:
            graph_kwargs['graph_attr'].update({k: v
                                               for k, v in graph_style.items()
                                               if k not in graph_kwargs['graph_attr']})
        else:
            graph_kwargs['graph_attr'] = graph_style

        graph = self.get_dependency_graph_drawing(format=format, **graph_kwargs)
        if graph is None:
            logging.error('Could not import graphviz, drawing failed.')

        graph.render(filename)

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
            if not self.no_loading and self.can_load(item):
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
            # Traverse all accessible dependencies of current object and mark
            # them as "to be loaded"
            logging.debug('Current object that will be loaded: {0}'.format(current_obj))
            # This should also be accessible deps only
            deps = self.get_accesible_objects(current_obj)
            while len(deps) > 0:
                current_dep = deps.pop()
                if current_dep not in to_resolve:
                    continue
                to_resolve.remove(current_dep)
                will_be_loaded.add(current_dep)
                accessible_deps = self.get_accesible_objects(current_dep)
                deps.update(accessible_deps)
        # Whatever's left will be initialized, as it is unreachable from
        # the load-able objects.
        will_init = to_resolve

        logging.debug('Working with configuration: {0}'.format(self._info.name))
        logging.debug('  Will load: {0}'.format(pprint.pformat(will_load)))
        logging.debug('  Will be loaded transitively: {0}'.format(pprint.pformat(will_be_loaded)))
        logging.debug('  Will initialize: {0}'.format(pprint.pformat(will_init)))

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
            if symbol in names_in_code(symbol_conf):
                return True
        else:
            conf = self.configuration.objects[name]
            if '_access_deps' in conf:
                if symbol in conf['_access_deps']:  # This is a dict: no parsing
                    return True
            # Currently disabled, TODO: determine correctness
            # if self._uses_block(name, symbol):  # This is not always true!
            #     return True
        return False

    def get_accesible_objects(self, name):
        """Returns a set of names of objects that should be accessible from the
        object with the given name."""
        output = set(dep for dep in self.deps_graph[name]
                     if self.is_dep_obj_accessible(name, dep))
        return output

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
        the ``obj`` with the ``name``.

        With blocks: the requested object can be either the ``corpus`` member
        of a TransformedCorpus, or an ``obj`` member.
        """
        if not self.is_dep_obj_accessible(name, symbol):
            raise ValueError('Dependency {0} not accessible from object {1}.'.format(symbol, name))
        # If obj is a block, the accessible object can be either the previous
        # block, or the block's transformer.
        logging.debug('Retrieving dep symbol {0}\n\tfrom obj {1}\n'
                      '\twith name {2}'.format(symbol, obj, name))
        # TODO: Problem with CompositeCorpus around here.
        # Currently solved by ugly hack that checks for A=B pattern. Should be
        # resolved through a better _assembly parsing mechanism, or a different
        # way of assembling pipelines.
        #       -- when _combined_ is delegated to [combined], the dependency
        #          symbol for the dependency of _combined_ ('combined') is
        #          never found, because it doesn't stick to the corpus/obj
        #          pattern. The problem is in the _combined_=combined pattern:
        #          while _combined_ is a block, in reality, it is the [combined]
        #          obj and _access_deps should be applied.
        if self.is_block(name):
            logging.debug('Assuming the current obj is a block.')
            #  -- quick fix: try to detect '_a_=b' assignments and move along
            #     the dependency chain
            logging.debug('Detecting a=b direct assignment pattern in assembly.')
            assembly_code = self.configuration._assembly[name]
            logging.debug('  Assembly code: {0}'.format(assembly_code))
            pure_assignment_pattern = re.compile('^([^\[]*)$')
            # If the block was assigned through a straightforward a=b scheme
            match = re.match(pure_assignment_pattern, assembly_code)
            if match:
                logging.debug('  Match detected! Assigned name: {0}'.format(match.group(1)))
                assigned_name = match.group(1)
                _hook_active = True
                if assigned_name not in self.configuration.objects:
                    logging.debug('  Assigned name {0} not found among '
                                  'configuration objects, checking if it is at '
                                  'least a block.'.format(assigned_name))
                    if assigned_name not in self.configuration._assembly:
                        logging.debug('  Assigned name not found among block '
                                      'names, either. Assignment cannot be'
                                      ' interpreted by this hook, proceeding '
                                      'as though nothing was detected.')
                        _hook_active = False
                # ...we should search for the dependency in the assigned
                if _hook_active:
                    logging.debug('Assuming the dependency we were trying to find'
                                  ' was caused by the simple assignment in the'
                                  ' _assembly section of the form A=B, so the'
                                  ' current object under name A is requesting'
                                  ' a dependency called B that is in fact the'
                                  ' same object as the one under name A.')
                    dep_obj = obj
                    return dep_obj
            else:
                logging.debug('  Assembly code not a simple assignment, '
                              'proceeding without hook.')

            if self.is_block(symbol):
                dep_obj = obj.corpus
            else:
                dep_obj = obj.obj
        else:
            obj_conf = self.configuration.objects[name]
            if '_access_deps' not in obj_conf:
                return None
            access_expr = obj_conf['_access_deps'][symbol]
            dep_obj = eval(access_expr, globals(), {name: obj})
        return dep_obj

    def run_saving(self):
        """Saves the built objects according to the persistence instruction."""
        for obj_name, obj_label in self.configuration._persistence.items():
            if obj_name == 'loader' and 'loader' not in self.objects:
                continue
            logging.info('Running save for obj_name {0}, label {1}'
                         ''.format(obj_name, obj_label))
            fname = self.get_loading_filename(obj_label)
            # print 'Saving object {0} to file {1}'.format(obj_name, fname)
            self.objects[obj_name].save(fname)

    def clear_saved(self):
        """Clears the saved objects according to the persistence
        instruction."""
        for obj_name, obj_label in self.configuration._persistence.items():
            if obj_name == 'loader' and 'loader' not in self.objects:
                continue
            fname = self.get_loading_filename(obj_label)
            # print 'Saving object {0} to file {1}'.format(obj_name, fname)
            if os.path.isfile(fname):
                logging.info('Removing file {0}...'.format(fname))
                os.remove(fname)
            else:
                logging.info('Cannot remove file {0}: not found!'.format(fname))

    def clear_saved_obj(self, obj_name):
        """Clears the selected saved object. Note that this does *not* clear
        serialization (yet)."""
        persistence = self.configuration._persistence
        if obj_name not in persistence:
            raise ValueError('Object with the name {0} was never saved.'
                             ''.format(obj_name))
        obj_label = persistence[obj_name]
        fname = self.get_loading_filename(obj_label)
        if os.path.isfile(fname):
            logging.info('Removing file {0} (obj name: {1}, obj label: {2})...'
                         ''.format(fname, obj_name, obj_label))
            os.remove(fname)
        else:
            logging.info('Cannot remove file {0}: not found! (obj name: {1},'
                         'obj label: {2})'.format(fname, obj_name, obj_label))

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
            logging.debug('Saving object {0} as {1}'.format(name, fname))
            self.objects[name].save(fname)

    def init_object(self, name, conf_obj, **kwargs):
        # print 'Initializing object with name: {0}'.format(name)
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
        logging.debug('Executing import: {0}'.format(imported))
        # print 'Executing import: {0}'.format(imported)
        try:
            imported_obj = __import__(imported)
            # The imported_obj is the root module (e.g. for
            # safire.utils.transcorp, imported_obj is the safire module and
            # utils.transcorp are respective attributes (that are also of type
            # module)). We need to store with the given module the actual module
            # name corresponding to that module, e.g. if imported was
            # safire.utils.transcorp, we put just 'safire' in the names.
            #  ... this has a problem with multiple 'safire...' imports, right?
            #  ... wrong: the various safire-based modules get added to the
            #      safire namespace, so they are both accessible through
            #      safire.foo/safire.bar
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
        # print 'Executing init for object:\n{0}'.format(pprint.pformat(obj))
        if '_import' in obj:
            for imported in obj['_import']:
                # Importing a module:
                imported_name, imported_obj = ConfigBuilder._execute_import(
                    imported)
                current_locals[imported_name] = imported_obj
        if '_exec' in obj:
            obj = eval(obj['_exec'], globals(), current_locals)
            return obj
        if '_init' in obj:
            init_expr = eval(obj['_init'], globals(), current_locals)
        elif '_class' in obj:
            cls_string = obj['_class']
            modulename, classname = cls_string.rsplit('.', 1)
            init_expr = getattr(__import__(modulename, fromlist=[classname]),
                                classname)
        else:
            raise ValueError('Cannot initialize without _class, _init or _exec'
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
