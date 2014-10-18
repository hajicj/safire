.. Safire documentation master file, created by
   sphinx-quickstart on Wed Jul 30 17:53:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SAFIRE: Multimodal Deep Learning
==================================

The Safire library is a Python deep learning experimentation library created
for multimodal experiments. It has two major funcitonalities: the deep learning
component in the ``learning`` module, and the experiment management components
in the rest of the library, especially in the ``data`` module.

It is built around the Theano library for speed and the gensim
library for experiment management. Apart from the library modules, there
is the ``scripts`` component which uses the library to run and evaluate
experiments.

.. warning::

  This library is NOT a mature software product; I do not recommend anyone to
  use it in their research yet. With hindsight, there are multiple suboptimal
  design decisions and refactoring will occur.

.. warning::

  While I'm not looking for users, I'm certainly looking for developers. :)

Architecture
============

We will describe the architecture from the inside out, starting from the deep
learning models and working outwards through datasets and filters to loaders
and layouts. This is the direction in which the library grew.

.. note::

  As I learned how to work with Theano, what seemed to be like a good design
  decision in the beginning turned into bad design decisions. Changes that I
  believe should be made are marked in a **TODO** box in the appropriate place.
  Of course, should additional developers arise, these things are up for
  discussion.

Models
------

At the heart of a deep learning experiment lies a *model*. A model is just a set
of equations: it is a *definition* of what should be trained. The most important
part of a model is a cost function to optimize and the dimension of input and
output. Other methods can be define how to sample from the model.

All model classes are derived from the :class:`BaseModel` class. This class
defines the common interface and implements common functionality for random
parameter initialization and persistence.

Models fall into two broad categories: *supervised* and *unsupervised*. The
difference is minimal: while supervised models define their cost function with
respect to some input and output variables, unsupervised models define their
cost function with respect to inputs only. This means that the output dimension
of an unsupervised model has to be specified explicitly, while for a supervised
model it can (and should!) be derived from the dimension of the output.

A base class for each model category is also implemented. As opposed to
:class:`BaseModel`, these classes already implement important functionality:
the ``setup()`` class method. **This is the preferred method for creating new
models.** This method does not directly return a model instance, as calling
``__init__`` would. Instead, it returns a *model handle*, which is the subject
of the next section.

Model classes also define their parameter updates for training. These updates
are used during setup and merged together with updates coming from the supplied
``updater`` class.

.. note::

  **TODO** Because parameters of a model are public instance attributes, the whole
  parameter update business should be moved to an Updater, outside the model.

.. note::

  **TODO** It is still questionable whether the model should have the init/setup
  dichotomy. Maybe the setup functionality should be moved to handle
  initialization.

  If we think about the CD problem: what if we relaxed the requirement that all
  models had a differentiable cost function? This is a more general question:
  *how much should the model class know about its training?*

Model handles
-------------

The model classes only implements a definition. In order to *use* a model for
something, a *model handle* has to be instantiated. Model handle is a way of
making the model do something. For example, the base :class:`ModelHandle` class
provides a method for training, validating and running a feedforward network on
a dataset. Other handles are available that perform backward sampling and
clamped sampling of a joint multimodal layer. Generally, the "action" of
a handle is defined by its ``run()`` method.

Handles can be *cloned*: if you need to use the model differently, for instance
you want to sample backwards from it after training it in a feedforward manner,
you can initialize a handle that does just that by cloning the existing handle.

As a rule of thumb, models are not intended to be initialized directly. Rather,
a handle should be created using the ``setup`` method of the model class.

.. note::

  **TODO** Could various methods of training, e.g. Contrastive Divergence, be
  delegated to handles?


Updaters
--------

Updater classes are a way of separating the parameter update equations for
training from the model class. Updaters are stuck between a rock and a hard
place: they should implement things independently on underlying models (i.e.:
momentum can be used for any model), but they still need access to the model's
parameters and gradients to get the updates.

The updater is used during the model's ``setup()`` classmethod. It uses a
modified visitor pattern: it gets passed to the model's ``_training_updates()`` method,
where the model computes its gradients and passes them and their respective
model parameters to the updater. The updater then creates the dictionary of
updates that should be passed to ``theano.function()`` as the ``updates``
argument when compiling the training function.

The updater should hold all the shared variables that are used during training
but not a part of the model, like the previous velocities for each parameter
when using momentum, or the current step matrices during resillient backprop.

.. note::

   **TODO** I haven't figured out yet how to get updaters to actually do this
   last thing properly; some kind of Theano magic is involved there.


Learners
--------

Learner classes are the place where training happens. They take a model handle
and a dataset and control the training process. They do *not* control the training
algorithm. They train models only through calling a handle's ``train()`` method
(and ``validate()``), monitor progress, decide when to stop, track parameter
changes, etc. Learners are the "managers" of training.

.. note::

   Keeping the training algorithm hidden from the learner means that
   the name :class:`BaseSGDLearner`` is wrong; the learner has no idea whether
   it is really training the model using stochastic gradient descent.

.. note::

   **TODO** It needs to be determined whether having a learner agnostic to the
   training algorithm is a good thing. For example, what about monitoring
   the auxilliary variables for more complex training algorithms like rprop?
   Maybe updater functionality should somehow be merged into an inheritance
   hierarchy of learners. This would complicate ``setup()``, as we couldn't say
   at setup-time what the actual updates are. On the other hand, maybe this is
   a way of getting rid of ``setup()`` completely: delegate the *entire*
   training setup to learner initialization. Which would break the idea
   of handles?

Datasets
--------

Datasets are classes that provide input for a model. Datasets return *batches*:
sets of training examples of a given size. Each batch is a dense matrix.
The dataset internally keeps a train-devel-test split (which is, of course,
persistent), so it can answer queries like "give me the 88th training batch of
size 1000".

Datasets are also split into supervised and unsupervised. Unsupervised datasets
can only respond to queries for input variables, supervised datasets can provide
both input and response variables.

There is a somewhat complex hierarchy of datasets available in safire, in order
to cover all types of modality queries. The pinnacle of this hierarchy is the
:class:`ShardedMultimodalDataset`` class, which provides access to any
combination of text and image data as input or response, constant memory usage,
simple caching, access to underlying information about the text and image data
and of course persistence.

Datasets are not gensim-style corpora: they have to support random access,
in order to be able to randomize training example order. However, some of the
more high-level datasets *contain* corpora in order to answer queries like
"what are the IDs of the texts I'm processing". These capabilities are added to
enable more in-depth tracking of the experiments; primarily during evaluation,
rather than during training.

.. note::

  **TODO** Support using sparse matrices in models.



Models provide

.. toctree::
   :maxdepth: 2

   Safire <safire>
   Scripts <scripts>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

