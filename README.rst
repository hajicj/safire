Copyright: Jan Hajic, jr., 2014
Contact: hajicj@ufal.mff.cuni.cz

This file is part of the Safire library.

Safire is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Safire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Safire.  If not, see <http://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------

The library was originally released as supplementary material for the Master thesis
**Matching Images to Texts** of Jan Hajic jr. from the CEMI multimodal
systems research group at the Institute of Formal and Applied Linguistics
at the Faculty of Mathematics and Physics, Charles University. (The thesis
itself is available under ``misc/hajicj-msc-thesis.pdf``.)

-------------------------------------------------------------------------------


The Safire library
==================

The Safire library is a library for experimenting with deep learning with multimodal datasets. It implements several types of standard neural network layer models like Autoencoders or Restricted Boltzmann Machines and supports a wide range of experimental options.

API documentation is available in HTML the ``doc/build/html/`` subfolder. In addition, each of the scripts in ``scripts/`` can be run with the ``--help`` option that details usage.

All questions, comments, problems, etc. should be thrown, bowled, harpooned or otherwise directed at::

  hajicj@ufal.mff.cuni.cz
  
**A disclaimer: the library is not a mature software product.** Quite some refactoring needs to be done and tests added/fixed. I wouldn't encourage anyone to quite start using it yet. (Developers, on the other hand, are welcome!)

Requirements
------------

  
You will need:

* Python 2.7.6 and higher, NOT Python 3

For installing individual Python libraries, we recommend using the ``pip``
package manager.

You will need the following Python libraries (minimum versions indicated):

* numpy 1.8

* scipy 0.14.0

* matplotlib 1.3.1

* PIL 1.1.7-14 (may or may not be bundled with your Python installation)

* gensim 0.10.0 (see the Errata below on instructions for 0.10.1)

* Theano 0.6.0

Installing Theano is covered at http://deeplearning.net/software/theano/install.html. Generally, standard installation via pip should work. Note that to successfully use Theano, you will need a C/C++ compiler in your ``PATH``. For Windows, the MinGW toolchain can be used; Visual Studio compilers have not been tested.

.. note:: 

  Theano is a bit of a special case; it may not be trivial to set it up. Do not
  despair: we managed to run Theano on a Windows-64bit machine, which is one of
  the more complicated ones. Guides and walkthroughs are available and the theano
  community is very helpful and reacts promptly. Running theano with CUDA GPUs
  should also be possible. **The** ``.theanorc`` **configuration file we used is
  a part of this package, as** ``misc/.theanorc`` **.**

Tips for Windows users
^^^^^^^^^^^^^^^^^^^^^^

We strongly recommend using the Canopy distribution (formerly known as Enthought,
or EPD), which includes numpy, scipy, matplotlib and PIL. The distribution also
comes with a package manager that takes out a lot of the hassle with installing
Python packages. Plus, it comes with the numpy package linked to Intel's MKL
library for fast math.

In addition to the installation instructions at
http://deeplearning.net/software/theano/install.html,
installing Theano on Windows is covered on the project's Github pages:
https://github.com/Theano/Theano/wiki/Windowsinstallation

Lemmatization & POS tagging
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To tag and lemmatize data using the MorphoDiTa tagger, you can either use
the REST API providet by the LINDAT/CLARIN repository, or you will need to
download a local copy of MorphoDiTa.

If you choose to download it, you will need trained models to run it. There
are Czech and English models available for the tagger from the LINDAT/CLARIN
repository:

* Czech: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-68D8-1
 
* English: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-68D9-0  
   

Installation
------------

Currently, a standard ``setuptools`` installation is available. To install,
run ``python setup.py install``. The alternative way to start using Safire
is to put this directory into the ``PYTHONPATH`` environmental variable and
to use the srcipt suite from anywhere on your system, simply add the ``scripts/``
directory to your ``PATH``.

You can run tests using the ``nose`` package. Run ``nosetests`` in this directory.
(Run individually, they should work; when run in a suite, however, they will
probably fail, as some last-minute changes were done in the experiment management
component that caused problems in how the test use shared test data files. This
will get fixed ASAP, as soon as a decent documentation front page is done.)

Getting started
---------------

The file **TUTORIAL.rst** explains what you can do with Safire and how. It will
walk you through creating a small multimodal model from a toy data sample. By the
end, you should be able to understand how to run experiments and inspect results.

For the full scope of script options/capabilities, run the scripts with a ``--help`` option.
  
If you plan to really use Safire for experimentation, be warned that it will start
occupying a lot of disk space very quickly. It is designed to leave traces: every
intermediate step is saved, so that you can re-trace your steps. (Of course, you
can delete things, but it's not meant to be doing that.)


Available data
==============

Unfortunately, we cannot currently give you the full web-pic corpus; there are
some issues that need to be ironed out first. However, we have at least put
together a smaller dataset of about 7000 documents and pictures that we sort of
can let y'all play with.
**Please do not use these images for anything else than checking out the Safire library.**

The dataset is available in the ``../data/safire/`` directory. The directory
``../data/safire/`` acts like the dataset root dir, which is something you will
be using a *lot*. Do **not** manually modify any of the files there (unless you
really know what you're doing); some of the scripts are not as robust to missing
data point IDs and such as we would like them.

Two datasets are available: ``safire`` and ``safire-notabloid``. The latter does
not contain any image-article pairs from tabloid news sites.

For testing, the ``../data/mini-safire`` root with the ``mini-safire`` name is
provided.


Errata
======

* On some Python installations, the PIL library is not available as ``import 
  Image``. It is necessary to use ``from PIL import Image``.  (In: 
  ``safire.utils.__init__``, ``safire.data.image_browser``)
  (Fixed in patch.)
* In Gensim 0.10.1: the  ``matutils.Dense2Corpus`` class doesn't accept ``eps``
  as an ``__init__`` argument; needs to be corrected at line 192 in
  ``safire.learning.interfaces.safire_transformer``
  (Fixed in patch.)

